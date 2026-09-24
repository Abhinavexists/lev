"""Modal app: train and serve lev on one H100.

    modal setup                                   # once
    modal run   modal/app.py::smoke               # ~5 min, proves the path works
    modal run   modal/app.py::train --preset 4b   # the real run, ~2h
    modal serve modal/app.py                      # /v1/systemone on an H100, dev URL
    modal deploy modal/app.py                     # ...with a stable URL

`modal serve` and `modal run` are both *ephemeral* apps, and every ephemeral
app of this file registers the `serve` web function under the same `-dev`
label. Running any function here -- an export, a diagnostic -- while a dev
server is up steals that label, and the server's URL returns 404 once the run
exits. Serve with `modal deploy` for a URL that other runs cannot take.

Design notes worth knowing before you change anything here:

* **The model cache is a Volume, not part of the image.** A 4B checkpoint is ~8 GB;
  baking it into the image makes every rebuild slow and every push enormous.
* **Checkpoints go to a second Volume and are committed *during* training**, not at
  the end. A run that loses everything to a preemption is a bad trade for the two
  lines it takes to commit periodically.
* **`smoke` exists to be run first.** It uses the 0.8B preset and a few dozen
  steps, so the whole path -- image, volumes, data, both readouts, loss,
  checkpoint write -- is exercised for a few minutes of GPU time instead of
  discovering a bug most of the way through the real run.
* **H100 timeout is 24h against a ~2h estimate.** Deliberately generous: the
  estimate assumes a sustained throughput that has not been measured yet
  (DECISIONS.md Q5), and a run that dies just before it saves costs everything.

PARTLY RUN ON MODAL. The image builds, all six functions register, and
`download` has run green -- it wrote the HF cache layout that `train` reads
(`models--Qwen--Qwen3.5-4B-Base/snapshots/...` in the models Volume). Still
unrun remotely: `build_data`, `smoke`, `train`, `calibrate`, `serve`. The
pipeline underneath them is verified end to end on CPU (docs/TRAINING.md,
"What has actually been run"). Run `smoke` next.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

APP_NAME = "lev"

# Resolved from this file, not from the working directory: `modal run` may be
# invoked from anywhere, and a relative path would silently mount nothing.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Pinned rather than floating: an unpinned torch or transformers turns a
# reproducible run into a lottery. `transformers` must be 5.x -- the prefix-cache
# fork calls `reorder_cache` on a hybrid cache, which 4.x cannot do.
image = (
    # A CUDA *devel* base, not `debian_slim`: it carries `nvcc`, which is what
    # `causal-conv1d` needs to build. Without that kernel 24 of the 32 layers
    # run their depthwise conv on a reference PyTorch path -- every training and
    # serving log said so -- and the served forward measured ~160 ms flat for a
    # 4B model on an H100, launch-bound rather than FLOP-bound. The CUDA major
    # must match torch's build (2.14.0+cu130).
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
    .pip_install(
        "torch==2.14.0",
        "transformers==5.17.0",
        "accelerate==1.15.0",
        "peft==0.21.0",
        "datasets==5.0.1",
        "safetensors==0.8.0",
        "pydantic==2.13.5",
        "fastapi==0.141.1",
        "huggingface-hub==1.32.0",
    )
    # Speed, not correctness: the linear-attention kernels (pure Triton) and the
    # depthwise conv kernel (CUDA, built here against the torch above). If the
    # conv build breaks on a pin change, delete that line -- the run gets slow,
    # not wrong, and says so in its log.
    .pip_install("flash-linear-attention", "ninja", "packaging")
    # The devel image ships nvcc but no host C++ compiler, and torch's
    # extension builder refuses to proceed without one ("clang++ 0.0.0").
    # Arch 9.0 only: this image runs on H100s, and compiling every arch takes
    # several times longer for nothing.
    .apt_install("build-essential")
    .env({"CC": "gcc", "CXX": "g++", "TORCH_CUDA_ARCH_LIST": "9.0", "MAX_JOBS": "8"})
    # Length-bucketed batches vary widely in shape; without this the caching
    # allocator fragments (the OOM at step 6,075 had 29 GiB reserved-but-free).
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .pip_install("causal-conv1d", extra_options="--no-build-isolation")
    .add_local_dir(
        REPO_ROOT / "packages" / "lev" / "src" / "lev",
        remote_path="/root/lev",
        ignore=["**/__pycache__", "**/*.pyc"],
    )
)

app = modal.App(APP_NAME, image=image)

# Persisted across runs. Downloading a 4B checkpoint on every invocation is
# minutes of wall clock and bandwidth you pay for.
models = modal.Volume.from_name("lev-models", create_if_missing=True)
checkpoints = modal.Volume.from_name("lev-checkpoints", create_if_missing=True)
datasets_vol = modal.Volume.from_name("lev-data", create_if_missing=True)

MODELS_DIR = "/models"
CKPT_DIR = "/checkpoints"
DATA_DIR = "/data"

VOLUMES = {MODELS_DIR: models, CKPT_DIR: checkpoints, DATA_DIR: datasets_vol}

# Which preset `serve` exposes. Not a function argument: Modal requires
# `@modal.asgi_app` functions to take none.
SERVE_PRESET = "4b"


def _hf_secrets() -> list:
    """A token only if there is one to pass.

    The default backbone is public, so most runs need no credential at all.
    `Secret.from_name("huggingface")` is resolved when the function is created
    and fails outright if no such Secret exists -- which made `modal run` fail
    on a fresh account for a backbone that needs no token. Reading the local
    `HF_TOKEN` instead means the common case works untouched and a gated
    backbone works by exporting one variable.

    Prefer a real Modal Secret for a shared or long-lived deployment:
        modal secret create huggingface HF_TOKEN=hf_...
    then set LEV_HF_SECRET=huggingface.
    """
    named = os.environ.get("LEV_HF_SECRET")
    if named:
        return [modal.Secret.from_name(named, required_keys=[])]
    token = os.environ.get("HF_TOKEN")
    return [modal.Secret.from_dict({"HF_TOKEN": token})] if token else []


# Deploy-time knobs, read where `modal deploy` runs. Decorator arguments are
# fixed at import, so these cannot travel as Secrets the way the preset does.
#   LEV_SERVE_CONCURRENCY  requests one container handles at once (default 4).
#                          The engine serialises GPU forwards; everything else
#                          -- parsing, tokenising, the network -- overlaps.
#   LEV_SERVE_WARM         containers kept running (default 0). One removes the
#                          20-55 s cold start and the compile warmup at the cost
#                          of an idle GPU.
#   LEV_SERVE_REGION       a Modal region near the client; the measured 280 ms
#                          TCP round trip is a continent, not a server.
#   LEV_SERVE_SCALEDOWN    idle seconds before a container stops (default 300).
SERVE_CONCURRENCY = int(os.environ.get("LEV_SERVE_CONCURRENCY", "4"))
SERVE_WARM = int(os.environ.get("LEV_SERVE_WARM", "0"))
SERVE_REGION = os.environ.get("LEV_SERVE_REGION")
SERVE_SCALEDOWN = int(os.environ.get("LEV_SERVE_SCALEDOWN", "300"))


def _serve_overrides() -> list:
    """Which checkpoint the server loads, chosen from the local environment.

    `LEV_SERVE_PRESET=4b-instruct make deploy` serves that preset's newest
    checkpoint. `LEV_SERVE_MODEL=Qwen/Qwen3.5-4B` serves that model frozen --
    no adapter, binary Noul -- the zero-shot baseline every trained run has to
    beat (reflex-4b: 0.719 on S1Bench). Local env does not reach the
    container, so whichever are set travel as a Secret.
    """
    overrides = {
        key: value
        for key in (
            "LEV_SERVE_PRESET",
            "LEV_SERVE_MODEL",
            "LEV_SERVE_COMPILE",
            "LEV_SERVE_MAX_LABEL_OPTIONS",
            "LEV_SERVE_PROMPT",
            "LEV_SERVE_SKIP_CODES",
        )
        if (value := os.environ.get(key))
    }
    return [modal.Secret.from_dict(overrides)] if overrides else []


SECRETS = _hf_secrets() + _serve_overrides()


@app.function(volumes={MODELS_DIR: models}, secrets=SECRETS, timeout=60 * 60)
def download(model_id: str = "Qwen/Qwen3.5-4B-Base") -> str:
    """Pre-fetch a checkpoint into the models Volume. Idempotent.

    `cache_dir`, not `local_dir`. Training loads with
    `from_pretrained(model_id, cache_dir=MODELS_DIR)`, which looks for the HF
    cache layout (`models--Qwen--...`); `local_dir` writes a flat directory
    that lookup never finds, so the warm-up silently did nothing and the real
    run downloaded 8 GB again on the clock.
    """
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        model_id,
        cache_dir=MODELS_DIR,
        ignore_patterns=["*.pth", "*.gguf", "original/*"],
    )
    models.commit()
    print(f"cached {model_id} -> {path}")
    return path


@app.function(
    gpu="H100",
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=24 * 60 * 60,  # ~2h estimate; margin is cheap, a truncated run is not
)
def train(
    preset: str = "4b",
    resume: str | None = None,
    dry_run: bool = False,
    max_steps: int | None = None,
    fresh: bool = False,
) -> dict:
    """Fine-tune on one H100. See `lev.train.config.PRESETS`.

    Resumes from the newest checkpoint in the preset's directory unless
    `--fresh`; `--resume <path>` names one explicitly. A preempted run costs
    at most `checkpoint_every` steps, which is the point of writing them.
    """
    from lev.train.config import PRESETS
    from lev.train.loop import run_training

    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; have {sorted(PRESETS)}")

    config = PRESETS[preset]
    config.output_dir = f"{CKPT_DIR}/{preset}"
    config.validate()

    print(config.summary())
    if dry_run:
        return {"preset": preset, "dry_run": True, "hours": config.estimated_hours}

    summary = run_training(
        config,
        data_dir=DATA_DIR,
        model_cache=MODELS_DIR,
        resume_from=resume,
        max_steps=max_steps,
        fresh=fresh,
        # Commit mid-run so a preemption near the end does not cost the whole run.
        on_checkpoint=lambda: checkpoints.commit(),
    )
    checkpoints.commit()
    return summary


@app.function(volumes={DATA_DIR: datasets_vol}, secrets=SECRETS, timeout=4 * 60 * 60)
def build_data(limit_per_source: int = 20_000, n_examples: int = 200_000) -> dict:
    """Download every source and write the three splits to the data volume.

    No GPU: this is downloads and CPU, and paying H100 rates to wait on the
    HuggingFace CDN is the most avoidable line on the bill. Run it once; `train`
    then starts against a volume that already has the data, and fails in seconds
    rather than minutes if it does not.
    """
    from lev.data.build import build_dataset

    # The HF download cache on the volume, not in the container: the first
    # build downloads ~25 corpora, and without this every rebuild does too.
    manifest = build_dataset(
        DATA_DIR,
        limit_per_source=limit_per_source,
        n_examples=n_examples,
        cache_dir=f"{DATA_DIR}/hf-cache",
    )
    datasets_vol.commit()
    return manifest


@app.function(gpu="H100", volumes=VOLUMES, secrets=SECRETS, timeout=2 * 60 * 60)
def smoke(steps: int = 40) -> dict:
    """Exercise the whole path on the 0.8B preset. Run this first, always.

    Builds a small mixture if the volume is empty, so a fresh workspace needs
    exactly one command. It covers image, volumes, both readout modes, the loss,
    and a checkpoint write -- every part of `train` except its duration.
    """
    from pathlib import Path as _Path

    if not (_Path(DATA_DIR) / "train.jsonl").is_file():
        # On the GPU, which `build_data` exists to avoid -- but only for the
        # 2k-per-source smoke mixture, a couple of minutes, and it saves a
        # fresh workspace from needing two commands in the right order. Run
        # `build_data` yourself before `train`; do not let `train` land here.
        print("data volume is empty; building a small mixture first")
        build_data.local(limit_per_source=2_000, n_examples=4_000)

    # Always from scratch: a smoke test that resumed a previous smoke would
    # skip the very steps it exists to exercise.
    summary = train.local(preset="smoke", max_steps=steps, fresh=True)
    losses = [h["loss"] for h in summary["history"]]
    modes = {h["mode"] for h in summary["history"]}
    if len(modes) < 2:
        raise RuntimeError(
            f"only readout mode(s) {sorted(modes)} were exercised. The smoke run "
            f"has to cover both, or Mode B ships untested."
        )
    print(f"{len(losses)} steps, modes {sorted(modes)}, loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    return summary


@app.function(gpu="H100", volumes=VOLUMES, secrets=SECRETS, timeout=60 * 60)
def calibrate(preset: str = "4b", split: str = "calibration", method: str = "transfer") -> dict:
    """Fit per-bucket temperatures on a split that is neither train nor test.

    Separate from `train` on purpose: calibration is fitted *after* training, on
    held-out data, and re-fitting must not require another fine-tune.
    """
    from lev.train.calibration_run import fit_profile
    from lev.train.config import PRESETS

    config = PRESETS[preset]
    config.output_dir = f"{CKPT_DIR}/{preset}"
    return fit_profile(
        # The preset's output directory, not a step directory: `fit_profile`
        # resolves the newest checkpoint inside it.
        checkpoint_dir=config.output_dir,
        data_dir=DATA_DIR,
        split=split,
        on_complete=lambda: checkpoints.commit(),
        # Without this the head is built at the 4B hidden size whatever preset
        # was trained, and the state dict silently fails to match.
        config=config,
        method=method,
    )


@app.function(gpu="H100", volumes=VOLUMES, secrets=SECRETS, timeout=2 * 60 * 60)
def evaluate(
    preset: str = "4b",
    split: str = "test",
    limit_per_source: int | None = None,
) -> dict:
    """Score the newest checkpoint on a held-out split, with and without the
    fitted temperature.

    Runs in the container so the checkpoint and the data stay on their volumes.
    Reports both so the temperature's effect is visible: it cannot change
    accuracy, only calibration.
    """
    from lev.calibrate import CalibrationProfile
    from lev.train.checkpoints import CALIBRATION
    from lev.train.config import PRESETS
    from lev.train.evaluate import evaluate_split

    config = PRESETS[preset]
    config.output_dir = f"{CKPT_DIR}/{preset}"

    profile = CalibrationProfile()
    calibration = Path(config.output_dir) / CALIBRATION
    if calibration.is_file():
        profile = CalibrationProfile.load(calibration)
    else:
        print(f"WARNING: no calibration at {calibration}; both reports are uncalibrated")

    plain, tuned = evaluate_split(
        config.output_dir,
        DATA_DIR,
        split=split,
        config=config,
        profile=profile,
        limit_per_source=limit_per_source,
    )
    print(plain.summary())
    print()
    print(tuned.summary())
    return {"uncalibrated": plain.summary(), "calibrated": tuned.summary()}


@app.function(
    gpu="H100",
    volumes=VOLUMES,
    secrets=SECRETS,
    scaledown_window=SERVE_SCALEDOWN,
    min_containers=SERVE_WARM,
    **({"region": SERVE_REGION} if SERVE_REGION else {}),
)
@modal.concurrent(max_inputs=SERVE_CONCURRENCY)
@modal.asgi_app()
def serve():
    """Serve `/v1/systemone`, wire-compatible with the TypeSafe API.

    Point the benchmark at it with no code change:
        levbench eval --backend lev --base-url <the URL Modal prints> \
                      --tasks data/eval

    Serving the *untrained* backbone is a legitimate first step (TRAINING.md
    step 1), so a missing checkpoint falls back to the base model rather than
    refusing to start. `create_app` still raises when a checkpoint is named
    explicitly and is not there -- the difference is that here nothing was
    named, we only looked.
    """
    from pathlib import Path as _Path

    from lev.server import create_app
    from lev.train.checkpoints import latest_checkpoint
    from lev.train.config import PRESETS

    # Not an argument: `@modal.asgi_app` functions must be nullary. The preset
    # comes from LEV_SERVE_PRESET (see `_serve_overrides`), else the default.
    preset = os.environ.get("LEV_SERVE_PRESET", SERVE_PRESET)
    if preset not in PRESETS:
        raise ValueError(f"LEV_SERVE_PRESET={preset!r} is not a preset; have {sorted(PRESETS)}")
    config = PRESETS[preset]
    output = _Path(f"{CKPT_DIR}/{preset}")
    # `latest_checkpoint` ignores a step directory without weights, which is what
    # a save interrupted between `mkdir` and `save_pretrained` leaves behind.
    trained = (
        latest_checkpoint(output) is not None or (output / "adapter_model.safetensors").is_file()
    )

    # A frozen model is a different backbone; the adapter trained on -Base
    # cannot be applied to it, so the checkpoint is deliberately not loaded.
    # Off unless asked: measured slower than eager on this model (ADR-023).
    compile = os.environ.get("LEV_SERVE_COMPILE", "0") in ("1", "true", "yes")
    # Default None: Mode A up to the tokenizer limit (ADR-025). Set to force a
    # lower cap, e.g. 26 to reproduce the ADR-020 policy.
    cap_env = os.environ.get("LEV_SERVE_MAX_LABEL_OPTIONS")
    max_label_options = int(cap_env) if cap_env else None
    # The style the adapter trained under, from its preset; LEV_SERVE_PROMPT
    # overrides -- for a frozen model, which trained under neither.
    prompt_style = os.environ.get("LEV_SERVE_PROMPT") or config.prompt_style
    skip_codes = os.environ.get("LEV_SERVE_SKIP_CODES", "1") not in ("0", "false", "no")
    frozen = os.environ.get("LEV_SERVE_MODEL")
    if frozen:
        print(f"serving {frozen} frozen: no adapter, binary Noul, raw softmax")
        return create_app(
            checkpoint_dir=None,
            model_cache=MODELS_DIR,
            model_id=frozen,
            compile=compile,
            max_label_options=max_label_options,
            prompt_style=prompt_style,
            skip_multi_token_codes=skip_codes,
        )

    if not trained:
        print(f"WARNING: nothing trained at {output}; serving the base backbone uncalibrated")

    return create_app(
        checkpoint_dir=str(output) if trained else None,
        model_cache=MODELS_DIR,
        model_id=config.model_id,
        compile=compile,
        max_label_options=max_label_options,
        prompt_style=prompt_style,
        skip_multi_token_codes=skip_codes,
    )


@app.local_entrypoint()
def main(preset: str = "4b", dry_run: bool = True):
    """Default entrypoint: print the budget without spending it."""
    result = train.remote(preset=preset, dry_run=dry_run)
    print(result)


RELEASES_DIR = f"{CKPT_DIR}/releases"


@app.function(volumes={CKPT_DIR: checkpoints}, secrets=SECRETS, timeout=30 * 60)
def export_checkpoint(preset: str = "4b", name: str | None = None) -> dict:
    """Package the newest checkpoint of a preset into `/checkpoints/releases/<name>`.

    No GPU: this copies files. The result is what `make weights` pulls and
    `lev release publish` uploads -- adapter, head, tokenizer, calibration and
    a manifest naming the base model and serving policy.
    """
    from lev.release import build_release
    from lev.train.config import PRESETS

    manifest = build_release(
        f"{CKPT_DIR}/{preset}",
        f"{RELEASES_DIR}/{name or preset}",
        preset=preset,
        name=name,
        prompt_style=PRESETS[preset].prompt_style,
    )
    checkpoints.commit()
    target = name or preset
    print(
        f"release {manifest['name']} -> {RELEASES_DIR}/{target}  ({len(manifest['files'])} files)"
    )
    print(f"  pull it:    make weights RELEASE={target}")
    print(f"  publish it: make publish RELEASE={target} REPO=<org/name>")
    return manifest


@app.function(gpu="H100", volumes=VOLUMES, secrets=SECRETS, timeout=30 * 60)
def check_release(name: str = "4b") -> dict:
    """Load a packaged release the way a user will -- `lev.load` on the flat
    directory, then the HTTP server over it -- and answer one request of every
    question type through each. Raises if the release loads without its
    calibration or head, or if the two paths disagree.
    """
    import json

    from fastapi.testclient import TestClient
    from lev import load
    from lev.server import create_app

    release = f"{RELEASES_DIR}/{name}"
    request = {
        "state": "Hi, I was charged twice for my order #4471 and I want a refund.",
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "What does the customer want?",
                "criteria": {
                    "refund": "wants money back",
                    "cancel": "wants to cancel an order",
                    "track": "wants to know where an order is",
                    "other": "anything else",
                },
            },
            "urgent": {"type": "noul", "instructions": "Does this need a human within the hour?"},
            "frustration": {
                "type": "score",
                "instructions": "How frustrated is the customer?",
                "criteria": ["calm", "mildly annoyed", "annoyed", "angry"],
            },
        },
    }

    engine = load(release, cache_dir=MODELS_DIR)
    if not engine.calibration.temperatures or engine.mode_b_head is None:
        raise RuntimeError(f"{release} loaded without its calibration or its Mode B head")
    direct = engine.system_one(request["state"], request["questions"]).model_dump(mode="json")
    del engine

    with TestClient(create_app(release, model_cache=MODELS_DIR)) as client:
        health = client.get("/health").json()
        served = client.post("/v1/systemone", json=request)
    served.raise_for_status()
    served = served.json()
    for question, answer in served["answers"].items():
        expected = direct["answers"][question]
        gaps = [abs(answer["probabilities"][k] - p) for k, p in expected["probabilities"].items()]
        gap = max(gaps, default=0.0)
        if answer.get("choice") != expected.get("choice") or gap > 1e-3:
            raise RuntimeError(f"server and lev.load disagree on {question}: {answer}, {expected}")

    print(json.dumps({"health": health, "response": served}, indent=2))
    return {"health": health, "response": served}


@app.function(gpu="H100", volumes=VOLUMES, secrets=SECRETS, timeout=30 * 60)
def diagnose_candidates(model_id: str = "Qwen/Qwen3.5-4B-Base") -> dict:
    """Is a candidate string's representation independent of its batch neighbours?

    Mode B embeds each option string by running the whole option set through
    the backbone as one right-padded batch and reading the last real position.
    The head is permutation-invariant, yet reordering the options changes the
    served distribution almost entirely (L1 1.23 on massive-en-US). If a
    string's vector differs between "alone" and "in a batch", or between two
    batch orders, the padded forward is leaking across rows and every Mode B
    representation the head was trained on was position-contaminated.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=MODELS_DIR)
    model = (
        AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, cache_dir=MODELS_DIR)
        .cuda()
        .eval()
    )
    texts = [
        "datetime query",
        "iot hue lightchange",
        "transport ticket",
        "takeaway query",
        "qa stock",
        "general greet",
        "recommendation events",
        "music dislikeness",
        "iot wemo off",
        "cooking recipe",
        "qa currency",
        "transport traffic",
        "general quirky",
        "weather query",
        "audio volume up",
        "email addcontact",
        "takeaway order",
        "email querycontact",
        "iot hue lightup",
        "recommendation locations",
        "play audiobook",
        "lists createoradd",
        "news query",
        "alarm query",
        "iot wemo on",
        "general joke",
        "qa definition",
        "social query",
        "music settings",
        "audio volume other",
        "calendar remove",
        "iot hue lightdim",
        "calendar query",
        "email sendemail",
        "iot cleaning",
        "audio volume down",
        "play radio",
        "cooking query",
        "datetime convert",
        "qa maths",
    ]

    def reprs(batch, side="right"):
        tok.padding_side = side
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states[-1]
        last = (
            enc["attention_mask"].sum(1) - 1
            if side == "right"
            else torch.full((len(batch),), hs.size(1) - 1, device="cuda")
        )
        return hs[torch.arange(len(batch), device="cuda"), last].float()

    solo = torch.cat([reprs([t]) for t in texts])
    in_order = reprs(texts)
    reversed_ = reprs(texts[::-1]).flip(0)
    left = reprs(texts, side="left")

    def gap(a, b):
        diff = (a - b).abs()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        return {
            "max_abs": round(diff.max().item(), 4),
            "mean_abs": round(diff.mean().item(), 5),
            "min_cosine": round(cos.min().item(), 4),
            "rows_changed": int((diff.max(dim=1).values > 1e-2).sum().item()),
        }

    report = {
        "model": model_id,
        "n_texts": len(texts),
        "repr_norm_mean": round(solo.norm(dim=-1).mean().item(), 3),
        "batch_vs_solo (right pad)": gap(in_order, solo),
        "order_vs_reversed (right pad)": gap(in_order, reversed_),
        "leftpad_vs_solo": gap(left, solo),
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
    }
    for key, value in report.items():
        print(f"{key:<32} {value}")
    return report


@app.function(gpu="H100", volumes=VOLUMES, secrets=SECRETS, timeout=30 * 60)
def profile_engine(preset: str = "4b", rounds: int = 20) -> dict:
    """Where a `/v1/systemone` call spends its time inside the container.

    Loads the served checkpoint exactly as `serve` does and times each stage of
    the engine with CUDA synchronised: tokenising, the prefix prefill, the cache
    fork, the suffix forward, the readout. Medians over `rounds` after warmup,
    for the request shapes the benchmark and the demo actually send. Network
    is excluded by construction; subtract these from a client-side latency to
    get it.
    """
    import statistics
    import time

    import lev.model as engine_module
    import torch
    from lev.model import DecisionEngine, EngineConfig, load
    from lev.train.config import PRESETS
    from lev.types import Choice, Noul, Score

    config = PRESETS[preset]
    engine = load(
        f"{CKPT_DIR}/{preset}",
        model_id=config.model_id,
        cache_dir=MODELS_DIR,
        prompt_style=config.prompt_style,
    )
    model, tokenizer = engine.model, engine.tokenizer
    profile, head = engine.calibration, engine.mode_b_head

    kernels = {}
    for name in ("fla", "causal_conv1d"):
        try:
            __import__(name)
            kernels[name] = "installed"
        except ImportError:
            kernels[name] = "MISSING (reference PyTorch path)"

    # Instrument the engine's stages.
    timings: dict[str, list[float]] = {}

    def timed(name, fn):
        def wrapper(*a, **k):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = fn(*a, **k)
            torch.cuda.synchronize()
            timings.setdefault(name, []).append((time.perf_counter() - t) * 1000)
            return out

        return wrapper

    engine._render = timed("render+tokenise", engine._render)
    engine._forward = timed("prefill+fork+suffix forward", engine._forward)
    engine_module._fork = timed("  of which: cache fork (deepcopy)", engine_module._fork)
    engine._candidate_reprs = timed("mode B candidate encode", engine._candidate_reprs)

    state = (
        "Customer writes: I was charged twice for my order #48213 last Tuesday, the second "
        "charge has not been refunded, and I need this fixed before my rent is due on Friday. "
        "I have already called once and was told to wait 5 days."
    )
    shapes = {
        "1 noul": {"urgent": Noul(instructions="Is the customer blocked right now?")},
        "3 mixed (choice4, score3, noul)": {
            "dept": Choice(
                instructions="Which team?",
                criteria={"billing": None, "technical": None, "sales": None, "account": None},
            ),
            "frustration": Score(
                instructions="How frustrated?", criteria=["calm", "annoyed", "angry"]
            ),
            "urgent": Noul(instructions="Is the customer blocked right now?"),
        },
        "8 nouls": {
            f"q{i}": Noul(instructions=f"Question {i} about the ticket?") for i in range(8)
        },
        "60-option choice (Mode B)": {
            "intent": Choice(
                instructions="What is the user's intent?",
                criteria={f"intent number {i}": None for i in range(60)},
            )
        },
    }

    report: dict = {"gpu": torch.cuda.get_device_name(0), "kernels": kernels, "shapes": {}}
    # The two strategies must agree before the faster one is trusted.
    answers = {}
    for prefix_mode in ("fork", "single"):
        engine.config.prefix_mode = prefix_mode
        answers[prefix_mode] = {
            label: engine.system_one(state, q).answers for label, q in shapes.items()
        }
    worst = 0.0
    for label in shapes:
        for name, a in answers["fork"][label].items():
            b = answers["single"][label][name]
            pa = a.probabilities or {1: a.noul}
            pb = b.probabilities or {1: b.noul}
            worst = max(worst, *(abs(pa[k] - pb[k]) for k in pa))
    report["fork_vs_single_max_prob_diff"] = round(worst, 5)
    print(f"fork vs single: max |dp| over all answers = {worst:.5f}", flush=True)
    # Both prefix strategies, because which is faster depends on whether the
    # forward is FLOP-bound (fork wins) or launch-bound (single wins); then the
    # compiled single path, with its warmup cost reported separately.
    variants = [("fork", False), ("single", False), ("single", True)]
    for prefix_mode, compiled in variants:
        if compiled:
            compiled_engine = DecisionEngine(
                model,
                tokenizer,
                EngineConfig(
                    model_id=config.model_id,
                    prompt_style=config.prompt_style,
                    prefix_mode="single",
                    compile=True,
                ),
                profile,
                head,
            )
            warm = compiled_engine.warmup()
            print(
                f"compile warmup {warm:.1f}s  (compiled={compiled_engine.config.compile})",
                flush=True,
            )
            report["compile_warmup_s"] = round(warm, 1)
            if not compiled_engine.config.compile:
                report["compile"] = "failed; see warning above"
                break
            active = compiled_engine
            active._render = timed("render+tokenise", active._render)
            active._forward = timed("prefill+fork+suffix forward", active._forward)
            reference = engine.system_one(state, shapes["3 mixed (choice4, score3, noul)"]).answers
            got = active.system_one(state, shapes["3 mixed (choice4, score3, noul)"]).answers
            drift = max(
                abs(
                    (a.probabilities or {1: a.noul})[k]
                    - (got[n].probabilities or {1: got[n].noul})[k]
                )
                for n, a in reference.items()
                for k in (a.probabilities or {1: a.noul})
            )
            report["compiled_vs_eager_max_prob_diff"] = round(drift, 5)
            print(f"compiled vs eager: max |dp| = {drift:.5f}", flush=True)
        else:
            active = engine
            active.config.prefix_mode = prefix_mode
        tag = f"{prefix_mode}{'+compile' if compiled else ''}"
        for label, questions in shapes.items():
            for _ in range(5):
                active.system_one(state, questions)
            timings.clear()
            totals = []
            for _ in range(rounds):
                torch.cuda.synchronize()
                t = time.perf_counter()
                active.system_one(state, questions)
                torch.cuda.synchronize()
                totals.append((time.perf_counter() - t) * 1000)
            stages = {k: round(statistics.median(v), 2) for k, v in timings.items()}
            key = f"{tag}: {label}"
            report["shapes"][key] = {
                "total_ms_median": round(statistics.median(totals), 2),
                **stages,
            }
            detail = "  ".join(f"{k}={v}" for k, v in stages.items())
            print(f"{key:<42} total {statistics.median(totals):7.2f} ms  {detail}", flush=True)
    print("kernels:", kernels)
    return report


@app.function(secrets=SECRETS, timeout=10 * 60)
def env_info() -> dict:
    """What the image actually runs: torch, its CUDA build, and which kernels import."""
    import platform

    import torch

    # Plain strings: `torch.__version__` is a TorchVersion, which the local
    # `modal run` process cannot unpickle without torch installed.
    info = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
    }
    for name in ("fla", "causal_conv1d", "triton"):
        try:
            module = __import__(name)
            info[name] = str(getattr(module, "__version__", "installed"))
        except ImportError:
            info[name] = "missing"
    print(info)
    return info
