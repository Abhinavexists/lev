"""Modal app: train and serve lev on one H100.

    modal setup                                   # once
    modal run   modal/app.py::smoke               # ~5 min, proves the path works
    modal run   modal/app.py::train --preset 4b   # the real run, ~2h
    modal serve modal/app.py                      # /v1/systemone on an H100

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
    modal.Image.debian_slim(python_version="3.12")
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
    # Speed, not correctness -- but 24 of the 32 layers are linear-attention, so
    # without this three quarters of the model runs on a reference PyTorch path
    # (ADR-017). Pure Python and Triton, so no compiler needed.
    #
    # `causal-conv1d` is deliberately absent: it is a CUDA source build needing
    # `nvcc`, which `debian_slim` has no compiler for, and it accelerates only
    # the short depthwise conv. Unpinned because it tracks torch closely; if the
    # build breaks, delete this layer -- the run gets slow, not wrong.
    .pip_install("flash-linear-attention")
    # The package last, so editing it does not invalidate the expensive
    # dependency layer above. `add_local_dir`, not `add_local_python_source`:
    # the latter resolves through the local interpreter's import system and so
    # fails unless `lev` is installed in whichever Python runs the `modal` CLI.
    # `/root` is on `sys.path` in a Modal container, so `import lev` resolves.
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


def _serve_overrides() -> list:
    """`LEV_SERVE_MODEL=Qwen/Qwen3.5-4B modal serve modal/app.py` serves that
    checkpoint frozen -- no adapter, binary Noul -- which is the zero-shot
    baseline every trained run has to beat (reflex-4b: 0.719 on S1Bench).
    Local env does not reach the container, so it travels as a Secret."""
    model = os.environ.get("LEV_SERVE_MODEL")
    return [modal.Secret.from_dict({"LEV_SERVE_MODEL": model})] if model else []


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

    manifest = build_dataset(DATA_DIR, limit_per_source=limit_per_source, n_examples=n_examples)
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
def calibrate(preset: str = "4b", split: str = "calibration") -> dict:
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
    from lev.train.config import PRESETS
    from lev.train.evaluate import evaluate_split

    config = PRESETS[preset]
    config.output_dir = f"{CKPT_DIR}/{preset}"

    profile = CalibrationProfile()
    calibration = Path(config.output_dir) / "calibration.json"
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


@app.function(gpu="H100", volumes=VOLUMES, secrets=SECRETS, scaledown_window=300)
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
    from lev.train.config import PRESETS

    # A module constant, not an argument: `@modal.asgi_app` functions must be
    # nullary. Change SERVE_PRESET to serve a different one.
    config = PRESETS[SERVE_PRESET]
    output = _Path(f"{CKPT_DIR}/{SERVE_PRESET}")
    trained = output.is_dir() and (
        any(output.glob("step-*")) or (output / "adapter_model.safetensors").is_file()
    )

    # A frozen model is a different backbone; the adapter trained on -Base
    # cannot be applied to it, so the checkpoint is deliberately not loaded.
    frozen = os.environ.get("LEV_SERVE_MODEL")
    if frozen:
        print(f"serving {frozen} frozen: no adapter, binary Noul, raw softmax")
        return create_app(checkpoint_dir=None, model_cache=MODELS_DIR, model_id=frozen)

    if not trained:
        print(f"WARNING: nothing trained at {output}; serving the base backbone uncalibrated")

    return create_app(
        checkpoint_dir=str(output) if trained else None,
        model_cache=MODELS_DIR,
        model_id=config.model_id,
    )


@app.local_entrypoint()
def main(preset: str = "4b", dry_run: bool = True):
    """Default entrypoint: print the budget without spending it."""
    result = train.remote(preset=preset, dry_run=dry_run)
    print(result)


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
RELEASES_DIR = f"{CKPT_DIR}/releases"


@app.function(volumes={CKPT_DIR: checkpoints}, secrets=SECRETS, timeout=30 * 60)
def export_checkpoint(preset: str = "4b", name: str | None = None) -> dict:
    """Package the newest checkpoint of a preset into `/checkpoints/releases/<name>`.

    No GPU: this copies files. The result is what `make weights` pulls and
    `lev release publish` uploads -- adapter, head, tokenizer, calibration and
    a manifest naming the base model and serving policy.
    """
    from lev.release import build_release

    manifest = build_release(
        f"{CKPT_DIR}/{preset}", f"{RELEASES_DIR}/{name or preset}", preset=preset, name=name
    )
    checkpoints.commit()
    target = name or preset
    print(
        f"release {manifest['name']} -> {RELEASES_DIR}/{target}  ({len(manifest['files'])} files)"
    )
    print(f"  pull it:    make weights RELEASE={target}")
    print(f"  publish it: make publish RELEASE={target} REPO=<org/name>")
    return manifest


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
