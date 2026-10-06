"""Build, train, calibrate, evaluate, and serve lev on Modal (SETUP.md).

Use modal deploy for a stable URL; modal serve uses a shared development label."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # `lev` is installed in the container, not where deploy runs.
    from lev.types import Answer

import modal

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

APP_NAME = "lev"

REPO_ROOT = Path(__file__).resolve().parent.parent

# Pin dependencies; hybrid-cache reordering requires Transformers 5.x.
image = (
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
    .pip_install("flash-linear-attention", "ninja", "packaging")
    .apt_install("build-essential")
    .env({"CC": "gcc", "CXX": "g++", "TORCH_CUDA_ARCH_LIST": "9.0", "MAX_JOBS": "8"})
    # Expandable segments reduce fragmentation from varying batch shapes.
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .pip_install("causal-conv1d", extra_options="--no-build-isolation")
    .add_local_dir(
        REPO_ROOT / "packages" / "lev" / "src" / "lev",
        remote_path="/root/lev",
        ignore=["**/__pycache__", "**/*.pyc"],
    )
)

app = modal.App(APP_NAME, image=image)

models = modal.Volume.from_name("lev-models", create_if_missing=True)
checkpoints = modal.Volume.from_name("lev-checkpoints", create_if_missing=True)
datasets_vol = modal.Volume.from_name("lev-data", create_if_missing=True)

MODELS_DIR = "/models"
CKPT_DIR = "/checkpoints"
DATA_DIR = "/data"

VOLUMES: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
    MODELS_DIR: models,
    CKPT_DIR: checkpoints,
    DATA_DIR: datasets_vol,
}

# Use an environment preset because Modal ASGI functions accept no arguments.
SERVE_PRESET = "4b"


def _hf_secrets() -> list:
    named = os.environ.get("LEV_HF_SECRET")
    if named:
        return [modal.Secret.from_name(named, required_keys=[])]
    token = os.environ.get("HF_TOKEN")
    return [modal.Secret.from_dict({"HF_TOKEN": token})] if token else []


SERVE_CONCURRENCY = int(os.environ.get("LEV_SERVE_CONCURRENCY", "32"))
SERVE_WARM = int(os.environ.get("LEV_SERVE_WARM", "0"))
SERVE_REGION = os.environ.get("LEV_SERVE_REGION")
SERVE_SCALEDOWN = int(os.environ.get("LEV_SERVE_SCALEDOWN", "300"))
SERVE_MAX = int(os.environ["LEV_SERVE_MAX"]) if os.environ.get("LEV_SERVE_MAX") else None


def _prompt_style(value: str | None) -> Literal["plain", "chat"] | None:
    if not value:
        return None
    if value not in ("plain", "chat"):
        raise ValueError(f"LEV_SERVE_PROMPT={value!r} is not 'plain' or 'chat'")
    return value


def _serve_overrides() -> list:
    overrides: dict[str, str | None] = {
        key: value
        for key in (
            "LEV_SERVE_PRESET",
            "LEV_SERVE_MODEL",
            "LEV_SERVE_COMPILE",
            "LEV_SERVE_MAX_LABEL_OPTIONS",
            "LEV_SERVE_PROMPT",
            "LEV_SERVE_SKIP_CODES",
            "LEV_SERVE_CONCURRENCY",
        )
        if (value := os.environ.get(key))
    }
    return [modal.Secret.from_dict(overrides)] if overrides else []


SECRETS = _hf_secrets() + _serve_overrides()


@app.function(volumes={MODELS_DIR: models}, secrets=SECRETS, timeout=60 * 60)
def download(model_id: str = "Qwen/Qwen3.5-4B-Base") -> str:
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
    timeout=24 * 60 * 60,  # Allow the full training run to finish.
)
def train(
    preset: str = "4b",
    resume: str | None = None,
    dry_run: bool = False,
    max_steps: int | None = None,
    fresh: bool = False,
) -> dict:
    from lev.train.config import PRESETS
    from lev.train.loop import run_training

    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; have {sorted(PRESETS)}")

    config = replace(PRESETS[preset], output_dir=f"{CKPT_DIR}/{preset}")
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
        # Commit mid-run, so a late preemption keeps the checkpoints.
        on_checkpoint=lambda: checkpoints.commit(),
    )
    checkpoints.commit()
    return summary


@app.function(volumes={DATA_DIR: datasets_vol}, secrets=SECRETS, timeout=4 * 60 * 60)
def build_data(limit_per_source: int = 20_000, n_examples: int = 200_000) -> dict:
    from lev.data.build import build_dataset

    # The HF download cache lives on the volume, so rebuilds reuse the corpora.
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
    if not (Path(DATA_DIR) / "train.jsonl").is_file():
        # Builds on the GPU, acceptable only for this small mixture (a couple of minutes).
        print("data volume is empty; building a small mixture first")
        build_data.local(limit_per_source=2_000, n_examples=4_000)

    # From scratch: resuming a previous smoke would skip the steps it tests.
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
    from lev.train.calibration_run import fit_profile
    from lev.train.config import PRESETS

    config = replace(PRESETS[preset], output_dir=f"{CKPT_DIR}/{preset}")
    return fit_profile(
        checkpoint_dir=config.output_dir,
        data_dir=DATA_DIR,
        split=split,
        on_complete=lambda: checkpoints.commit(),
        # Use the preset's hidden size when building the head.
        config=config,
        method=method,
    )


@app.function(gpu="H100", volumes=VOLUMES, secrets=SECRETS, timeout=2 * 60 * 60)
def evaluate(
    preset: str = "4b",
    split: str = "test",
    limit_per_source: int | None = None,
) -> dict:
    from lev.calibrate import CalibrationProfile
    from lev.train.checkpoints import CALIBRATION
    from lev.train.config import PRESETS
    from lev.train.evaluate import evaluate_split

    config = replace(PRESETS[preset], output_dir=f"{CKPT_DIR}/{preset}")

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
    max_containers=SERVE_MAX,
    region=SERVE_REGION or None,  # `LEV_SERVE_REGION=` must mean unset, not ""
)
@modal.concurrent(max_inputs=SERVE_CONCURRENCY)
@modal.asgi_app()
def serve():
    from lev.server import create_app
    from lev.train.checkpoints import ADAPTER_WEIGHTS, latest_checkpoint
    from lev.train.config import PRESETS

    preset = os.environ.get("LEV_SERVE_PRESET", SERVE_PRESET)
    if preset not in PRESETS:
        raise ValueError(f"LEV_SERVE_PRESET={preset!r} is not a preset; have {sorted(PRESETS)}")
    config = PRESETS[preset]

    # Off unless asked: measured slower than eager on this model (ADR-023).
    compile = os.environ.get("LEV_SERVE_COMPILE", "0") in ("1", "true", "yes")
    # Default to the tokenizer limit; allow an explicit lower label cap.
    cap_env = os.environ.get("LEV_SERVE_MAX_LABEL_OPTIONS")
    max_label_options = int(cap_env) if cap_env else None
    prompt_style = _prompt_style(os.environ.get("LEV_SERVE_PROMPT")) or config.prompt_style
    skip_codes = os.environ.get("LEV_SERVE_SKIP_CODES", "1") not in ("0", "false", "no")
    # A frozen-model override must bypass the adapter checkpoint.
    frozen = os.environ.get("LEV_SERVE_MODEL")
    if frozen:
        print(f"serving {frozen} frozen: no adapter, binary Noul, raw softmax")
        checkpoint_dir, model_id = None, frozen
    else:
        output = Path(f"{CKPT_DIR}/{preset}")
        # `latest_checkpoint` skips a step directory without weights (an interrupted save).
        trained = latest_checkpoint(output) is not None or (output / ADAPTER_WEIGHTS).is_file()
        if not trained:
            print(f"WARNING: nothing trained at {output}; serving the base backbone uncalibrated")
        checkpoint_dir = str(output) if trained else None
        model_id = config.model_id

    return create_app(
        checkpoint_dir=checkpoint_dir,
        model_cache=MODELS_DIR,
        model_id=model_id,
        compile=compile,
        max_label_options=max_label_options,
        prompt_style=prompt_style,
        skip_multi_token_codes=skip_codes,
        max_pending=SERVE_CONCURRENCY,
    )


@app.local_entrypoint()
def main(preset: str = "4b", dry_run: bool = True):
    """Default entrypoint: print the budget without spending it."""
    result = train.remote(preset=preset, dry_run=dry_run)
    print(result)


RELEASES_DIR = f"{CKPT_DIR}/releases"


@app.function(volumes={CKPT_DIR: checkpoints}, secrets=SECRETS, timeout=30 * 60)
def export_checkpoint(preset: str = "4b", name: str | None = None) -> dict:
    from lev.release import build_release
    from lev.train.config import PRESETS

    target = name or preset
    manifest = build_release(
        f"{CKPT_DIR}/{preset}",
        f"{RELEASES_DIR}/{target}",
        preset=preset,
        name=name,
        prompt_style=PRESETS[preset].prompt_style,
    )
    checkpoints.commit()
    print(
        f"release {manifest['name']} -> {RELEASES_DIR}/{target}  ({len(manifest['files'])} files)"
    )
    print(f"  pull it:    make weights RELEASE={target}")
    print(f"  publish it: make publish RELEASE={target} REPO=<org/name>")
    return manifest


@app.function(gpu="H100", volumes=VOLUMES, secrets=SECRETS, timeout=30 * 60)
def check_release(name: str = "4b") -> dict:
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
        response = client.post("/v1/systemone", json=request)
    response.raise_for_status()
    served = response.json()
    for question, answer in served["answers"].items():
        expected = direct["answers"][question]
        gaps = [abs(answer["probabilities"][k] - p) for k, p in expected["probabilities"].items()]
        gap = max(gaps, default=0.0)
        if answer.get("choice") != expected.get("choice") or gap > 1e-3:
            raise RuntimeError(f"server and lev.load disagree on {question}: {answer}, {expected}")

    print(json.dumps({"health": health, "response": served}, indent=2))
    return {"health": health, "response": served}


@app.function(gpu="H100", volumes=VOLUMES, secrets=SECRETS, timeout=2 * 60 * 60)
def diagnose_ordinal(
    name: str = "lev",
    scales: str = "0,0.25,0.5,0.75,1.0",
    subsets: str = "summeval-consistency,summeval-relevance,helpsteer2",
) -> dict:
    """Sweep adapter scales on rating tasks; use scale zero as the frozen-backbone control."""
    import collections
    import json

    from lev import load
    from lev.data.s1bench import load_eval_subset
    from peft.tuners.lora import LoraLayer

    engine = load(f"{RELEASES_DIR}/{name}", cache_dir=MODELS_DIR, merge_adapter=False)
    layers = [m for m in engine.model.modules() if isinstance(m, LoraLayer)]
    if not layers:
        raise RuntimeError("no LoraLayer found; the adapter was merged or never applied")
    adapters = sorted({a for m in layers for a in m.scaling})
    print(f"{len(layers)} lora layers, adapters {adapters}")

    data = {s: load_eval_subset(s) for s in subsets.split(",")}
    report: dict = {"layers": len(layers), "subsets": {}}
    for scale in [float(x) for x in scales.split(",")]:
        for m in layers:
            for a in m.scaling:
                m.set_scale(a, scale)
        for subset, (question, items) in data.items():
            hits = 0
            picks: collections.Counter = collections.Counter()
            for item in items:
                answer = engine.system_one(item.state, {"decision": question}).answers["decision"]
                probs = {int(k): float(v) for k, v in answer.probabilities.items()}
                pick = max(probs, key=probs.get)
                picks[pick] += 1
                hits += pick == item.truth
            row = report["subsets"].setdefault(subset, {})
            row[f"{scale:g}"] = {
                "accuracy": round(hits / len(items), 4),
                "n": len(items),
                "argmax": dict(sorted(picks.items())),
            }
            acc = hits / len(items)
            print(f"  scale {scale:<5g} {subset:22} {acc:.4f}  {dict(sorted(picks.items()))}")

    print(json.dumps(report, indent=2))
    return report


@app.function(gpu="H100", volumes=VOLUMES, secrets=SECRETS, timeout=30 * 60)
def diagnose_candidates(model_id: str = "Qwen/Qwen3.5-4B-Base") -> dict:
    """Compare candidate embeddings alone, batched, reversed, and left-padded."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok: Any = AutoTokenizer.from_pretrained(model_id, cache_dir=MODELS_DIR)
    # `from_pretrained` is typed as returning the class, not an instance.
    loaded: Any = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, cache_dir=MODELS_DIR
    )
    model = loaded.cuda().eval()
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


def _probabilities(answer: Answer) -> dict[Any, float]:
    from lev.types import NoulAnswer  # noqa: PLC0415

    if isinstance(answer, NoulAnswer):
        return dict(answer.probabilities) if answer.probabilities else {1: answer.noul}
    return dict(answer.probabilities)


@app.function(gpu="H100", volumes=VOLUMES, secrets=SECRETS, timeout=30 * 60)
def profile_engine(preset: str = "4b", rounds: int = 20) -> dict:
    """Time CUDA-synchronized engine stages after warmup, excluding network time."""
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
        "60-option choice": {
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
            pa, pb = _probabilities(a), _probabilities(b)
            worst = max(worst, *(abs(pa[k] - pb[k]) for k in pa))
    report["fork_vs_single_max_prob_diff"] = round(worst, 5)
    print(f"fork vs single: max |dp| over all answers = {worst:.5f}", flush=True)
    # Compare forked, single, and compiled forwards; report warmup separately.
    variants: list[tuple[Literal["fork", "single"], bool]] = [
        ("fork", False),
        ("single", False),
        ("single", True),
    ]
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
                abs(_probabilities(a)[k] - _probabilities(got[n])[k])
                for n, a in reference.items()
                for k in _probabilities(a)
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
    import platform

    import torch

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
