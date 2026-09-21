"""Modal app: train and serve lev on one H100.

    modal setup                                   # once
    modal run   modal/app.py::smoke               # ~5 min, proves the path works
    modal run   modal/app.py::train --preset 4b   # the real run, ~16h
    modal serve modal/app.py                      # /v1/systemone on an H100

Design notes worth knowing before you change anything here:

* **The model cache is a Volume, not part of the image.** A 4B checkpoint is ~8 GB;
  baking it into the image makes every rebuild slow and every push enormous.
* **Checkpoints go to a second Volume and are committed *during* training**, not at
  the end. A 16-hour run that loses everything to a preemption is a bad trade for
  the two lines it takes to commit periodically.
* **`smoke` exists to be run first.** It uses the 0.8B preset and a few hundred
  steps, so the whole path -- image, volumes, data, loss, checkpoint write -- is
  exercised for a few minutes of GPU time instead of discovering a bug at hour 15.
* **H100 timeout is set to 24h.** The 4B estimate is 16h; leaving no margin means a
  slightly slow run dies just before it saves.

NOT YET RUN. This file has not been executed against Modal. Run `smoke` first.
"""

from __future__ import annotations

import modal

APP_NAME = "lev"

# Pinned rather than floating: an unpinned torch or transformers turns a
# reproducible run into a lottery, and this is the file people will copy.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.57.0",
        "accelerate==1.2.1",
        "peft==0.14.0",
        "datasets==3.2.0",
        "pydantic==2.12.0",
        "fastapi==0.115.6",
        "huggingface-hub==0.27.0",
    )
    # The package itself last, so editing our code does not invalidate the
    # expensive dependency layer above.
    .add_local_python_source("lev")
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

# A gated or private backbone needs a token; a public one does not.
SECRETS = [modal.Secret.from_name("huggingface", required_keys=[])]


@app.function(volumes={MODELS_DIR: models}, secrets=SECRETS, timeout=60 * 60)
def download(model_id: str = "Qwen/Qwen3.5-4B-Base") -> str:
    """Pre-fetch a checkpoint into the models Volume. Idempotent."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        model_id,
        local_dir=f"{MODELS_DIR}/{model_id.replace('/', '__')}",
        ignore_patterns=["*.pth", "*.gguf", "original/*"],
    )
    models.commit()
    print(f"cached {model_id} -> {path}")
    return path


@app.function(
    gpu="H100",
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=24 * 60 * 60,  # 4B estimate is 16h; leave margin, do not cut it fine
)
def train(preset: str = "4b", resume: str | None = None, dry_run: bool = False) -> dict:
    """Fine-tune on one H100. See `lev.train.config.PRESETS`."""
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

    return run_training(
        config,
        data_dir=DATA_DIR,
        model_cache=MODELS_DIR,
        resume_from=resume,
        # Commit mid-run so a preemption at hour 15 does not cost the whole run.
        on_checkpoint=lambda: checkpoints.commit(),
    )


@app.function(gpu="H100", volumes=VOLUMES, secrets=SECRETS, timeout=60 * 60)
def smoke() -> dict:
    """Exercise the whole path on the 0.8B preset. Run this first, always."""
    return train.local(preset="smoke")


@app.function(gpu="H100", volumes=VOLUMES, secrets=SECRETS, timeout=60 * 60)
def calibrate(preset: str = "4b", split: str = "calibration") -> dict:
    """Fit per-bucket temperatures on a split that is neither train nor test.

    Separate from `train` on purpose: calibration is fitted *after* training, on
    held-out data, and re-fitting must not require another fine-tune.
    """
    from lev.train.calibration_run import fit_profile

    return fit_profile(
        checkpoint_dir=f"{CKPT_DIR}/{preset}",
        data_dir=DATA_DIR,
        split=split,
        on_complete=lambda: checkpoints.commit(),
    )


@app.function(gpu="H100", volumes=VOLUMES, secrets=SECRETS, scaledown_window=300)
@modal.asgi_app()
def serve():
    """Serve `/v1/systemone`, wire-compatible with the TypeSafe API.

    Point the benchmark at it with no code change:
        levbench eval --backend jev --base-url <the URL Modal prints>
    """
    from lev.server import create_app

    return create_app(
        checkpoint_dir=f"{CKPT_DIR}/4b",
        model_cache=MODELS_DIR,
        calibration=f"{CKPT_DIR}/4b/calibration.json",
    )


@app.local_entrypoint()
def main(preset: str = "4b", dry_run: bool = True):
    """Default entrypoint: print the budget without spending it."""
    result = train.remote(preset=preset, dry_run=dry_run)
    print(result)
