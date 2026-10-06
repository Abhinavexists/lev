"""Package adapter, head, tokenizer, and calibration with a base-model and prompt manifest."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from .train.checkpoints import CALIBRATION, MODE_B_HEAD, TRAINING_STATE, resolve_checkpoint

RELEASE_MANIFEST = "lev_release.json"
MODEL_CARD = "README.md"


def build_release(
    checkpoint: str | Path,
    out: str | Path,
    *,
    preset: str,
    calibration: str | Path | None = None,
    name: str | None = None,
    metrics: dict | None = None,
    prompt_style: str = "plain",
) -> dict:
    """Package a checkpoint; default to parent calibration, otherwise serve raw softmax."""
    step_dir = resolve_checkpoint(checkpoint)
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for file in sorted(step_dir.iterdir()):
        # Exclude resume state and replace PEFT's README with the release card.
        if file.is_file() and file.name not in (TRAINING_STATE, MODEL_CARD):
            shutil.copy2(file, target / file.name)
            copied.append(file.name)

    profile = Path(calibration) if calibration else step_dir.parent / CALIBRATION
    calibrated = profile.is_file()
    if calibrated:
        shutil.copy2(profile, target / CALIBRATION)
        copied.append(CALIBRATION)

    adapter_config = json.loads((step_dir / "adapter_config.json").read_text())
    manifest = {
        "name": name or f"lev-{preset}-{step_dir.name}",
        "preset": preset,
        "step": int(step_dir.name.split("-")[1]) if step_dir.name.startswith("step-") else None,
        "base_model": adapter_config["base_model_name_or_path"],
        "lora_rank": adapter_config.get("r"),
        "mode_b_head": MODE_B_HEAD in copied,
        "calibrated": calibrated,
        # Record the model, prompt format, and readout required by these weights and temperatures.
        "noul_readout": "rating",
        "prompt_style": prompt_style,
        "files": copied,
        "metrics": metrics or {},
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (target / RELEASE_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    (target / MODEL_CARD).write_text(model_card(manifest))
    return manifest


def model_card(manifest: dict) -> str:
    """Build Hub metadata and usage instructions for a release."""
    metrics = manifest.get("metrics") or {}
    metric_lines = "\n".join(f"| {k} | {v} |" for k, v in metrics.items()) or "| — | not recorded |"
    calibration = (
        "Per-bucket temperatures are included (`calibration.json`) and applied by the server."
        if manifest["calibrated"]
        else "**Uncalibrated**: no `calibration.json`; the server returns raw softmax."
    )
    adapter_note = f"step {manifest['step']}, preset `{manifest['preset']}`"
    return f"""---
license: apache-2.0
base_model: {manifest["base_model"]}
library_name: peft
tags:
  - lev
  - system-one
  - decision-model
  - calibration
---

# {manifest["name"]}

A typed, calibrated decision model: one forward pass, read logits, never generate.
LoRA adapter (r={manifest["lora_rank"]}) over `{manifest["base_model"]}`, with the
candidate-path (Mode B) head that removes the single-token option ceiling.
Wire-compatible with TypeSafe's `/v1/systemone`.

## Serve

```bash
uv sync --extra serve
uv run lev serve --checkpoint <this directory or its Hub id>
uv run levbench eval --backend lev --tasks data/s1bench --base-url http://localhost:8000
```

The server reads `lev_release.json` for the base model and the prompt style
(`{manifest["prompt_style"]}`) the adapter trained under. Questions route to
label-token readout (Mode A) while the tokenizer can express one single-token
code per option -- codes that split into two tokens are skipped -- and to the
candidate-path head (Mode B) above that. Mode A trains on sets up to the
tokenizer's limit; the head trains on the full large taxonomies.
{calibration}

## What is in here

| file | role |
|---|---|
| `adapter_model.safetensors`, `adapter_config.json` | LoRA adapter ({adapter_note}) |
| `mode_b_head.pt` | candidate-path matching head |
| `tokenizer*` | the tokenizer the label-token readout was verified against |
| `calibration.json` | temperatures per question type, readout mode and Choice option-count band |
| `lev_release.json` | this manifest |

## Measured

| metric | value |
|---|---|
{metric_lines}

Numbers on S1Bench, the harness and the decision log are in the repository.
"""


def publish(release_dir: str | Path, repo_id: str, *, private: bool = False) -> str:
    """Upload a release using HF authentication; create the repo if needed and return its URL."""
    from huggingface_hub import HfApi

    folder = Path(release_dir)
    if not (folder / RELEASE_MANIFEST).is_file():
        raise FileNotFoundError(
            f"{folder} has no {RELEASE_MANIFEST}; build it with `lev release build` first"
        )
    manifest = json.loads((folder / RELEASE_MANIFEST).read_text())
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    message = f"{manifest['name']}: step {manifest['step']} of preset {manifest['preset']}"
    api.upload_folder(
        folder_path=str(folder), repo_id=repo_id, repo_type="model", commit_message=message
    )
    return f"https://huggingface.co/{repo_id}"
