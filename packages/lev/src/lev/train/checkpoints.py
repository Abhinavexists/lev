"""Read and write adapter, head and tokenizer checkpoints, with optional resume state."""

from __future__ import annotations

from pathlib import Path

TRAINING_STATE = "training_state.pt"
MODE_B_HEAD = "mode_b_head.pt"
CALIBRATION = "calibration.json"


def latest_checkpoint(output: str | Path) -> Path | None:
    """The newest `step-N` under `output` that holds adapter weights, or None.

    By step number, not mtime: a resumed run rewrites older directories.
    """
    source = Path(output)
    steps = sorted(
        (d for d in source.glob("step-*") if (d / "adapter_model.safetensors").is_file()),
        key=lambda d: int(d.name.split("-")[1]),
    )
    return steps[-1] if steps else None


def fetch_checkpoint(spec: str | Path, cache_dir: str | None = None) -> Path:
    """A local directory as-is; a Hub id (`hf://org/name` or `org/name`) downloaded.

    Anything on disk is local; otherwise only `org/name` is treated as a Hub
    repo, so a mistyped local path fails instead of reaching the Hub.
    """
    local = Path(spec)
    if local.exists():
        return local
    repo = str(spec).removeprefix("hf://")
    if repo.count("/") != 1 or repo.startswith("/") or repo.startswith("."):
        raise FileNotFoundError(f"{spec!r} is neither a local path nor a Hub id like org/name")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, repo_type="model", cache_dir=cache_dir))


def resolve_checkpoint(path: str | Path, cache_dir: str | None = None) -> Path:
    """Accept a `step-N` directory, the parent holding several, a flat release
    directory, or a Hub id for one.

    `save_checkpoint` writes `<output_dir>/step-<n>`, while callers name
    `<output_dir>`, so the newest step is resolved here.
    """
    source = fetch_checkpoint(path, cache_dir)
    if (source / "adapter_model.safetensors").is_file():
        return source
    latest = latest_checkpoint(source)
    if latest is None:
        raise FileNotFoundError(
            f"no checkpoint under {source}: expected adapter weights there or in "
            f"a step-N subdirectory"
        )
    return latest


def load_training_state(path: str | Path) -> dict | None:
    """The optimiser, schedule and position saved beside the weights, if any.

    None for a weights-only checkpoint (including those from before ADR-021);
    the caller then starts the optimiser and schedule fresh.
    """
    import torch

    file = resolve_checkpoint(path) / TRAINING_STATE
    if not file.is_file():
        return None
    # `weights_only=False`: the payload holds the RNG state (a tuple). The file
    # is written by `save_checkpoint`, never downloaded.
    return torch.load(file, map_location="cpu", weights_only=False)


def load_checkpoint(model, head, path: str | Path) -> None:
    """Restore adapter and head weights into existing parameter tensors.

    Keeping the tensors preserves optimiser references. A missing or unexpected
    head is an error, since dropping it would discard Mode B training.
    """
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    source = resolve_checkpoint(path)
    set_peft_model_state_dict(model, load_file(str(source / "adapter_model.safetensors")))

    head_file = source / MODE_B_HEAD
    if head is not None:
        if not head_file.is_file():
            raise FileNotFoundError(
                f"{source} has adapter weights but no {head_file.name}. Resuming "
                f"would reinitialise the Mode B head and quietly throw away every "
                f"Mode B step taken before the interruption. Pass "
                f"`train_mode_b_head=False` if that is genuinely what you want."
            )
        device = next(model.parameters()).device
        head.load_state_dict(torch.load(head_file, map_location=device))
    elif head_file.is_file():
        raise ValueError(
            f"{source} carries a Mode B head but this run has "
            f"`train_mode_b_head=False`; it would be dropped."
        )


def save_checkpoint(
    model,
    head,
    tokenizer,
    output: Path,
    step: int,
    on_checkpoint=None,
    state: dict | None = None,
) -> Path:
    """Write adapter, head and tokenizer, then optional training state.

    Files are written individually; an interrupted save may be incomplete.
    """
    import torch

    path = output / f"step-{step}"
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    if head is not None:
        torch.save(head.state_dict(), path / MODE_B_HEAD)
    if state is not None:
        torch.save(state, path / TRAINING_STATE)
    if on_checkpoint is not None:
        on_checkpoint()
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    print(f"  checkpoint -> {path}  ({size / 2**20:.0f} MB)", flush=True)
    return path
