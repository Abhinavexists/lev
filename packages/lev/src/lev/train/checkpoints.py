"""Read and write adapter, head and tokenizer checkpoints, with optional resume state."""

from __future__ import annotations

from pathlib import Path

ADAPTER_WEIGHTS = "adapter_model.safetensors"
TRAINING_STATE = "training_state.pt"
MODE_B_HEAD = "mode_b_head.pt"
CALIBRATION = "calibration.json"


def latest_checkpoint(output: str | Path) -> Path | None:
    """Return the highest numbered step with adapter weights, regardless of mtime."""
    source = Path(output)
    steps = sorted(
        (d for d in source.glob("step-*") if (d / ADAPTER_WEIGHTS).is_file()),
        key=lambda d: int(d.name.split("-")[1]),
    )
    return steps[-1] if steps else None


def fetch_checkpoint(spec: str | Path, cache_dir: str | None = None) -> Path:
    """Use an existing local path or download a Hub id; report ambiguous missing paths clearly."""
    local = Path(spec)
    if local.exists():
        return local
    repo = str(spec).removeprefix("hf://")
    if repo.count("/") != 1 or repo.startswith(("/", ".")):
        raise FileNotFoundError(f"{spec!r} is neither a local path nor a Hub id like org/name")
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        return Path(snapshot_download(repo, repo_type="model", cache_dir=cache_dir))
    except RepositoryNotFoundError as error:
        raise FileNotFoundError(
            f"no local directory {spec!r}, and no Hub repo {repo!r} (or no access to it)"
        ) from error


def resolve_checkpoint(path: str | Path, cache_dir: str | None = None) -> Path:
    """Resolve a step directory, training output, flat release, or Hub id."""
    source = fetch_checkpoint(path, cache_dir)
    if (source / ADAPTER_WEIGHTS).is_file():
        return source
    latest = latest_checkpoint(source)
    if latest is None:
        raise FileNotFoundError(
            f"no checkpoint under {source}: expected adapter weights there or in "
            f"a step-N subdirectory"
        )
    return latest


def load_training_state(path: str | Path) -> dict | None:
    """Load resume state, or return None for a weights-only checkpoint."""
    import torch

    file = resolve_checkpoint(path) / TRAINING_STATE
    if not file.is_file():
        return None
    # Use weights_only to avoid arbitrary unpickling of downloaded resume state.
    return torch.load(file, map_location="cpu", weights_only=True)


def load_checkpoint(model, head, path: str | Path) -> None:
    """Restore weights in place to preserve optimizer references; reject head mismatches."""
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    source = resolve_checkpoint(path)
    set_peft_model_state_dict(model, load_file(str(source / ADAPTER_WEIGHTS)))

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
    """Write weights, tokenizer, and optional resume state; interrupted saves may be incomplete."""
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
