"""Reading and writing training checkpoints.

A checkpoint is the LoRA adapter, the Mode B head and the tokenizer together,
plus the training state -- optimiser moments, schedule position, step and
epoch, and the RNG state that reproduces the epoch's data order. The first
three are what a server needs; the rest is what lets a preempted run continue
from where it stopped instead of replaying it from good weights (ADR-021).
An adapter without the head it was trained beside cannot serve Mode B, and a
tokenizer mismatch silently changes which label token ids the readout reads --
a failure that produces plausible numbers.

Separate from `loop` so that `lev.server` and `lev.train.calibration_run`, which
only ever *read* a checkpoint, do not have to import the training loop to do it.
"""

from __future__ import annotations

from pathlib import Path

TRAINING_STATE = "training_state.pt"


def latest_checkpoint(output: str | Path) -> Path | None:
    """The newest `step-N` under `output` that holds adapter weights, or None.

    Newest by step number rather than mtime, because a resumed run rewrites
    older directories.
    """
    source = Path(output)
    steps = sorted(
        (d for d in source.glob("step-*") if (d / "adapter_model.safetensors").is_file()),
        key=lambda d: int(d.name.split("-")[1]),
    )
    return steps[-1] if steps else None


def fetch_checkpoint(spec: str | Path, cache_dir: str | None = None) -> Path:
    """A local directory as-is; a Hub id (`hf://org/name` or `org/name`) downloaded.

    Anything that exists on disk is local. Otherwise a string with exactly one
    slash and no path separators beyond it is treated as a Hub repo, so a typo
    in a local path fails as a missing repo rather than being silently created.
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

    `save_checkpoint` writes `<output_dir>/step-<n>`, but every caller naturally
    names `<output_dir>` -- the preset's output directory is what appears in the
    config, in the Modal volume layout and in the docs. Resolving the newest
    step here means one spelling works everywhere.
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

    None for a checkpoint written before ADR-021 or by a run that saved weights
    only; the caller then starts the optimiser and schedule fresh from the
    restored weights, which is what every resume did before.
    """
    import torch

    file = resolve_checkpoint(path) / TRAINING_STATE
    if not file.is_file():
        return None
    # `weights_only=False`: the payload carries the RNG state (a tuple), not
    # just tensors. The file is ours, written by `save_checkpoint`.
    return torch.load(file, map_location="cpu", weights_only=False)


def load_checkpoint(model, head, path: str | Path) -> None:
    """Restore adapter and head weights into an already-built model.

    Not `model.load_adapter(path, adapter_name="default")`: `get_peft_model`
    has already created an adapter under that name, so loading another one
    there either errors or leaves two. Writing the state dict into the existing
    adapter is the operation actually wanted, and it keeps the optimiser's
    parameter list valid -- it was built from these exact tensors.

    A missing head file is fatal rather than ignored. Resuming a run with a
    freshly initialised Mode B head would look like training and would silently
    discard every Mode B step taken before the preemption.
    """
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    source = resolve_checkpoint(path)
    set_peft_model_state_dict(model, load_file(str(source / "adapter_model.safetensors")))

    head_file = source / "mode_b_head.pt"
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
    """Write adapters, head and tokenizer together, and the training state.

    All three weights files, because a LoRA adapter without the head it was
    trained beside cannot serve Mode B, and a tokenizer mismatch silently
    changes which label token ids the readout reads -- a failure that produces
    plausible numbers. `state` is what `run_training` needs to continue from
    this exact step; it is written last, so a checkpoint interrupted mid-write
    degrades to weights-only rather than to a corrupt state file.
    """
    import torch

    path = output / f"step-{step}"
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    if head is not None:
        torch.save(head.state_dict(), path / "mode_b_head.pt")
    if state is not None:
        torch.save(state, path / TRAINING_STATE)
    if on_checkpoint is not None:
        on_checkpoint()
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    print(f"  checkpoint -> {path}  ({size / 2**20:.0f} MB)", flush=True)
    return path
