"""Reading and writing training checkpoints.

A checkpoint is the LoRA adapter, the Mode B head and the tokenizer together.
All three, because an adapter without the head it was trained beside cannot
serve Mode B, and a tokenizer mismatch silently changes which label token ids
the readout reads -- a failure that produces plausible numbers.

Separate from `loop` so that `lev.server` and `lev.train.calibration_run`, which
only ever *read* a checkpoint, do not have to import the training loop to do it.
"""

from __future__ import annotations

from pathlib import Path


def resolve_checkpoint(path: str | Path) -> Path:
    """Accept either a `step-N` directory or the parent holding several.

    `save_checkpoint` writes `<output_dir>/step-<n>`, but every caller naturally
    names `<output_dir>` -- the preset's output directory is what appears in the
    config, in the Modal volume layout and in the docs. Resolving the newest
    step here means one spelling works everywhere, and "newest" is by step
    number rather than mtime because a resumed run rewrites older directories.
    """
    source = Path(path)
    if (source / "adapter_model.safetensors").is_file():
        return source
    steps = sorted(
        (d for d in source.glob("step-*") if (d / "adapter_model.safetensors").is_file()),
        key=lambda d: int(d.name.split("-")[1]),
    )
    if not steps:
        raise FileNotFoundError(
            f"no checkpoint under {source}: expected adapter weights there or in "
            f"a step-N subdirectory"
        )
    return steps[-1]


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


def save_checkpoint(model, head, tokenizer, output: Path, step: int, on_checkpoint=None) -> Path:
    """Write adapters, head and tokenizer together.

    All three, because a LoRA adapter without the head it was trained beside
    cannot serve Mode B, and a tokenizer mismatch silently changes which label
    token ids the readout reads -- a failure that produces plausible numbers.
    """
    import torch

    path = output / f"step-{step}"
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    if head is not None:
        torch.save(head.state_dict(), path / "mode_b_head.pt")
    if on_checkpoint is not None:
        on_checkpoint()
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    print(f"  checkpoint -> {path}  ({size / 2**20:.0f} MB)", flush=True)
    return path
