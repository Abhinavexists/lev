"""Fine-tune the backbone and candidate head, with checkpointing and resume.

The objective combines cross-entropy, Brier and an ordinal term.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import Any

from ..prompt import Style
from .checkpoints import (
    CALIBRATION,
    latest_checkpoint,
    load_checkpoint,
    load_training_state,
    save_checkpoint,
)
from .collate import DecisionCollator, ModeBatcher, RouteCache
from .config import TrainConfig
from .progress import ProgressLog, hms


def build_model(config: TrainConfig, model_cache: str | None = None):
    """Load the backbone, attach LoRA, and keep new heads at full precision."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, cache_dir=model_cache)
    if tokenizer.pad_token_id is None:
        # Rows are right-padded and `last_positions` comes from the attention mask,
        # so the pad token only has to exist.
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # bfloat16 on CPU is glacial, and `make smoke-local` runs on a laptop.
    dtype = getattr(torch, config.dtype)
    if dtype is torch.bfloat16 and not torch.cuda.is_available():
        print("no CUDA; training in float32 instead of bfloat16")
        dtype = torch.float32

    # The adapter wrapper and backbone expose different static types.
    model: Any = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        dtype=dtype,
        # Multi-GPU only: on an Apple machine accelerate dispatches to MPS and
        # the process dies with SIGSEGV part-way through loading.
        device_map="auto" if torch.cuda.device_count() > 1 else None,
        cache_dir=model_cache,
    )
    if torch.cuda.is_available() and torch.cuda.device_count() == 1:
        model = model.cuda()

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Without it, checkpointed activations have no grad_fn and the LoRA
        # adapters get no gradient: a silent no-op run.
        model.enable_input_require_grads()

    if config.use_lora:
        model = get_peft_model(
            model,
            LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=list(config.lora_targets),
                task_type="CAUSAL_LM",
            ),
        )
        model.print_trainable_parameters()

    return model, tokenizer


def decision_loss(logits, targets, config: TrainConfig, ordinal=False, soft_targets=None):
    """Cross-entropy, Brier and ordinal loss over padded candidate sets.

    Args:
        logits: `(B, K)`, with `-inf` at padded slots.
        targets: `(B,)` gold indices; `IGNORE_INDEX` for abstain rows.
        ordinal: `(B,)` bool for ordered targets, or a scalar bool.
        soft_targets: `(B, K)` distributions for abstain rows.
    """
    import torch
    import torch.nn.functional as F

    from .collate import IGNORE_INDEX

    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()

    hard = targets != IGNORE_INDEX
    target_dist = torch.zeros_like(probs)
    if hard.any():
        target_dist[hard] = F.one_hot(targets[hard], logits.size(-1)).to(probs.dtype)
    if soft_targets is not None and (~hard).any():
        target_dist[~hard] = soft_targets[~hard].to(probs.dtype)

    # One cross-entropy over both kinds of row; with a one-hot target it equals
    # F.cross_entropy. `torch.where`, not a product: a padded slot has logit -inf
    # and target 0, and `0 * -inf` is NaN, which poisons the batch mean.
    weighted = torch.where(target_dist > 0, target_dist * log_probs, torch.zeros_like(log_probs))
    ce = -weighted.sum(dim=-1).mean()
    brier = ((probs - target_dist) ** 2).sum(dim=-1).mean()
    loss = ce + config.brier_weight * brier

    if isinstance(ordinal, bool):
        ordinal = torch.full((logits.size(0),), ordinal, dtype=torch.bool, device=logits.device)
    rows = ordinal & hard
    if rows.any():
        from ..readout.mode_b import ordinal_penalty

        loss = loss + config.ordinal_weight * ordinal_penalty(logits[rows], targets[rows])
    return loss


def candidate_logits(model, batch, head=None):
    """One forward pass -> `(B, K)` logits over the batch's candidate set.

    Mode A keeps the label-token ids from the vocabulary logits at the answer
    boundary. Mode B scores candidate text through the matching head; the
    collator pools candidate strings, so a 77-option question costs one extra
    short forward, not 77.
    """
    import torch

    from ..router import Mode

    if batch.mode is Mode.LABEL_TOKEN:
        # Project only answer positions to avoid a full (B, seq, vocab) allocation.
        # Rows with different answer positions must pick their own output column.
        keep, column = torch.unique(batch.last_positions, return_inverse=True)
        out = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            logits_to_keep=keep,
        )
        rows = torch.arange(batch.size, device=out.logits.device)
        vocab = out.logits[rows, column.to(out.logits.device)]  # (B, V)
        logits = vocab.gather(1, batch.candidate_token_ids)  # (B, K)
    else:
        if head is None:
            raise ValueError("a Mode B batch needs the candidate-path head")
        out = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            output_hidden_states=True,
        )
        hidden = out.hidden_states[-1]
        rows = torch.arange(batch.size, device=hidden.device)
        question_repr = hidden[rows, batch.last_positions]  # (B, H)

        pooled = model(
            input_ids=batch.candidate_input_ids,
            attention_mask=batch.candidate_attention_mask,
            output_hidden_states=True,
        ).hidden_states[-1]
        pool_rows = torch.arange(pooled.size(0), device=pooled.device)
        candidate_repr = pooled[pool_rows, batch.candidate_last_positions]  # (C, H)

        flat = batch.candidate_index.reshape(-1)
        reprs = candidate_repr[flat].view(*batch.candidate_index.shape, -1)  # (B, K, H)
        # Cast hidden states up to the fp32 head. Casting down raises on GPU and
        # is invisible on CPU, where both are fp32.
        target_dtype = next(head.parameters()).dtype
        logits = head(question_repr.to(target_dtype), reprs.to(target_dtype), batch.candidate_mask)

    # Mode B already masks padded slots; Mode A's gather returns a real (wrong)
    # vocabulary logit there.
    return logits.float().masked_fill(batch.candidate_mask.to(logits.device), float("-inf"))


def build_head(config: TrainConfig, model=None):
    """The Mode B matching head, in fp32 beside the bf16 backbone."""
    import torch

    from ..readout.mode_b import CandidatePathReadout

    hidden = config.hidden_size
    if model is not None:
        hidden = getattr(model.config, "hidden_size", hidden)
    head = CandidatePathReadout(hidden_size=hidden, proj_dim=config.mode_b_proj_dim)
    if model is not None:
        head = head.to(device=device_of(model), dtype=torch.float32)
    return head


def device_of(model):
    return next(model.parameters()).device


def to_device(batch, device):
    import torch

    for f in fields(batch):
        value = getattr(batch, f.name)
        if isinstance(value, torch.Tensor):
            setattr(batch, f.name, value.to(device))
    return batch


def prepare_data(config: TrainConfig, data_dir: str):
    """Validate data and contamination before loading the model."""
    from ..data.build import MANIFEST, load_split
    from ..data.splits import Split

    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"data directory {data_dir!r} does not exist. Build it with:\n"
            f"    uv run lev data build --out {data_dir}"
        )

    train = load_split(root, Split.TRAIN)
    if not train:
        raise ValueError(f"{root / 'train.jsonl'} is empty")

    # The guard again, at the last moment: the only place a hand-edited mixture
    # is caught.
    manifest_path = root / MANIFEST
    if manifest_path.is_file():
        from ..data.contamination import assert_clean

        manifest = json.loads(manifest_path.read_text())
        assert_clean(list(manifest.get("sources", {}).values()))

    calibration = load_split(root, Split.CALIBRATION)
    return {"train": train, "calibration": calibration}


def run_training(
    config: TrainConfig,
    data_dir: str,
    model_cache: str | None = None,
    resume_from: str | None = None,
    on_checkpoint: Callable[[], None] | None = None,
    max_steps: int | None = None,
    fresh: bool = False,
) -> dict:
    """Train, checkpointing periodically so a long run survives a preemption.

    With no `resume_from`, the newest `step-N` under `config.output_dir` is
    resumed; `fresh=True` starts over. A checkpoint carries the optimiser
    moments, schedule position, step, epoch, and the RNG state behind the
    epoch's data order, so a resumed run continues through the batches it had
    not seen, at the learning rate it had reached. Checkpoints from before
    ADR-021 restore weights only.
    """
    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    config.validate()

    # Everything cheap and fallible happens before the model is touched.
    data = prepare_data(config, data_dir)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if resume_from is None and not fresh and (found := latest_checkpoint(output)) is not None:
        resume_from = str(found)
        print(f"resuming from {found} (pass --fresh to start over)", flush=True)
    # Move, not delete, the previous run, so auto-resume (highest step) and the
    # calibration lookup cannot pick it up.
    if fresh and (moved := set_aside_previous_run(output)) is not None:
        print(f"previous run moved to {moved}", flush=True)

    model, tokenizer = build_model(config, model_cache)
    head = build_head(config, model) if config.train_mode_b_head else None
    state = None
    if resume_from:
        load_checkpoint(model, head, resume_from)
        state = load_training_state(resume_from)
    device = device_of(model)

    # One cache shared by batcher and collator, so each question routes once.
    routes = RouteCache(tokenizer, config.max_label_options, Style(config.prompt_style))
    collator = DecisionCollator(tokenizer, max_seq_len=config.max_seq_len, routes=routes)
    batcher = ModeBatcher(
        tokenizer,
        batch_size=config.per_device_batch,
        bucket_window=config.bucket_window,
        seed=config.seed,
        routes=routes,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    if head is not None:
        params += list(head.parameters())
    optimiser = AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)

    train = data["train"]
    steps_per_epoch = max(1, len(train) // (config.per_device_batch * config.grad_accum))
    total_steps = max_steps or steps_per_epoch * config.epochs
    # Sized in optimiser steps: sized in micro-steps, the cosine would finish
    # `grad_accum` times early and the tail would train at a learning rate of 0.
    optimiser_steps = max(1, total_steps // config.grad_accum)
    scheduler = get_cosine_schedule_with_warmup(
        optimiser, int(optimiser_steps * config.warmup_ratio), optimiser_steps
    )

    trainable = sum(p.numel() for p in params)
    print(
        f"training {len(train):,} examples x {config.epochs} epochs "
        f"= {total_steps:,} steps at batch {config.per_device_batch}"
        f"{' x ' + str(config.grad_accum) + ' accum' if config.grad_accum > 1 else ''}  "
        f"| {trainable / 1e6:.1f}M trainable params on {device}",
        flush=True,
    )

    rng = random.Random(config.seed)
    history: list[dict] = []
    step = 0
    start_epoch = 0
    skip_groups = 0
    last_saved = -1
    if state is not None:
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        step, start_epoch, skip_groups = state["step"], state["epoch"], state["step_in_epoch"]
        rng.setstate(state["rng_before_epoch"])
        history = [h for h in _read_history(output) if h["step"] < step]
        last_saved = step
        print(
            f"resumed at step {step:,}/{total_steps:,}: epoch {start_epoch}, "
            f"{skip_groups:,} batches in, lr {scheduler.get_last_lr()[0]:.2e}",
            flush=True,
        )
    elif resume_from:
        print("checkpoint carries weights only; optimiser and schedule start fresh", flush=True)
    progress = ProgressLog(total_steps, config.log_every)
    model.train()

    for epoch in range(start_epoch, config.epochs):
        if step >= total_steps:
            break
        # Before the shuffle, so a resumed run reproduces this epoch's order.
        rng_before_epoch = rng.getstate()
        order = list(train)
        # Shuffled before bucketing so windows differ per epoch.
        rng.shuffle(order)
        groups = batcher(order, epoch=epoch)
        step_in_epoch = 0
        if epoch == start_epoch and skip_groups:
            for _ in range(skip_groups):
                next(groups, None)
            step_in_epoch = skip_groups
            skip_groups = 0
        for group in groups:
            batch = to_device(collator(group), device)
            logits = candidate_logits(model, batch, head)
            loss = decision_loss(
                logits,
                batch.targets,
                config,
                ordinal=batch.ordinal,
                soft_targets=batch.soft_targets,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"loss is {float(loss)} at step {step} (mode {batch.mode.value}, "
                    f"{batch.size} rows, K={batch.candidate_mask.size(1)}). Training "
                    f"on a non-finite loss corrupts the adapter silently, so this "
                    f"stops rather than continues."
                )
            (loss / config.grad_accum).backward()

            if (step + 1) % config.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimiser.step()
                scheduler.step()
                optimiser.zero_grad(set_to_none=True)

            value = float(loss.detach())
            history.append({"step": step, "epoch": epoch, "loss": value, "mode": batch.mode.value})
            progress.record(
                step,
                value,
                batch.mode.value,
                scheduler.get_last_lr()[0],
                tokens=int(batch.attention_mask.sum()),
            )
            step += 1
            step_in_epoch += 1

            if config.checkpoint_every and step % config.checkpoint_every == 0:
                save_checkpoint(
                    model,
                    head,
                    tokenizer,
                    output,
                    step,
                    on_checkpoint,
                    state={
                        "step": step,
                        "epoch": epoch,
                        "step_in_epoch": step_in_epoch,
                        "optimiser": optimiser.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "rng_before_epoch": rng_before_epoch,
                    },
                )
                last_saved = step
                # So a run that dies late still leaves its loss curve.
                _write_history(output, history, step)
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    if step != last_saved:
        save_checkpoint(model, head, tokenizer, output, step, on_checkpoint)
    summary = _write_history(output, history, step)
    print(
        f"done: {step:,} steps in {hms(time.monotonic() - progress.start)} -> {output}",
        flush=True,
    )
    return summary


def set_aside_previous_run(output: Path) -> Path | None:
    """Move prior checkpoints, history and calibration under `superseded-<utc>`.

    Return the directory, or None if there was nothing to move.
    """
    stale = list(output.glob("step-*")) + [
        output / name for name in ("history.json", CALIBRATION) if (output / name).exists()
    ]
    if not stale:
        return None
    target = output / f"superseded-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    target.mkdir()
    for path in stale:
        path.rename(target / path.name)
    return target


def _read_history(output: Path) -> list[dict]:
    file = output / "history.json"
    if not file.is_file():
        return []
    return json.loads(file.read_text()).get("history", [])


def _write_history(output: Path, history: list[dict], step: int) -> dict:
    summary = {
        "steps": step,
        "final_loss": history[-1]["loss"] if history else None,
        "first_loss": history[0]["loss"] if history else None,
        "output_dir": str(output),
        "history": history,
    }
    (output / "history.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
