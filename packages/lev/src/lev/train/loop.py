"""The LoRA fine-tune loop.

Objective is a proper scoring rule -- cross-entropy plus a Brier term, plus an
ordinal term for Score under Mode B. Calibration is the thing we are competing on,
so it is in the loss rather than bolted on at the end. (A temperature is *also*
fitted afterwards; the two are complementary, not alternatives.)

Run `make smoke` (or `modal run modal/app.py::smoke`) before the real preset --
it exercises data, routing, collation, both readouts, loss and checkpoint write on
a small backbone in a few minutes.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from pathlib import Path

from .checkpoints import latest_checkpoint, load_checkpoint, load_training_state, save_checkpoint
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
        # Right-padding is what the collator assumes, and `last_positions` is
        # derived from the attention mask, so the pad token's identity does not
        # reach the loss. It only has to exist.
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # bfloat16 on CPU is supported but glacial, and `make smoke-local` is
    # documented as a thing you run on a laptop. Fall back rather than let the
    # documented command appear to hang.
    dtype = getattr(torch, config.dtype)
    if dtype is torch.bfloat16 and not torch.cuda.is_available():
        print("no CUDA; training in float32 instead of bfloat16")
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        dtype=dtype,
        # `device_map="auto"` is for the multi-GPU case and is actively harmful
        # elsewhere: on an Apple machine accelerate dispatches to MPS and the
        # process dies with SIGSEGV part-way through loading, which reads as a
        # corrupt download rather than a placement bug. One H100 does not need
        # sharding anyway, and `make smoke` has to run on a laptop.
        device_map="auto" if torch.cuda.device_count() > 1 else None,
        cache_dir=model_cache,
    )
    if torch.cuda.is_available() and torch.cuda.device_count() == 1:
        model = model.cuda()

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Required, or checkpointed activations arrive with no grad_fn and the
        # LoRA adapters receive no gradient at all -- a silent no-op run.
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
    """Cross-entropy + Brier (+ ordinal for Score), over a padded candidate set.

    Cross-entropy alone optimises the argmax and tolerates overconfidence; the
    Brier term penalises the whole probability vector, which is what makes the
    result calibratable. Both are proper scoring rules, and calibration is the
    one axis this project competes on, so they belong in the loss rather than
    only in a temperature fitted afterwards.

    Args:
        logits: `(B, K)`, already `-inf` at padded candidate slots.
        targets: `(B,)` gold index, or `IGNORE_INDEX` where the row is an
            abstain example with no single correct answer.
        ordinal: `(B,)` bool, True for Score rows, or a bare bool.
        soft_targets: `(B, K)` target distribution, read for the rows where
            `targets` is `IGNORE_INDEX`. Those rows carry a uniform vector:
            with no evidence in the state, every candidate is equally supported,
            and spreading mass is the calibrated answer rather than a hedge.
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

    # One cross-entropy over both kinds of row: with a one-hot target this is
    # exactly F.cross_entropy, so the hard rows are unaffected by supporting
    # the soft ones.
    #
    # `torch.where`, not a plain product: a padded slot carries a logit of -inf
    # and a target of 0, so the product is `0 * -inf` = NaN. One padded slot
    # poisons the batch mean, and backward still runs, so the symptom is a nan
    # loss rather than a crash.
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

    Mode A reads the vocabulary logits at the answer boundary and keeps the
    label-token ids. No parameters are added, which is why it works on a stock
    checkpoint before any training at all.

    Mode B reads hidden states instead and scores candidate *text* through the
    matching head. The candidate strings were pooled by the collator, so a
    77-option question costs one extra short forward rather than 77.
    """
    import torch

    from ..router import Mode

    if batch.mode is Mode.LABEL_TOKEN:
        out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
        rows = torch.arange(batch.size, device=out.logits.device)
        vocab = out.logits[rows, batch.last_positions]  # (B, V)
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
        # The head is kept at full precision beside a bf16 backbone, so hidden
        # states must be cast *up* to the head's dtype. Casting down to the
        # activations' dtype instead raises on any GPU run, and is invisible on
        # CPU where the backbone is fp32 and the two dtypes coincide.
        target_dtype = next(head.parameters()).dtype
        logits = head(question_repr.to(target_dtype), reprs.to(target_dtype), batch.candidate_mask)

    # Padded slots must not compete for softmax mass. Mode B already masks, but
    # Mode A's gather returns a real (wrong) vocabulary logit in a padded slot.
    return logits.float().masked_fill(batch.candidate_mask.to(logits.device), float("-inf"))


def build_head(config: TrainConfig, model=None):
    """The Mode B matching head, at full precision beside a quantised backbone."""
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
    from dataclasses import fields as _fields

    import torch

    for f in _fields(batch):
        value = getattr(batch, f.name)
        if isinstance(value, torch.Tensor):
            setattr(batch, f.name, value.to(device))
    return batch


def prepare_data(config: TrainConfig, data_dir: str):
    """Resolve and validate the training mixture. Must run before the model loads.

    Loading a multi-GB checkpoint and *then* discovering the data is missing costs
    real GPU minutes on Modal, and it is the failure that happens most often.
    Raises with an actionable message rather than returning empty: a run that
    silently trains on nothing wastes the whole budget.
    """
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

    # The guard again, at the last possible moment. The manifest only records
    # what some earlier run loaded, possibly on another machine; re-checking is
    # cheap and is the only place that catches a hand-edited mixture.
    manifest_path = root / MANIFEST
    if manifest_path.is_file():
        import json

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

    A run picks up where it stopped. With no `resume_from`, the newest `step-N`
    under `config.output_dir` is resumed automatically -- `fresh=True` starts
    over. A checkpoint carries the optimiser moments, the schedule position,
    the step and epoch, and the RNG state that reproduces the epoch's data
    order, so the resumed run continues at step N through the batches it had
    not yet seen, on the learning rate it had reached. A checkpoint written
    without that state (an older run) restores weights only and starts the
    optimiser and schedule fresh, which is what every resume did before ADR-021.
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

    model, tokenizer = build_model(config, model_cache)
    head = build_head(config, model) if config.train_mode_b_head else None
    state = None
    if resume_from:
        load_checkpoint(model, head, resume_from)
        state = load_training_state(resume_from)
    device = device_of(model)

    # One cache, so a question is routed once for the whole run rather than
    # once for the batcher and again for the collator.
    routes = RouteCache(tokenizer, config.max_label_options)
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
    # The scheduler advances once per *optimiser* step, not once per micro-batch,
    # so it must be sized in optimiser steps. Sizing it in micro-steps makes the
    # cosine finish `grad_accum` times early and the tail of training run at a
    # learning rate of zero.
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
        # Captured before the shuffle: a checkpoint inside this epoch stores it,
        # so the resumed run reproduces the same order and skips what it saw.
        rng_before_epoch = rng.getstate()
        order = list(train)
        # Shuffled before bucketing so window membership differs per epoch; the
        # batcher then sorts within each window and shuffles the batch order.
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
                # Alongside the weights, not only at the end: a run that dies at
                # step 17,000 should still leave its loss curve behind.
                _write_history(output, history, step, str(output))
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    # Only if the interval save did not already land on this exact step --
    # otherwise a total that happens to be a multiple of `checkpoint_every`
    # writes the same adapter twice and commits the Volume twice for nothing.
    if step != last_saved:
        save_checkpoint(model, head, tokenizer, output, step, on_checkpoint)
    summary = _write_history(output, history, step, str(output))
    print(
        f"done: {step:,} steps in {hms(time.monotonic() - progress.start)} -> {output}",
        flush=True,
    )
    return summary


def _read_history(output: Path) -> list[dict]:
    """The loss curve a previous run left behind, so a resumed run extends it."""
    file = output / "history.json"
    if not file.is_file():
        return []
    return json.loads(file.read_text()).get("history", [])


def _write_history(output: Path, history: list[dict], step: int, output_dir: str) -> dict:
    summary = {
        "steps": step,
        "final_loss": history[-1]["loss"] if history else None,
        "first_loss": history[0]["loss"] if history else None,
        "output_dir": output_dir,
        "history": history,
    }
    (output / "history.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
