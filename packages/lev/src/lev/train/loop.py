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

from .collate import DecisionCollator, ModeBatcher
from .config import TrainConfig


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
    # `torch.where`, not a plain product: padded candidate slots carry a logit of
    # -inf so they win no softmax mass, and their target probability is 0, so the
    # product is `0 * -inf` = NaN. One padded slot anywhere in the batch poisons
    # the mean, and because the backward pass still runs, the symptom is a loss
    # of nan rather than a crash -- observed on the first smoke run.
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
        # The head is kept at full precision beside a bf16 backbone, so the
        # hidden states have to be cast *up* to the head's dtype. Casting the
        # other way -- to the activations' dtype, which is what this line did
        # first -- raises on any GPU run and is invisible on CPU, where the
        # backbone is fp32 already and the two dtypes coincide.
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
        head = head.to(device=_device_of(model), dtype=torch.float32)
    return head


def _device_of(model):
    return next(model.parameters()).device


def _to_device(batch, device):
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

    # The guard again, at the last possible moment. The manifest records what was
    # loaded months ago on another machine; re-checking it here is cheap and is
    # the only place that catches a hand-edited mixture.
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
) -> dict:
    """Train, checkpointing periodically so a long run survives a preemption.

    `resume_from` restores **weights only** -- the LoRA adapter and the Mode B
    head. The optimiser moments and the step counter are not restored, so a
    resumed run repeats the warmup and starts its cosine schedule over. That is
    a deliberate limit rather than an oversight: at the measured ~2 h for the 4B
    preset (ADR-016) the cost of redoing a warmup is small, and a half-restored
    optimiser is harder to reason about than a clean restart from good weights.
    Revisit it if a preset ever runs long enough for that trade to flip.
    """
    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    config.validate()

    # Everything cheap and fallible happens before the model is touched.
    data = prepare_data(config, data_dir)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    model, tokenizer = build_model(config, model_cache)
    head = build_head(config, model) if config.train_mode_b_head else None
    if resume_from:
        load_checkpoint(model, head, resume_from)
    device = _device_of(model)

    collator = DecisionCollator(tokenizer, max_seq_len=config.max_seq_len)
    batcher = ModeBatcher(
        tokenizer,
        batch_size=config.per_device_batch,
        bucket_window=config.bucket_window,
        seed=config.seed,
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
    progress = ProgressLog(total_steps, config.log_every)
    step = 0
    last_saved = -1
    model.train()

    for epoch in range(config.epochs):
        order = list(train)
        # Shuffled before bucketing so window membership differs per epoch; the
        # batcher then sorts within each window and shuffles the batch order.
        rng.shuffle(order)
        for group in batcher(order, epoch=epoch):
            batch = _to_device(collator(group), device)
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

            if config.checkpoint_every and step % config.checkpoint_every == 0:
                save_checkpoint(model, head, tokenizer, output, step, on_checkpoint)
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
        f"done: {step:,} steps in {_hms(time.monotonic() - progress.start)} -> {output}",
        flush=True,
    )
    return summary


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


class ProgressLog:
    """Periodic one-line progress, because a silent run looks like a hung one.

    On Modal the only window into a running job is stdout, and Python
    block-buffers stdout when it is not a tty -- so an unflushed `print` shows
    nothing for hours and then everything at once. Every write here is flushed.

    Losses are reported per mode over the window rather than cumulatively: the
    two readouts sit at different scales (Mode B starts near `ln(K)`), so a
    single blended average hides which one is actually moving.

    Throughput counts the tokens actually fed to the model, taken from the
    attention mask. Deriving it from `config.avg_tokens_per_example` instead
    would report the plan rather than the run -- and that constant was wrong by
    10x once already (ADR-016), which is exactly the error a throughput readout
    should be able to expose.
    """

    def __init__(self, total_steps: int, log_every: int):
        self.total = total_steps
        self.every = max(1, log_every)
        self.start = time.monotonic()
        self.window: dict[str, list[float]] = {}
        self.tokens = 0

    def record(self, step: int, loss: float, mode: str, lr: float, tokens: int = 0) -> None:
        self.window.setdefault(mode, []).append(loss)
        self.tokens += tokens
        if (step + 1) % self.every and step + 1 != self.total:
            return

        done = step + 1
        # A coarse clock can report zero on a fast first window; never divide by it.
        elapsed = max(time.monotonic() - self.start, 1e-9)
        rate = done / elapsed
        remaining = (self.total - done) / rate if rate else 0.0
        losses = "  ".join(f"{m}={sum(v) / len(v):.4f}" for m, v in sorted(self.window.items()))
        print(
            f"step {done:>6}/{self.total}  {100 * done / self.total:5.1f}%  "
            f"{losses}  lr={lr:.2e}  {rate:.2f} it/s  "
            f"{self.tokens / elapsed:,.0f} tok/s  "
            f"elapsed {_hms(elapsed)}  eta {_hms(remaining)}{_gpu_mem()}",
            flush=True,
        )
        self.window.clear()


def _hms(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _gpu_mem() -> str:
    import torch

    if not torch.cuda.is_available():
        return ""
    return f"  mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G"


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
        head.load_state_dict(torch.load(head_file, map_location=_device_of(model)))
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
