"""The LoRA fine-tune loop.

Objective is a proper scoring rule -- cross-entropy plus a Brier term, plus an
ordinal term for Score under Mode B. Calibration is the thing we are competing on,
so it is in the loss rather than bolted on at the end. (A temperature is *also*
fitted afterwards; the two are complementary, not alternatives.)

NOT YET RUN. Run `modal run modal/app.py::smoke` before the real preset -- it
exercises image, volumes, data, loss and checkpoint write in a few minutes.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .config import TrainConfig


def build_model(config: TrainConfig, model_cache: str | None = None):
    """Load the backbone, attach LoRA, and keep new heads at full precision."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, cache_dir=model_cache)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        dtype=getattr(torch, config.dtype),
        device_map="auto",
        cache_dir=model_cache,
    )

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


def decision_loss(logits, targets, config: TrainConfig, ordinal: bool = False):
    """Cross-entropy + Brier (+ ordinal for Score under Mode B).

    Cross-entropy alone optimises the argmax and tolerates overconfidence; the
    Brier term penalises the whole probability vector, which is what makes the
    result calibratable.
    """
    import torch.nn.functional as F

    ce = F.cross_entropy(logits, targets)
    probs = logits.softmax(dim=-1)
    onehot = F.one_hot(targets, logits.size(-1)).to(probs.dtype)
    brier = ((probs - onehot) ** 2).sum(dim=-1).mean()

    loss = ce + config.brier_weight * brier
    if ordinal:
        from ..readout.mode_b import ordinal_penalty

        loss = loss + config.ordinal_weight * ordinal_penalty(logits, targets)
    return loss


def run_training(
    config: TrainConfig,
    data_dir: str,
    model_cache: str | None = None,
    resume_from: str | None = None,
    on_checkpoint: Callable[[], None] | None = None,
) -> dict:
    """Train, checkpointing periodically so a long run survives a preemption."""
    config.validate()
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    model, tokenizer = build_model(config, model_cache)
    raise NotImplementedError(
        "The data pipeline is the hacking-phase task: implement loaders in "
        "lev.data.mixture, then batch them through decision_loss above. "
        "Everything around it -- model, LoRA, loss, checkpointing, Modal wiring -- "
        "is in place. See docs/TRAINING.md."
    )
