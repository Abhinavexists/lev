"""Training presets and approximate H100 compute and memory budgets.

Estimate assumptions are documented in docs/ARCHITECTURE.md §5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Sustained throughput measured on the released 4b-instruct run: 18,750 steps in
# 7.8 h of training (~1.5 s/step, from checkpoint times), i.e. 2.46e18 FLOPs over
# 28,100 s. Padded batches, per-step overhead and the chat template's extra tokens
# are all inside it; peak bf16 is ~990 TFLOP/s. The plain-prompt 4B runs took
# 4h50 and 5h53, so this reads high for them.
H100_EFFECTIVE_FLOPS = 8.75e13

H100_VRAM_GB = 80.0

# Forward+backward is ~6*N FLOPs/token; checkpointing recomputes activations for
# roughly a third more. True under LoRA too: the backward pass still traverses
# the frozen weights to reach the adapters.
FLOPS_PER_PARAM_PER_TOKEN = 8.0


@dataclass
class TrainConfig:
    model_id: str = "Qwen/Qwen3.5-4B-Base"
    params_b: float = 4.0
    hidden_size: int = 2560
    dtype: str = "bfloat16"

    # LoRA leaves room for activations on one H100 (§5.2).
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    # The new head trains in full even when LoRA freezes the backbone.
    train_mode_b_head: bool = True
    mode_b_proj_dim: int = 512
    # None uses the tokenizer limit; larger option sets train Mode B (ADR-026).
    max_label_options: int | None = None
    # Serving must match; the release manifest records this format.
    prompt_style: Literal["plain", "chat"] = "plain"

    # Data.
    n_examples: int = 200_000
    # Re-measure with `lev plan --data <dir>` after changing the mixture (ADR-016).
    avg_tokens_per_example: int = 128
    # A truncation cap, not the training length: batches pad to their longest row.
    max_seq_len: int = 2_048
    schema_first_fraction: float = 0.5
    # Unanswerable examples with a uniform target, teaching the model to spread
    # mass instead of guessing (decider's trick, §5.6).
    abstain_fraction: float = 0.1

    # Optimisation.
    epochs: int = 3
    per_device_batch: int = 32
    # Length-sorting window in batches; limits padding while preserving randomness.
    bucket_window: int = 64
    grad_accum: int = 1
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    gradient_checkpointing: bool = True
    brier_weight: float = 0.5
    # Every ordered readout: Score and Noul, both modes (see readout/mode_b.py).
    ordinal_weight: float = 0.25

    # Bookkeeping.
    seed: int = 17
    checkpoint_every: int = 2_000
    log_every: int = 25
    output_dir: str = "checkpoints/lev"

    @property
    def tokens_per_epoch(self) -> int:
        return self.n_examples * self.avg_tokens_per_example

    @property
    def total_tokens(self) -> int:
        return self.tokens_per_epoch * self.epochs

    @property
    def tokens_per_step(self) -> int:
        # The work per step is the padded batch, not `max_seq_len` (~16x larger).
        return self.avg_tokens_per_example * self.per_device_batch * self.grad_accum

    @property
    def examples_per_step(self) -> int:
        return self.per_device_batch * self.grad_accum

    @property
    def steps_per_epoch(self) -> int:
        return max(1, self.n_examples // self.examples_per_step)

    @property
    def total_steps(self) -> int:
        return self.steps_per_epoch * self.epochs

    @property
    def total_flops(self) -> float:
        return FLOPS_PER_PARAM_PER_TOKEN * self.params_b * 1e9 * self.total_tokens

    @property
    def estimated_hours(self) -> float:
        return self.total_flops / H100_EFFECTIVE_FLOPS / 3600

    @property
    def weights_gb(self) -> float:
        return self.params_b * 2  # bf16

    @property
    def optimizer_gb(self) -> float:
        """Weights + grads + AdamW fp32 (m, v, master). ~16 bytes/param full-FT."""
        if self.use_lora:
            return self.weights_gb + 0.3
        return self.params_b * 16

    @property
    def headroom_gb(self) -> float:
        return H100_VRAM_GB - self.optimizer_gb

    def summary(self) -> str:
        adaptation = f"LoRA r{self.lora_rank}" if self.use_lora else "full fine-tune"
        prompt = f"{self.prompt_style} prompts, Mode A to the tokenizer limit"
        return "\n".join(
            [
                f"model            {self.model_id}  ({self.params_b}B, {self.dtype})",
                f"adaptation       {adaptation}",
                f"prompt           {prompt}\n"
                f"data             {self.n_examples:,} examples x "
                f"{self.avg_tokens_per_example} tok x {self.epochs} epochs"
                f"  = {self.total_tokens / 1e9:.2f}B tokens",
                f"steps            {self.total_steps:,} "
                f"({self.examples_per_step} ex/step, ~{self.tokens_per_step:,} tok/step)",
                f"compute          {self.total_flops:.2e} FLOPs",
                f"H100 estimate    {self.estimated_hours:.1f} hours "
                f"({self.estimated_hours / 24:.1f} days)",
                f"memory           {self.optimizer_gb:.1f} GB state, "
                f"{self.headroom_gb:.1f} GB headroom of {H100_VRAM_GB:.0f} GB",
            ]
        )

    def validate(self) -> None:
        if self.optimizer_gb >= H100_VRAM_GB:
            raise ValueError(
                f"{self.optimizer_gb:.0f} GB of optimiser state does not fit one H100. "
                "Enable LoRA or choose a smaller backbone."
            )
        if self.headroom_gb < 20:
            raise ValueError(
                f"only {self.headroom_gb:.1f} GB left for activations. "
                f"Expect OOM at seq_len={self.max_seq_len}."
            )
        if not 0.0 <= self.schema_first_fraction <= 1.0:
            raise ValueError("schema_first_fraction must be in [0, 1]")


# Copy with dataclasses.replace before applying per-run overrides.
PRESETS: dict[str, TrainConfig] = {
    "4b": TrainConfig(),
    # A lower learning rate preserves the instruct backbone's existing abilities (ADR-020).
    "4b-instruct": TrainConfig(
        model_id="Qwen/Qwen3.5-4B",
        learning_rate=5e-5,
        prompt_style="chat",
        output_dir="checkpoints/lev-instruct",
    ),
    "2b": TrainConfig(
        model_id="Qwen/Qwen3.5-2B-Base",
        params_b=2.0,
        hidden_size=2048,
        use_lora=False,
        output_dir="checkpoints/lev-2b",
    ),
    "9b": TrainConfig(
        model_id="Qwen/Qwen3.5-9B-Base",
        params_b=9.0,
        hidden_size=4096,
        per_device_batch=16,
        output_dir="checkpoints/lev-9b",
    ),
    "smoke": TrainConfig(
        model_id="Qwen/Qwen3.5-0.8B-Base",
        params_b=0.8,
        hidden_size=1024,
        n_examples=2_000,
        epochs=1,
        max_seq_len=1_024,
        per_device_batch=2,  # CPU-runnable; `make smoke-local` uses this
        checkpoint_every=0,  # one checkpoint, at the end
        output_dir="checkpoints/lev-smoke",
    ),
}
