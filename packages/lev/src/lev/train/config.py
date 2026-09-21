"""Training configuration, with the H100 budget encoded as arithmetic.

Every default here traces to a measurement in docs/ARCHITECTURE.md §5. The
`estimate` methods exist so that changing a knob shows you the new cost in hours
before you rent the GPU, rather than after.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Effective bf16 throughput on one H100 with gradient checkpointing. Peak is
# ~990 TFLOP/s; 400 is a realistic sustained figure and the number §5.8 quotes.
H100_EFFECTIVE_FLOPS = 4.0e14

# Forward+backward is ~6*N FLOPs/token; checkpointing recomputes activations for
# roughly a third more. True under LoRA too: the backward pass still traverses
# the frozen weights to reach the adapters.
FLOPS_PER_PARAM_PER_TOKEN = 8.0


@dataclass
class TrainConfig:
    # --- backbone -----------------------------------------------------------
    # Qwen3.5-4B-Base: 32 layers = 24 linear + 8 full attention, 262k context,
    # natively multimodal. The hybrid split is why a persistent prefix cache is
    # affordable without pretraining one. See §5.1.
    model_id: str = "Qwen/Qwen3.5-4B-Base"
    params_b: float = 4.0
    hidden_size: int = 2560
    dtype: str = "bfloat16"

    # --- adaptation ---------------------------------------------------------
    # LoRA, not full fine-tune: 4B full-FT needs ~64GB of optimiser state and
    # leaves ~16GB for activations, which is too tight at long context. §5.2.
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
    # Mode B's head is new parameters and trains at full precision regardless of
    # LoRA freezing the backbone. This was the non-obvious part of choosing LoRA.
    train_mode_b_head: bool = True
    mode_b_proj_dim: int = 512

    # --- data ---------------------------------------------------------------
    n_examples: int = 200_000
    avg_tokens_per_example: int = 1_200
    max_seq_len: int = 4_096
    # Trained 50/50 so both cache layouts work at inference. §5.5.
    schema_first_fraction: float = 0.5
    # Examples whose answer is not determinable from the state, teaching the model
    # to spread mass instead of guessing confidently. decider's trick. §5.6.
    abstain_fraction: float = 0.1

    # --- optimisation -------------------------------------------------------
    epochs: int = 3
    per_device_batch: int = 8
    grad_accum: int = 1
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    gradient_checkpointing: bool = True
    # Proper scoring rule: cross-entropy plus a Brier term. Calibration is the
    # objective, not an afterthought.
    brier_weight: float = 0.5
    ordinal_weight: float = 0.25  # Mode B Score only; see readout/mode_b.py

    # --- bookkeeping --------------------------------------------------------
    seed: int = 17
    output_dir: str = "checkpoints/lev-4b"
    blocked_subsets: tuple[str, ...] = field(default_factory=tuple)

    # --- derived ------------------------------------------------------------
    @property
    def tokens_per_epoch(self) -> int:
        return self.n_examples * self.avg_tokens_per_example

    @property
    def total_tokens(self) -> int:
        return self.tokens_per_epoch * self.epochs

    @property
    def tokens_per_step(self) -> int:
        return self.max_seq_len * self.per_device_batch * self.grad_accum

    @property
    def steps_per_epoch(self) -> int:
        return max(1, self.tokens_per_epoch // self.tokens_per_step)

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
        return 80.0 - self.optimizer_gb

    def summary(self) -> str:
        adaptation = f"LoRA r{self.lora_rank}" if self.use_lora else "full fine-tune"
        return "\n".join(
            [
                f"model            {self.model_id}  ({self.params_b}B, {self.dtype})",
                f"adaptation       {adaptation}",
                f"data             {self.n_examples:,} examples x "
                f"{self.avg_tokens_per_example} tok x {self.epochs} epochs"
                f"  = {self.total_tokens / 1e9:.2f}B tokens",
                f"steps            {self.total_steps:,} ({self.tokens_per_step:,} tok/step)",
                f"compute          {self.total_flops:.2e} FLOPs",
                f"H100 estimate    {self.estimated_hours:.1f} hours "
                f"({self.estimated_hours / 24:.1f} days)",
                f"memory           {self.optimizer_gb:.1f} GB state, "
                f"{self.headroom_gb:.1f} GB headroom of 80 GB",
            ]
        )

    def validate(self) -> None:
        if self.optimizer_gb >= 80:
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


# Named presets matching the backbone comparison in §5.1.
PRESETS: dict[str, TrainConfig] = {
    "4b": TrainConfig(),
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
        per_device_batch=4,
        output_dir="checkpoints/lev-9b",
    ),
    "smoke": TrainConfig(
        model_id="Qwen/Qwen3.5-0.8B-Base",
        params_b=0.8,
        hidden_size=1024,
        n_examples=2_000,
        epochs=1,
        max_seq_len=1_024,
        output_dir="checkpoints/lev-smoke",
    ),
}
