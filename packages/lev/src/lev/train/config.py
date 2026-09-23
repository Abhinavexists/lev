"""Training configuration, with the H100 budget encoded as arithmetic.

Every default here traces to a measurement in docs/ARCHITECTURE.md §5. The
`estimate` methods exist so that changing a knob shows you the new cost in hours
before you rent the GPU, rather than after.
"""

from __future__ import annotations

from dataclasses import dataclass

# Effective bf16 throughput on one H100 with gradient checkpointing. Peak is
# ~990 TFLOP/s; 400 is a realistic sustained figure and the number §5.8 quotes.
H100_EFFECTIVE_FLOPS = 4.0e14

# One H100 80GB. The memory estimates below are all against this.
H100_VRAM_GB = 80.0

# Forward+backward is ~6*N FLOPs/token; checkpointing recomputes activations for
# roughly a third more. True under LoRA too: the backward pass still traverses
# the frozen weights to reach the adapters.
FLOPS_PER_PARAM_PER_TOKEN = 8.0


@dataclass
class TrainConfig:
    # Qwen3.5-4B-Base: 32 layers = 24 linear + 8 full attention, 262k context,
    # natively multimodal. The hybrid split is why a persistent prefix cache is
    # affordable without pretraining one. See §5.1.
    model_id: str = "Qwen/Qwen3.5-4B-Base"
    params_b: float = 4.0
    hidden_size: int = 2560
    dtype: str = "bfloat16"

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
    # None: a question routes to Mode A whenever the tokenizer expresses its
    # option codes in single tokens, exactly as the server routes (ADR-025), so
    # Mode A trains on every set size it will serve. Mode B's data comes from
    # the large-taxonomy sources' full option sets, which stay above the limit
    # (`sources.augmentation_map`). ADR-026.
    max_label_options: int | None = None
    # `prompt.Style`: "plain" (Context/Question/Answer) or "chat" (ChatML with a
    # system prompt, the instruct backbone's native format). A model property:
    # serving must use the same style, so the release manifest records it.
    prompt_style: str = "plain"

    # Data.
    n_examples: int = 200_000
    # Measured, not assumed: a 121-token mean over 1,500 rendered prompts from
    # the real mixture under the Qwen3.5 tokenizer. Per source the mean runs 38
    # (clinc_oos) to 305 (imdb); p95 is 390 and the longest seen is 1,104. A
    # guessed value here put the 4B budget out by 10x (ADR-016), so re-measure
    # with `lev plan --data <dir>` after changing the mixture.
    avg_tokens_per_example: int = 128
    # A truncation cap, not the training length. The collator pads to the
    # longest row in the batch, so a generous cap costs nothing except on the
    # rare long row it saves.
    max_seq_len: int = 2_048
    # Trained 50/50 so both cache layouts work at inference. §5.5.
    schema_first_fraction: float = 0.5
    # Examples whose answer is not determinable from the state, teaching the model
    # to spread mass instead of guessing confidently. decider's trick. §5.6.
    abstain_fraction: float = 0.1

    # Optimisation.
    epochs: int = 3
    # Sized for the data, not for a 4k-token guess. At a ~128-token mean, a
    # batch of 8 makes 75,000 optimiser steps for 600k examples and the run
    # becomes step-overhead bound long before it becomes FLOP bound. 32 keeps
    # the H100 fed and still fits comfortably under the memory headroom below.
    per_device_batch: int = 32
    # Batches are formed from length-sorted windows of this many batches. A
    # batch pads to its longest row, so mixing a 1,300-token review with 31
    # short tickets spends 4.43x of the forward pass on padding. Sorting within
    # a window brings that to 1.43x while keeping the order random. ADR-017.
    bucket_window: int = 64
    grad_accum: int = 1
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    gradient_checkpointing: bool = True
    # Proper scoring rule: cross-entropy plus a Brier term. Calibration is the
    # objective, not an afterthought.
    brier_weight: float = 0.5
    # Applies to every ordered readout -- Score and Noul, under both modes.
    # See readout/mode_b.py for why ordinality stops being free under Mode B.
    ordinal_weight: float = 0.25

    # Bookkeeping.
    seed: int = 17
    # A long run on a preemptible H100 will be interrupted. Checkpointing
    # every ~2000 steps costs a few seconds and bounds the loss to that much.
    checkpoint_every: int = 2_000
    # How often to print progress. A silent run is indistinguishable from a
    # hung one, and on Modal the only thing you can see is stdout.
    log_every: int = 25
    output_dir: str = "checkpoints/lev-4b"

    @property
    def tokens_per_epoch(self) -> int:
        return self.n_examples * self.avg_tokens_per_example

    @property
    def total_tokens(self) -> int:
        return self.tokens_per_epoch * self.epochs

    @property
    def tokens_per_step(self) -> int:
        # The *work* per step, which is the padded batch, not `max_seq_len`.
        # Billing the cap would inflate every downstream estimate by the ratio
        # between the cap and the data -- here about 16x.
        return self.avg_tokens_per_example * self.per_device_batch * self.grad_accum

    @property
    def examples_per_step(self) -> int:
        return self.per_device_batch * self.grad_accum

    @property
    def steps_per_epoch(self) -> int:
        # Counted in examples. An epoch is one pass over the data, and how many
        # optimiser steps that takes depends on the batch, not on token budgets.
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


# Named presets matching the backbone comparison in §5.1.
PRESETS: dict[str, TrainConfig] = {
    "4b": TrainConfig(),
    # The instruct checkpoint of the same backbone. Frozen, it scores 0.719 on
    # S1Bench (reflex-4b); the Base fine-tune scored 0.489. Starting from a
    # model that already follows instructions makes zero-shot judgement the
    # floor rather than zero, and a gentler learning rate protects it. ADR-020.
    "4b-instruct": TrainConfig(
        model_id="Qwen/Qwen3.5-4B",
        learning_rate=5e-5,
        # The format the instruct backbone was trained on. Frozen, it scores
        # 0.710 on S1Bench in this style against 0.653 in `plain` (ADR-027).
        prompt_style="chat",
        output_dir="checkpoints/lev-4b-instruct",
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
        checkpoint_every=0,  # one checkpoint at the end is the whole point
        output_dir="checkpoints/lev-smoke",
    ),
}
