"""Progress reporting for a training run."""

from __future__ import annotations

import time


class ProgressLog:
    """Periodic one-line progress, flushed: on Modal stdout is the only view of a
    run, and Python block-buffers it when it is not a tty.

    Losses are per mode (Mode B starts near `ln(K)`, a different scale), and
    loss, rate, throughput and ETA are measured over the window since the last
    report. Throughput counts real tokens from the attention mask, not
    `config.avg_tokens_per_example`, so it can contradict a wrong plan (ADR-016).
    """

    def __init__(self, total_steps: int, log_every: int):
        self.total = total_steps
        self.every = max(1, log_every)
        self.start = time.monotonic()
        self.window: dict[str, list[float]] = {}
        self.tokens = 0
        self.window_tokens = 0
        self.mark = self.start
        self.mark_step = 0

    def record(self, step: int, loss: float, mode: str, lr: float, tokens: int = 0) -> None:
        self.window.setdefault(mode, []).append(loss)
        self.tokens += tokens
        self.window_tokens += tokens
        if (step + 1) % self.every and step + 1 != self.total:
            return

        done = step + 1
        now = time.monotonic()
        # A coarse clock can report zero on a fast window; never divide by it.
        elapsed = max(now - self.start, 1e-9)
        # Over the window, not since start: startup (weight loading, a Triton JIT
        # compile of minutes) made a cumulative rate read 0.33 it/s against 2.50
        # steady-state, with the ETA wrong by the same factor (ADR-017).
        span = max(now - self.mark, 1e-9)
        rate = (done - self.mark_step) / span
        remaining = (self.total - done) / rate if rate else 0.0
        losses = "  ".join(f"{m}={sum(v) / len(v):.4f}" for m, v in sorted(self.window.items()))
        print(
            f"step {done:>6}/{self.total}  {100 * done / self.total:5.1f}%  "
            f"{losses}  lr={lr:.2e}  {rate:.2f} it/s  "
            f"{self.window_tokens / span:,.0f} tok/s  "
            f"elapsed {hms(elapsed)}  eta {hms(remaining)}{_gpu_mem()}",
            flush=True,
        )
        self.window.clear()
        self.window_tokens = 0
        self.mark = now
        self.mark_step = done


def hms(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _gpu_mem() -> str:
    import torch

    if not torch.cuda.is_available():
        return ""
    return f"  mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G"
