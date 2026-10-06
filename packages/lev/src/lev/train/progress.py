"""Progress reporting for a training run."""

from __future__ import annotations

import time


class ProgressLog:
    """Flush periodic per-mode losses and throughput measured from real tokens."""

    def __init__(self, total_steps: int, log_every: int):
        self.total = total_steps
        self.every = max(1, log_every)
        self.start = time.monotonic()
        self.window: dict[str, list[float]] = {}
        self.window_tokens = 0
        self.mark = self.start
        self.mark_step = 0

    def record(self, step: int, loss: float, mode: str, lr: float, tokens: int = 0) -> None:
        self.window.setdefault(mode, []).append(loss)
        self.window_tokens += tokens
        done = step + 1
        if done % self.every and done != self.total:
            return

        now = time.monotonic()
        elapsed = max(now - self.start, 1e-9)
        # Use windowed rates to exclude startup delay; guard against zero elapsed time.
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
