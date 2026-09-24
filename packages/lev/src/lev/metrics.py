"""Per-row scoring rules over a candidate distribution."""

from __future__ import annotations

import math
from collections.abc import Sequence


def brier(probs: Sequence[float], truth: int) -> float:
    """Multiclass Brier score: summed squared error over the whole vector."""
    return sum((p - (1.0 if i == truth else 0.0)) ** 2 for i, p in enumerate(probs))


def log_loss(probs: Sequence[float], truth: int) -> float:
    return -math.log(max(probs[truth], 1e-15))
