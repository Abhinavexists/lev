"""Per-row scoring rules over a candidate distribution."""

from __future__ import annotations

import math
from collections.abc import Sequence


# Multiclass Brier: squared error summed over the whole vector.
def brier(probs: Sequence[float], truth: int) -> float:
    return sum((p - (1.0 if i == truth else 0.0)) ** 2 for i, p in enumerate(probs))


def log_loss(probs: Sequence[float], truth: int) -> float:
    return -math.log(max(probs[truth], 1e-15))
