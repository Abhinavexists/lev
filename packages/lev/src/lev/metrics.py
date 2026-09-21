"""Evaluation metrics for a decision model.

Accuracy alone hides the two failure modes that matter here, so both get their own
metric: miscalibration (ECE, Brier) and ordinal error (`ordinal_mae`, for Score
under Mode B where nothing structurally enforces level ordering).
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def brier(probs: Sequence[float], truth: int) -> float:
    """Multiclass Brier score: summed squared error over the whole vector."""
    return sum((p - (1.0 if i == truth else 0.0)) ** 2 for i, p in enumerate(probs))


def log_loss(probs: Sequence[float], truth: int) -> float:
    return -math.log(max(probs[truth], 1e-15))


def ordinal_mae(probs: Sequence[float], truth: int) -> float:
    """Expected |predicted level - true level|.

    Distinct from accuracy on purpose: predicting level 0 when the answer is 4 is
    four times worse than predicting level 3, and accuracy scores both as simply
    wrong. decider reports this separately for the same reason.
    """
    return sum(p * abs(i - truth) for i, p in enumerate(probs))


def selective_accuracy(
    records: Sequence[tuple[Sequence[float], int]], threshold: float
) -> tuple[float, float]:
    """Accuracy above a confidence threshold, and the fraction kept.

    The practical test of whether confidence is usable for routing: if accuracy
    does not rise as the threshold rises, gating on confidence buys nothing.
    """
    kept = [(p, t) for p, t in records if max(p) >= threshold]
    if not kept:
        return 0.0, 0.0
    correct = sum(1 for p, t in kept if max(range(len(p)), key=lambda i: p[i]) == t)
    return correct / len(kept), len(kept) / len(records)
