"""Accuracy and calibration over typed answer distributions.

ECE uses top probability, since servers define their `confidence` fields
differently. Noul is represented as {True: p, False: 1-p} for both backends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# Log clamp, so a confidently wrong answer scores badly rather than infinitely
# badly (the scikit-learn convention).
_EPS = 1e-15


def to_distribution(answer: Any) -> dict[Any, float]:
    """Flatten any answer type to a label -> probability map."""
    kind = answer.type
    if kind == "noul":
        p = float(answer.noul)
        return {True: p, False: 1.0 - p}
    if kind == "choice":
        return {k: float(v) for k, v in answer.probabilities.items()}
    if kind == "score":
        return {int(k): float(v) for k, v in answer.probabilities.items()}
    raise ValueError(f"Unknown answer type {kind!r}")


def predicted_label(answer: Any) -> Any:
    """The argmax label. Uses the model's own pick where it reports one."""
    kind = answer.type
    if kind == "noul":
        return float(answer.noul) >= 0.5
    if kind == "choice":
        # The model's own pick, not a recomputed argmax: they differ on ties, and
        # `choice` is what a caller acts on.
        return answer.choice
    if kind == "score":
        return max(answer.probabilities.items(), key=lambda kv: kv[1])[0]
    raise ValueError(f"Unknown answer type {kind!r}")


def top_probability(answer: Any) -> float:
    """The probability of the most likely label: the confidence ECE bins on.

    For a Noul that is max(p, 1-p).
    """
    return max(to_distribution(answer).values())


def log_loss(dist: dict[Any, float], truth: Any) -> float:
    return -math.log(max(dist.get(truth, 0.0), _EPS))


def brier(dist: dict[Any, float], truth: Any) -> float:
    """Multiclass Brier score: summed squared error over the whole vector."""
    labels = set(dist) | {truth}
    return sum((dist.get(k, 0.0) - (1.0 if k == truth else 0.0)) ** 2 for k in labels)


@dataclass
class Bin:
    lo: float
    hi: float
    n: int = 0
    correct: int = 0
    conf_sum: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def mean_confidence(self) -> float:
        return self.conf_sum / self.n if self.n else 0.0

    @property
    def gap(self) -> float:
        """Signed: positive means overconfident."""
        return self.mean_confidence - self.accuracy


@dataclass
class Calibration:
    n: int = 0
    n_correct: int = 0
    log_loss_sum: float = 0.0
    brier_sum: float = 0.0
    bins: list[Bin] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n if self.n else 0.0

    @property
    def mean_log_loss(self) -> float:
        return self.log_loss_sum / self.n if self.n else 0.0

    @property
    def mean_brier(self) -> float:
        return self.brier_sum / self.n if self.n else 0.0

    @property
    def ece(self) -> float:
        """Expected Calibration Error: sample-weighted mean absolute bin gap."""
        if not self.n:
            return 0.0
        return sum(b.n * abs(b.gap) for b in self.bins) / self.n


def calibration(
    records: list[tuple[dict[Any, float], Any, Any, float]], n_bins: int = 10
) -> Calibration:
    """Aggregate metrics over (distribution, predicted, truth, confidence) rows."""
    edges = [i / n_bins for i in range(n_bins + 1)]
    cal = Calibration(bins=[Bin(edges[i], edges[i + 1]) for i in range(n_bins)])

    for dist, pred, truth, conf in records:
        is_correct = pred == truth
        cal.n += 1
        cal.n_correct += int(is_correct)
        cal.log_loss_sum += log_loss(dist, truth)
        cal.brier_sum += brier(dist, truth)

        # Last bin is closed on the right so confidence == 1.0 lands somewhere.
        idx = min(int(conf * n_bins), n_bins - 1)
        b = cal.bins[idx]
        b.n += 1
        b.correct += int(is_correct)
        b.conf_sum += conf

    return cal


def selective_accuracy(
    records: list[tuple[dict[Any, float], Any, Any, float]], threshold: float
) -> tuple[float, float]:
    """Accuracy on answers above a confidence threshold, plus the kept fraction.

    Decides whether confidence is usable for routing: if accuracy does not rise
    with the threshold, confidence carries no signal.
    """
    kept = [r for r in records if r[3] >= threshold]
    if not kept:
        return 0.0, 0.0
    acc = sum(1 for _, pred, truth, _ in kept if pred == truth) / len(kept)
    return acc, len(kept) / len(records)
