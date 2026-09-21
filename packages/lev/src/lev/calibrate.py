"""Temperature scaling — the cheapest large win available, per the benchmark data.

Untuned Qwen3.5 backbones sit at ECE 0.4252 on S1Bench. reflex -- same model family
plus one fitted scalar -- reaches 0.0849. Jev is 0.0764. A single scalar per bucket
is most of that gap, and almost none of the open reproductions bothered to fit one.

Two rules this module enforces structurally rather than by convention:

1. **Fit per bucket, not globally.** Choice, Score and Noul produce differently shaped
   distributions, and Mode A and Mode B produce them by different mechanisms. One
   global temperature under-serves at least one bucket, so the key is (type, mode).
2. **Never fit on test.** `fit` takes an explicit split name and refuses `"test"`.
   The calibration split must be disjoint from both train and test.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

Logits = Sequence[float]


def softmax(logits: Logits, temperature: float = 1.0) -> list[float]:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    scaled = [x / temperature for x in logits]
    top = max(scaled)
    exps = [math.exp(x - top) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def nll(samples: Sequence[tuple[Logits, int]], temperature: float) -> float:
    """Mean negative log-likelihood of the true class. The fitting objective."""
    if not samples:
        return 0.0
    total = 0.0
    for logits, truth in samples:
        p = softmax(logits, temperature)[truth]
        total -= math.log(max(p, 1e-15))
    return total / len(samples)


def fit_temperature(
    samples: Sequence[tuple[Logits, int]],
    lo: float = 0.05,
    hi: float = 10.0,
    tol: float = 1e-4,
) -> float:
    """Minimise NLL over temperature by ternary search.

    NLL as a function of temperature is unimodal for a fixed set of logits, so
    ternary search converges without gradients and without a dependency on torch.
    """
    if not samples:
        return 1.0
    while hi - lo > tol:
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if nll(samples, m1) < nll(samples, m2):
            hi = m2
        else:
            lo = m1
    return (lo + hi) / 2


def expected_calibration_error(
    probs: Sequence[Sequence[float]], truths: Sequence[int], n_bins: int = 10
) -> float:
    """Sample-weighted mean gap between top-probability and accuracy."""
    if not probs:
        return 0.0
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for p, truth in zip(probs, truths, strict=True):
        conf = max(p)
        correct = max(range(len(p)), key=lambda i: p[i]) == truth
        bins[min(int(conf * n_bins), n_bins - 1)].append((conf, correct))
    total = sum(len(b) for b in bins)
    return (
        sum(
            len(b) * abs(sum(c for c, _ in b) / len(b) - sum(ok for _, ok in b) / len(b))
            for b in bins
            if b
        )
        / total
    )


@dataclass
class CalibrationProfile:
    """Fitted temperatures keyed by `"{question_type}:{mode}"`."""

    temperatures: dict[str, float] = field(default_factory=dict)
    fitted_on: str = ""
    n_samples: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def key(question_type: str, mode: str) -> str:
        return f"{question_type}:{mode}"

    def temperature(self, question_type: str, mode: str) -> float:
        # 1.0 is the identity, so an unfitted bucket degrades to raw softmax
        # rather than to a wrong temperature borrowed from another bucket.
        return self.temperatures.get(self.key(question_type, mode), 1.0)

    def apply(self, logits: Logits, question_type: str, mode: str) -> list[float]:
        return softmax(logits, self.temperature(question_type, mode))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "temperatures": self.temperatures,
                    "fitted_on": self.fitted_on,
                    "n_samples": self.n_samples,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> CalibrationProfile:
        d = json.loads(Path(path).read_text())
        return cls(
            temperatures=d["temperatures"],
            fitted_on=d.get("fitted_on", ""),
            n_samples=d.get("n_samples", {}),
        )


def fit(
    buckets: dict[str, Sequence[tuple[Logits, int]]],
    split_name: str,
    min_samples: int = 50,
) -> CalibrationProfile:
    """Fit one temperature per bucket.

    Refuses `split_name="test"`: fitting on test labels produces a profile that
    looks excellent and means nothing, and it is the single easiest way to invalidate
    the only number this project competes on.
    """
    if split_name.lower() in {"test", "eval", "holdout"}:
        raise ValueError(
            f"refusing to fit calibration on split {split_name!r}. "
            "Use a dedicated calibration split, disjoint from train and test."
        )

    profile = CalibrationProfile(fitted_on=split_name)
    for bucket, samples in buckets.items():
        if len(samples) < min_samples:
            # Leave it unfitted (identity) rather than fit a scalar on noise.
            continue
        profile.temperatures[bucket] = fit_temperature(samples)
        profile.n_samples[bucket] = len(samples)
    return profile
