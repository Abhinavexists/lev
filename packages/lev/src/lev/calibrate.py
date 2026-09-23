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
    """Sample-weighted mean gap between top-probability and accuracy.

    A perfectly calibrated model that says "80% confident" is right 80% of the
    time, so within each confidence bin the mean confidence should equal the
    accuracy. ECE is the average of those gaps, weighted by bin population.
    """
    if not probs:
        return 0.0

    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for distribution, truth in zip(probs, truths, strict=True):
        confidence = max(distribution)
        predicted = max(range(len(distribution)), key=distribution.__getitem__)
        # The top bin is closed: a confidence of exactly 1.0 would otherwise
        # index one past the end.
        bin_index = min(int(confidence * n_bins), n_bins - 1)
        bins[bin_index].append((confidence, predicted == truth))

    total_weighted_gap = 0.0
    for members in bins:
        if not members:
            continue
        mean_confidence = sum(c for c, _ in members) / len(members)
        accuracy = sum(hit for _, hit in members) / len(members)
        total_weighted_gap += len(members) * abs(mean_confidence - accuracy)
    return total_weighted_gap / len(probs)


# Choice temperatures are fitted per option-count band. One `choice:A` scalar
# fitted on 3-14 options left a 60-option question under-confident (ECE 0.20
# on massive-en-US at accuracy 0.95 above p=0.5): the softmax over many more
# candidates spreads mass differently, and one temperature cannot serve both.
CHOICE_BANDS = ((8, "small"), (26, "mid"))


def option_band(n_options: int | None) -> str | None:
    if n_options is None:
        return None
    for upper, name in CHOICE_BANDS:
        if n_options <= upper:
            return name
    return "large"


@dataclass
class CalibrationProfile:
    """Fitted temperatures keyed by `"{question_type}:{mode}"`, and for Choice
    by `"choice:{mode}:{band}"` as well."""

    temperatures: dict[str, float] = field(default_factory=dict)
    fitted_on: str = ""
    n_samples: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def key(question_type: str, mode: str, n_options: int | None = None) -> str:
        band = option_band(n_options) if question_type == "choice" else None
        return f"{question_type}:{mode}" + (f":{band}" if band else "")

    def temperature(self, question_type: str, mode: str, n_options: int | None = None) -> float:
        # Banded first, then the unbanded bucket a profile fitted before bands
        # existed carries; 1.0 -- the identity -- when neither was fitted, so an
        # unknown bucket degrades to raw softmax rather than to a borrowed scalar.
        banded = self.key(question_type, mode, n_options)
        plain = self.key(question_type, mode)
        return self.temperatures.get(banded, self.temperatures.get(plain, 1.0))

    def apply(
        self, logits: Logits, question_type: str, mode: str, n_options: int | None = None
    ) -> list[float]:
        return softmax(logits, self.temperature(question_type, mode, n_options))

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
        payload = json.loads(Path(path).read_text())
        return cls(
            temperatures=payload["temperatures"],
            fitted_on=payload.get("fitted_on", ""),
            n_samples=payload.get("n_samples", {}),
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
