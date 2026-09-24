"""Temperature scaling by question type, readout mode and Choice option count.

Fits reject test, eval and holdout splits. Transfer selection compares row-
and family-weighted fits using leave-one-family-out ECE (ADR-028).
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


def nll(
    samples: Sequence[tuple[Logits, int]],
    temperature: float,
    weights: Sequence[float] | None = None,
) -> float:
    """Mean (optionally weighted) negative log-likelihood of the true class."""
    if not samples:
        return 0.0
    total = 0.0
    weight_sum = 0.0
    for i, (logits, truth) in enumerate(samples):
        w = 1.0 if weights is None else weights[i]
        p = softmax(logits, temperature)[truth]
        total -= w * math.log(max(p, 1e-15))
        weight_sum += w
    return total / weight_sum


def fit_temperature(
    samples: Sequence[tuple[Logits, int]],
    lo: float = 0.05,
    hi: float = 10.0,
    tol: float = 1e-4,
    weights: Sequence[float] | None = None,
) -> float:
    """Minimise NLL over temperature by ternary search.

    NLL is unimodal in temperature for fixed logits, so this needs no gradients
    and no torch. Returns the identity, 1.0, when no temperature beats it (a flat
    objective, e.g. all-equal logits).
    """
    if not samples:
        return 1.0
    while hi - lo > tol:
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if nll(samples, m1, weights) < nll(samples, m2, weights):
            hi = m2
        else:
            lo = m1
    fitted = (lo + hi) / 2
    if nll(samples, fitted, weights) >= nll(samples, 1.0, weights) - 1e-12:
        return 1.0
    return fitted


MIN_FAMILIES_FOR_TRANSFER = 3


def family_weights(families: Sequence[str]) -> list[float]:
    """Each family's rows share a total weight of 1."""
    counts: dict[str, int] = {}
    for f in families:
        counts[f] = counts.get(f, 0) + 1
    return [1.0 / counts[f] for f in families]


def _fit(samples, families, method: str) -> float:
    return fit_temperature(
        samples, weights=family_weights(families) if method == "family" else None
    )


def leave_one_family_out_ece(
    samples: Sequence[tuple[Logits, int]], families: Sequence[str], method: str
) -> float:
    """Mean over families of the ECE on that family, at the temperature `method`
    fits on every other family. Families weigh equally, as a new task would."""
    eces = []
    for held_out in sorted(set(families)):
        train = [(s, f) for s, f in zip(samples, families, strict=True) if f != held_out]
        test = [s for s, f in zip(samples, families, strict=True) if f == held_out]
        t = _fit([s for s, _ in train], [f for _, f in train], method)
        probs = [softmax(logits, t) for logits, _ in test]
        eces.append(expected_calibration_error(probs, [y for _, y in test]))
    return sum(eces) / len(eces)


def fit_for_transfer(
    buckets: dict[str, Sequence[tuple[Logits, int, str]]],
    split_name: str,
    min_samples: int = 50,
) -> tuple[CalibrationProfile, dict[str, dict]]:
    """Per bucket, fit both ways and keep the one that transfers better.

    Buckets drawn from fewer than `MIN_FAMILIES_FOR_TRANSFER` families have no
    meaningful leave-one-out, and keep the row fit. Returns the profile and a
    per-bucket report of both temperatures, both transfer ECEs and the choice.
    """
    if split_name.lower() in {"test", "eval", "holdout"}:
        raise ValueError(f"refusing to fit calibration on split {split_name!r}")
    profile = CalibrationProfile(fitted_on=f"{split_name} (transfer-selected)")
    report: dict[str, dict] = {}
    for bucket, rows in buckets.items():
        if len(rows) < min_samples:
            continue
        samples = [(logits, y) for logits, y, _ in rows]
        families = [f for _, _, f in rows]
        entry: dict = {"n": len(rows), "families": len(set(families))}
        entry["t_rows"] = _fit(samples, families, "rows")
        if entry["families"] >= MIN_FAMILIES_FOR_TRANSFER:
            entry["t_family"] = _fit(samples, families, "family")
            entry["lofo_ece_rows"] = leave_one_family_out_ece(samples, families, "rows")
            entry["lofo_ece_family"] = leave_one_family_out_ece(samples, families, "family")
            entry["chosen"] = (
                "family" if entry["lofo_ece_family"] < entry["lofo_ece_rows"] else "rows"
            )
        else:
            entry["chosen"] = "rows"
        profile.temperatures[bucket] = entry[f"t_{entry['chosen']}"]
        profile.n_samples[bucket] = len(rows)
        report[bucket] = entry
    return profile, report


def expected_calibration_error(
    probs: Sequence[Sequence[float]], truths: Sequence[int], n_bins: int = 10
) -> float:
    """Sample-weighted mean absolute gap between top probability and accuracy."""
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


# Softmax spreads mass differently as option count grows, so Choice fits use bands.
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
        # Banded, then unbanded (profiles fitted before bands), then 1.0: an
        # unfitted bucket gets raw softmax rather than a borrowed scalar.
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
    """Fit one temperature per bucket; reject test, eval and holdout splits."""
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
