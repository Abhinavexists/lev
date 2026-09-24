"""Identify which statistic the server's `confidence` field actually is.

TypeSafe documents confidence only as "a statistic computed from the
probability distribution the answer already gives you". Choice and Score answers
return both, so each candidate statistic is computed over `probabilities` and
compared with the reported `confidence`. Against Jev this identified
chance-corrected max probability (`norm_max_prob`), rounded to 2 dp
(FINDINGS.md §2). LitJev states it uses normalized Gini, which makes a LitJev
server a known-answer test.

Answers without both probabilities and confidence (including Jev's Noul) are skipped.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

Distribution = list[float]


def gini(p: Distribution) -> float:
    """Normalized Gini concentration: (K*sum(p^2) - 1) / (K - 1)."""
    k = len(p)
    if k <= 1:
        return 1.0
    return (k * sum(x * x for x in p) - 1.0) / (k - 1)


def max_prob(p: Distribution) -> float:
    return max(p)


def one_minus_normalized_entropy(p: Distribution) -> float:
    k = len(p)
    if k <= 1:
        return 1.0
    entropy = -sum(x * math.log(x) for x in p if x > 0)
    return 1.0 - entropy / math.log(k)


def top_two_margin(p: Distribution) -> float:
    if len(p) < 2:
        return 1.0
    a, b = sorted(p, reverse=True)[:2]
    return a - b


def normalized_max_prob(p: Distribution) -> float:
    """Max probability, chance-corrected: (K*max - 1) / (K - 1).

    The max_prob analogue of `gini`: a raw 0.5 means something different over
    2 candidates than over 20.
    """
    k = len(p)
    if k <= 1:
        return 1.0
    return (k * max(p) - 1.0) / (k - 1)


def top_two_ratio(p: Distribution) -> float:
    """Share of the top two candidates' mass held by the winner."""
    if len(p) < 2:
        return 1.0
    a, b = sorted(p, reverse=True)[:2]
    return a / (a + b) if (a + b) else 1.0


CANDIDATES: dict[str, Callable[[Distribution], float]] = {
    "gini": gini,
    "max_prob": max_prob,
    "norm_max_prob": normalized_max_prob,
    "1-norm_entropy": one_minus_normalized_entropy,
    "top_two_margin": top_two_margin,
    "top_two_ratio": top_two_ratio,
}


def detect_precision(values: list[float], max_decimals: int = 6) -> float:
    """The quantisation step every value is a multiple of, or 0 if none is.

    The API rounds before sending, which sets the floor on any match.
    """
    for decimals in range(1, max_decimals + 1):
        unit = 10.0**-decimals
        if all(abs(v / unit - round(v / unit)) < 1e-6 for v in values):
            return unit
    return 0.0


def rounding_sensitivity(fn: Callable[[Distribution], float], p: Distribution, eps: float) -> float:
    """How far `fn` can move when each probability is off by up to `eps`.

    `max_prob` passes rounding error straight through; `gini` and
    `norm_max_prob` divide by (K-1) and amplify it by K/(K-1). So each formula
    gets its own tolerance.
    """
    base = fn(p)
    raised = fn([min(1.0, x + eps) for x in p])
    lowered = fn([max(0.0, x - eps) for x in p])
    return max(abs(raised - base), abs(lowered - base))


@dataclass
class Fit:
    name: str
    n: int
    mean_abs_error: float
    max_abs_error: float
    tolerance: float = 5e-3

    @property
    def matches(self) -> bool:
        """Within what the API's own rounding could account for.

        The tolerance is the reported value's rounding plus how far that rounding
        can move this statistic. A flat 5e-3 rejects Jev's formula:
        `norm_max_prob` amplifies 2 dp rounding by K/(K-1) to about 0.01.
        """
        return self.max_abs_error <= self.tolerance


def collect(answers: list[Any]) -> list[tuple[Distribution, float]]:
    """Pull (distribution, reported confidence) from answers that have both."""
    samples: list[tuple[Distribution, float]] = []
    for answer in answers:
        reported = getattr(answer, "confidence", None)
        probabilities = getattr(answer, "probabilities", None)
        if reported is None or not probabilities:
            continue  # Noul, or an answer type with no distribution.
        samples.append(([float(v) for v in probabilities.values()], float(reported)))
    return samples


def identify(samples: list[tuple[Distribution, float]]) -> list[Fit]:
    """Rank candidate formulas by how closely they reproduce `confidence`."""
    everything = [v for dist, reported in samples for v in (*dist, reported)]
    eps = detect_precision(everything) / 2

    fits: list[Fit] = []
    for name, fn in CANDIDATES.items():
        errors = [abs(fn(dist) - reported) for dist, reported in samples]
        if not errors:
            continue
        tolerance = max(
            (eps + rounding_sensitivity(fn, dist, eps) for dist, _ in samples), default=0.0
        )
        fits.append(
            Fit(
                name=name,
                n=len(errors),
                mean_abs_error=sum(errors) / len(errors),
                max_abs_error=max(errors),
                tolerance=max(tolerance, 1e-9),
            )
        )
    return sorted(fits, key=lambda f: f.max_abs_error)


def identify_by_size(samples: list[tuple[Distribution, float]]) -> dict[int, list[Fit]]:
    """`identify` per candidate-set size.

    A formula that holds for 4-option Choice but not 3-level Score reads as "no
    match" when pooled; in practice candidate count tracks question type.
    """
    by_size: dict[int, list[tuple[Distribution, float]]] = {}
    for dist, reported in samples:
        by_size.setdefault(len(dist), []).append((dist, reported))
    return {size: identify(group) for size, group in sorted(by_size.items())}


def worst_residuals(
    samples: list[tuple[Distribution, float]], name: str, limit: int = 3
) -> list[tuple[Distribution, float, float]]:
    """The samples a candidate fits worst, as (distribution, reported, predicted)."""
    fn = CANDIDATES[name]
    scored = [(dist, reported, fn(dist)) for dist, reported in samples]
    return sorted(scored, key=lambda row: abs(row[2] - row[1]), reverse=True)[:limit]


def format_fits(fits: list[Fit]) -> str:
    if not fits:
        return "No answers carried both `probabilities` and `confidence`."
    out = [
        "=== confidence formula identification ===",
        f"{'statistic':<16} {'n':>4}  {'mean abs err':>13}  {'max abs err':>12}  "
        f"{'tolerance':>10}  match",
    ]
    for f in fits:
        out.append(
            f"{f.name:<16} {f.n:>4}  {f.mean_abs_error:>13.6f}  "
            f"{f.max_abs_error:>12.6f}  {f.tolerance:>10.6f}  {'YES' if f.matches else '-'}"
        )
    winners = [f.name for f in fits if f.matches]
    out.append("")
    if len(winners) == 1:
        out.append(f"`confidence` is {winners[0]} over `probabilities`.")
    elif winners:
        out.append(
            f"Indistinguishable on this sample: {', '.join(winners)}. Need wider distributions."
        )
    else:
        out.append("No candidate matched -- the formula is none of these.")
    return "\n".join(out)


def format_diagnosis(samples: list[tuple[Distribution, float]]) -> str:
    """Per-size fits and the worst residuals, for when nothing matches overall."""
    out: list[str] = []

    off = [abs(sum(dist) - 1.0) for dist, _ in samples]
    if off:
        out.append(
            f"distributions sum to 1 within {max(off):.4g} "
            f"(mean {sum(off) / len(off):.4g}) -- larger means the API truncates, "
            f"and every candidate is then computed on an incomplete vector"
        )

    for size, fits in identify_by_size(samples).items():
        best = fits[0]
        out.append(
            f"\nK={size}  n={best.n}   best: {best.name} "
            f"(mean {best.mean_abs_error:.6f}, max {best.max_abs_error:.6f})"
            f"{'  MATCH' if best.matches else ''}"
        )
        for f in fits[1:3]:
            out.append(f"           then: {f.name} (max {f.max_abs_error:.6f})")

    overall = identify(samples)
    if overall:
        name = overall[0].name
        out.append(f"\nworst residuals for {name}:")
        for dist, reported, predicted in worst_residuals(samples, name):
            shown = ", ".join(f"{p:.3f}" for p in sorted(dist, reverse=True)[:5])
            out.append(
                f"  reported {reported:.4f}  {name} {predicted:.4f}  "
                f"diff {predicted - reported:+.4f}   dist=[{shown}]"
            )
    return "\n".join(out)
