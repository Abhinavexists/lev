"""Identify which statistic the server's `confidence` field actually is.

TypeSafe documents confidence only as "a statistic computed from the
probability distribution the answer already gives you" -- it never says which
statistic. Since every Choice and Score answer returns both `probabilities` and
`confidence`, the formula is directly identifiable from responses: compute each
candidate statistic over the returned distribution and see which reproduces the
returned confidence.

LitJev, an open reproduction, states it uses normalized Gini concentration and
explicitly disclaims numerical parity with Jev. That makes Gini the leading
hypothesis and gives this module a known-answer test: run it against a LitJev
server first, confirm it identifies Gini, then point it at Jev.

Noul answers carry no confidence field and are skipped.
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


CANDIDATES: dict[str, Callable[[Distribution], float]] = {
    "gini": gini,
    "max_prob": max_prob,
    "1-norm_entropy": one_minus_normalized_entropy,
    "top_two_margin": top_two_margin,
}


@dataclass
class Fit:
    name: str
    n: int
    mean_abs_error: float
    max_abs_error: float

    @property
    def matches(self) -> bool:
        """Within float-rounding of the reported value across every sample.

        The API rounds probabilities before sending them, so an exact match is
        not achievable; 5e-3 is loose enough to survive that rounding and tight
        enough that only one candidate should pass.
        """
        return self.max_abs_error < 5e-3


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
    fits: list[Fit] = []
    for name, fn in CANDIDATES.items():
        errors = [abs(fn(dist) - reported) for dist, reported in samples]
        if not errors:
            continue
        fits.append(
            Fit(
                name=name,
                n=len(errors),
                mean_abs_error=sum(errors) / len(errors),
                max_abs_error=max(errors),
            )
        )
    return sorted(fits, key=lambda f: f.max_abs_error)


def format_fits(fits: list[Fit]) -> str:
    if not fits:
        return "No answers carried both `probabilities` and `confidence`."
    out = [
        "=== confidence formula identification ===",
        f"{'statistic':<16} {'n':>4}  {'mean abs err':>13}  {'max abs err':>12}  match",
    ]
    for f in fits:
        out.append(
            f"{f.name:<16} {f.n:>4}  {f.mean_abs_error:>13.6f}  "
            f"{f.max_abs_error:>12.6f}  {'YES' if f.matches else '-'}"
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
