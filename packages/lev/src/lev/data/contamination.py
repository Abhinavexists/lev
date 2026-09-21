"""The contamination guard. This is a build requirement, not hygiene.

Our one differentiating claim is calibration measured on S1Bench's six evaluation
subsets. Three models on that leaderboard self-declare training contamination, and
their numbers cannot be trusted because of it. If any of those six subsets reaches
our training mixture, the number we are competing on becomes worthless -- and the
failure is silent, because a contaminated model looks *better*.

So the guard raises rather than warns, and it runs before training, not after.

Candidate data sources and their known risk:
  decider's registry (~95 public datasets)  -- CONTAINS several of the six
  Nimble (2,676 train / 324 test)           -- synthetic contrastive pairs
  NanoJev observed-event data               -- good for the calibration objective
  teacher-generated states                  -- safe if the teacher is not shown the six
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# The six S1Bench subsets we are evaluated on. Never train on these.
BLOCKED_SUBSETS: frozenset[str] = frozenset(
    {
        "vitaminc-dev",
        "massive-en-US",
        "boolq",
        "helpsteer2",
        "aegis2",
        "paws",
    }
)

# Aliases and parent datasets that resolve to a blocked subset. A mixture that lists
# `google-research-datasets/paws` is contaminated even though the string differs.
_ALIASES: dict[str, str] = {
    "vitaminc": "vitaminc-dev",
    "tals/vitaminc": "vitaminc-dev",
    "massive": "massive-en-US",
    "amazonscience/massive": "massive-en-US",
    "mteb/amazon_massive_intent": "massive-en-US",
    "super_glue/boolq": "boolq",
    "aps/super_glue": "boolq",
    "google/boolq": "boolq",
    "nvidia/helpsteer2": "helpsteer2",
    "helpsteer": "helpsteer2",
    "nvidia/aegis-ai-content-safety-dataset-2.0": "aegis2",
    "aegis": "aegis2",
    "paws-x": "paws",
    "google-research-datasets/paws": "paws",
}


class ContaminationError(RuntimeError):
    """Raised when a training mixture touches an evaluation subset."""


def normalise(name: str) -> str:
    """Lowercase, strip a HF org prefix and split suffix, collapse separators."""
    n = name.strip().lower()
    n = re.sub(r"[\s_]+", "-", n)
    n = re.sub(r":(train|validation|dev|test)$", "", n)
    return n


def resolve(name: str) -> str | None:
    """Map a dataset name to the blocked subset it belongs to, if any."""
    n = normalise(name)
    blocked = {normalise(b): b for b in BLOCKED_SUBSETS}

    if n in blocked:
        return blocked[n]
    for alias, target in _ALIASES.items():
        if n == normalise(alias):
            return target
    # Bare name after an org prefix: `tals/vitaminc` -> `vitaminc`.
    if "/" in n:
        return resolve(n.split("/", 1)[1])
    # A blocked name appearing as a whole path segment, e.g. `mix/boolq/v2`.
    for norm_b, original in blocked.items():
        if norm_b in n.split("-") or norm_b == n:
            return original
    return None


def check_mixture(dataset_names: Iterable[str]) -> dict[str, str]:
    """Return `{offending name: blocked subset}` for everything that collides."""
    hits: dict[str, str] = {}
    for name in dataset_names:
        if (subset := resolve(name)) is not None:
            hits[name] = subset
    return hits


def assert_clean(dataset_names: Iterable[str]) -> None:
    """Raise if any dataset resolves to a blocked evaluation subset."""
    names = list(dataset_names)
    if hits := check_mixture(names):
        listed = "\n".join(f"  {src!r} -> blocked subset {dst!r}" for src, dst in hits.items())
        raise ContaminationError(
            f"{len(hits)} dataset(s) in the mixture collide with S1Bench evaluation "
            f"subsets:\n{listed}\n\n"
            "Remove them. Training on an evaluation subset invalidates the calibration "
            "result this project exists to produce. See docs/ARCHITECTURE.md §5.7."
        )
