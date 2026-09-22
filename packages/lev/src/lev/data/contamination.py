"""The contamination guard. This is a build requirement, not hygiene.

Our one differentiating claim is calibration measured on S1Bench's evaluation
subsets. Three models on that leaderboard self-declare training contamination, and
their numbers cannot be trusted because of it. If any blocked subset reaches our
training mixture, the number we are competing on becomes worthless -- and the
failure is silent, because a contaminated model looks *better*.

So the guard raises rather than warns, and it runs before training, not after.

Candidate data sources and their known risk:
  decider's registry (~95 public datasets)  -- CONTAINS several of the thirteen
  Nimble (2,676 train / 324 test)           -- synthetic contrastive pairs
  NanoJev observed-event data               -- good for the calibration objective
  teacher-generated states                  -- safe if the teacher is not shown the thirteen
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# Every S1Bench evaluation subset -- all 13 in `published_jev`, not just the 6
# that executed in the `s1-fast` run. The other 7 were intended and unrun; they
# are still evaluation data, and training on them would contaminate any future
# full-suite comparison. Blocking only what happened to run is how you get a
# result that looks good and means nothing.
BLOCKED_SUBSETS: frozenset[str] = frozenset(
    {
        # ran in s1-fast
        "vitaminc-dev",
        "massive-en-US",
        "boolq",
        "helpsteer2",
        "aegis2",
        "paws",
        # intended, did not run
        "massive-de-DE",
        "squad2",
        "multinli",
        "civil_comments",
        "summeval-relevance",
        "summeval-consistency",
        "pubmedqa",
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
    # intended-but-unrun subsets
    "massive-de": "massive-de-DE",
    "squad-v2": "squad2",
    "squad_v2": "squad2",
    "rajpurkar/squad_v2": "squad2",
    "multi-nli": "multinli",
    "nyu-mll/multi_nli": "multinli",
    "nyu-mll/glue": "multinli",
    "google/civil_comments": "civil_comments",
    "civil-comments": "civil_comments",
    "pubmed-qa": "pubmedqa",
    "qiaojin/pubmedqa": "pubmedqa",
    "bigbio/pubmed_qa": "pubmedqa",
    "summeval": "summeval-relevance",
    "mteb/summeval": "summeval-relevance",
}


class ContaminationError(RuntimeError):
    """Raised when a training mixture touches an evaluation subset."""


def normalise(name: str) -> str:
    """Lowercase, strip a split suffix, collapse separators to hyphens."""
    cleaned = name.strip().lower()
    cleaned = re.sub(r"[\s_]+", "-", cleaned)
    return re.sub(r":(train|validation|dev|test)$", "", cleaned)


# Derived from the constants above, so build it once rather than per lookup.
_BLOCKED_BY_NORMALISED: dict[str, str] = {normalise(s): s for s in BLOCKED_SUBSETS}
_ALIASES_BY_NORMALISED: dict[str, str] = {normalise(a): t for a, t in _ALIASES.items()}


def resolve(name: str) -> str | None:
    """Map a dataset name to the blocked subset it belongs to, if any."""
    normalised = normalise(name)

    if normalised in _BLOCKED_BY_NORMALISED:
        return _BLOCKED_BY_NORMALISED[normalised]
    if normalised in _ALIASES_BY_NORMALISED:
        return _ALIASES_BY_NORMALISED[normalised]
    # Bare name after an org prefix: `tals/vitaminc` -> `vitaminc`.
    if "/" in normalised:
        return resolve(normalised.split("/", 1)[1])
    # A blocked name appearing as one hyphen-separated segment, e.g. `mix-boolq-v2`.
    segments = set(normalised.split("-"))
    for normalised_subset, subset in _BLOCKED_BY_NORMALISED.items():
        if normalised_subset in segments:
            return subset
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
def assert_eval_only(dataset_name: str) -> str:
    """The mirror of `assert_clean`: raise unless `dataset_name` *is* a blocked
    subset, and return the subset it resolves to.

    The evaluation harness has to load the thirteen, so it needs a door the
    training path does not have. Making that door open only onto blocked subsets
    is what stops it becoming a general-purpose loader -- one that would quietly
    grow a second route into the mixture and undo the guard it sits beside.
    """
    subset = resolve(dataset_name)
    if subset is None:
        raise ContaminationError(
            f"{dataset_name!r} is not an S1Bench evaluation subset. This loader "
            f"exists only to read the thirteen blocked subsets for evaluation; "
            f"training data must go through the normal source registry, which is "
            f"checked by `assert_clean`."
        )
    return subset


