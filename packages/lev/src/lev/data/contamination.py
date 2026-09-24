"""Reject training sources that resolve to any of the 13 S1Bench subsets."""

from __future__ import annotations

import re
from collections.abc import Iterable

# All 13 S1Bench evaluation subsets, not only the six the public board completed.
BLOCKED_SUBSETS: frozenset[str] = frozenset(
    {
        "vitaminc-dev",
        "massive-en-US",
        "boolq",
        "helpsteer2",
        "aegis2",
        "paws",
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
    "mteb/amazon_massive_scenario": "massive-en-US",
    "setfit/amazon_massive_intent_en-us": "massive-en-US",
    "setfit/amazon_massive_scenario_en-us": "massive-en-US",
    # Not the same rows, but the same 64-intent schema MASSIVE inherited via
    # SLURP (`alarm_query`, `iot_hue_lightchange`, ...). Training on it turns
    # massive-en-US from an unseen-taxonomy test into a seen one.
    "hwu64": "massive-en-US",
    "deeppavlov/hwu64": "massive-en-US",
    "super_glue/boolq": "boolq",
    "aps/super_glue": "boolq",
    "google/boolq": "boolq",
    "nvidia/helpsteer2": "helpsteer2",
    "helpsteer": "helpsteer2",
    "nvidia/aegis-ai-content-safety-dataset-2.0": "aegis2",
    # 1.0 is the same prompt corpus under an earlier annotation pass.
    "nvidia/aegis-ai-content-safety-dataset-1.0": "aegis2",
    "aegis": "aegis2",
    "paws-x": "paws",
    "google-research-datasets/paws": "paws",
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


_BLOCKED_BY_NORMALISED: dict[str, str] = {normalise(s): s for s in BLOCKED_SUBSETS}
_ALIASES_BY_NORMALISED: dict[str, str] = {normalise(a): t for a, t in _ALIASES.items()}
# Bare aliases (`massive`, `aegis`, `squad-v2`) as segment sets, so a re-hosted
# copy under a new name (`SetFit/amazon_massive_intent_en-US`,
# `nvidia/Aegis-AI-Content-Safety-Dataset-1.0`) is caught like a blocked subset
# name inside a longer id. Org-qualified aliases stay exact-match only.
_ALIAS_SEGMENTS: tuple[tuple[frozenset[str], str], ...] = tuple(
    (frozenset(alias.split("-")), target)
    for alias, target in _ALIASES_BY_NORMALISED.items()
    if "/" not in alias
)


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
    # Likewise for a bare alias: every segment of `squad-v2` present, in any order.
    # Conservative on purpose: a false block costs one dataset, a miss the result.
    for alias_segments, subset in _ALIAS_SEGMENTS:
        if alias_segments <= segments:
            return subset
    return None


def check_mixture(dataset_names: Iterable[str]) -> dict[str, str]:
    """Return `{offending name: blocked subset}` for everything that collides."""
    hits: dict[str, str] = {}
    for name in dataset_names:
        if (subset := resolve(name)) is not None:
            hits[name] = subset
    return hits


def assert_eval_only(dataset_name: str) -> str:
    """Return the blocked subset name; reject sources outside the evaluation set."""
    subset = resolve(dataset_name)
    if subset is None:
        raise ContaminationError(
            f"{dataset_name!r} is not an S1Bench evaluation subset. This loader "
            f"exists only to read the thirteen blocked subsets for evaluation; "
            f"training data must go through the normal source registry, which is "
            f"checked by `assert_clean`."
        )
    return subset


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
