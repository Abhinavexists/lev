"""Deterministic train / calibration / test splits.

Three splits, not two, because the temperature is fitted on held-out data and a
temperature fitted on the test set is not a measurement of anything. `calibrate.fit`
already refuses a split named test/eval/holdout; this module is the other half of
that guarantee -- it is what produces a genuinely disjoint `calibration` split for
it to accept.

Assignment is a hash of a stable per-row key, so it depends on the row and nothing
else: no RNG state, no iteration order, no dataset version. Re-running on a machine
that downloaded the corpus in a different order puts every row in the same split it
was in before, which is the property that makes a resumed or re-run experiment
comparable to the original.

The subtle failure this module exists to prevent: a purely random split over
banking77's 77 intents can leave an intent that appears in test and never in train.
Accuracy on it is then a measurement of nothing. `check_coverage` raises on that.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

from ..router import candidate_count
from .mixture import Example


class Split(StrEnum):
    TRAIN = "train"
    CALIBRATION = "calibration"
    TEST = "test"


# Fitting a temperature needs far fewer rows than fitting a model, and the test
# split only has to be big enough to separate two runs. Everything else trains.
DEFAULT_FRACTIONS: dict[Split, float] = {
    Split.TRAIN: 0.80,
    Split.CALIBRATION: 0.10,
    Split.TEST: 0.10,
}


def row_key(source: str, index: int, text: str) -> str:
    """A stable identity for a row.

    The text is included so that an upstream re-ordering moves a row's *index*
    without moving the row: the same sentence keeps the same split.
    """
    return f"{source}|{index}|{text[:512]}"


def bucket(key: str, salt: str = "lev-split-v1") -> float:
    """Map a key to a uniform float in [0, 1). Stable across processes.

    `hash()` is deliberately not used: Python salts it per process, so a split
    built today would not reproduce tomorrow.
    """
    digest = hashlib.blake2b(f"{salt}|{key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def assign(
    key: str,
    fractions: dict[Split, float] | None = None,
    salt: str = "lev-split-v1",
) -> Split:
    fractions = fractions or DEFAULT_FRACTIONS
    position = bucket(key, salt)
    cumulative = 0.0
    for split in (Split.TRAIN, Split.CALIBRATION, Split.TEST):
        cumulative += fractions[split]
        if position < cumulative:
            return split
    return Split.TEST  # float error at the top of the range


@dataclass
class SplitReport:
    counts: dict[Split, int]
    missing_from_train: dict[str, list[int]]
    unseen_labels: dict[str, list[int]] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def summary(self) -> str:
        parts = [f"{s.value}={self.counts[s]}" for s in Split]
        return f"{self.total} examples  " + "  ".join(parts)


def split_examples(
    examples: Iterable[Example],
    fractions: dict[Split, float] | None = None,
    salt: str = "lev-split-v1",
) -> dict[Split, list[Example]]:
    out: dict[Split, list[Example]] = {s: [] for s in Split}
    per_source_index: dict[str, int] = defaultdict(int)
    for example in examples:
        index = per_source_index[example.source]
        per_source_index[example.source] += 1
        key = row_key(example.source, index, str(example.state))
        out[assign(key, fractions, salt)].append(example)
    return out


def check_coverage(splits: dict[Split, list[Example]], strict: bool = True) -> SplitReport:
    """Two coverage checks, because one of them is not enough.

    1. Every label *observed* anywhere must also appear in train. A label present
       only in calibration or test is unmeasurable: the model was never shown it,
       so its error rate says nothing about the model's ability.

    2. Every label the *question offers* must be observed at all. This is the
       check that catches a truncated or label-sorted corpus, and the first check
       cannot: if a sample only ever contains 3 of banking77's 77 intents, those
       3 are in train, check 1 is satisfied, and the mixture is still junk. The
       option set comes from the question, so it is what the model is asked to
       choose between regardless of what the sample happens to contain.
    """
    seen: dict[str, set[int]] = defaultdict(set)
    in_train: dict[str, set[int]] = defaultdict(set)
    offered: dict[str, int] = {}
    for split, items in splits.items():
        for example in items:
            seen[example.source].add(example.target)
            # A Noul offers nine rating levels but its data supplies only the two
            # ends, so "every offered label must be observed" does not apply.
            if example.question.type != "noul":
                offered.setdefault(example.source, candidate_count(example.question))
            if split is Split.TRAIN:
                in_train[example.source].add(example.target)

    missing = {
        source: sorted(labels - in_train[source])
        for source, labels in seen.items()
        if labels - in_train[source]
    }
    unseen = {
        source: sorted(set(range(n)) - seen[source])
        for source, n in offered.items()
        if set(range(n)) - seen[source]
    }
    report = SplitReport(
        counts={s: len(items) for s, items in splits.items()},
        missing_from_train=missing,
        unseen_labels=unseen,
    )
    if strict and missing:
        raise ValueError(
            f"labels appear outside train but never in it: {missing}. "
            f"Accuracy on those labels would measure nothing. Draw more rows "
            f"for the affected source."
        )
    if strict and unseen:
        counts = {s: len(v) for s, v in unseen.items()}
        raise ValueError(
            f"these sources never show some of the options their question "
            f"offers: {counts} labels missing from {sorted(unseen)}. The sample "
            f"is too small or the corpus is label-sorted. Raise "
            f"`limit_per_source`; `load_source` already shuffles before it cuts."
        )
    return report
