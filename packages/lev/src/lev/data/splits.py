"""Deterministic, disjoint train, calibration and test splits.

Assignment hashes the source, row position and text. The same input order
reproduces the split across processes; reordering a corpus changes it.
Coverage checks reject labels missing from training or the sampled corpus.
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


# Changing this re-splits every row, making models before and after incomparable.
SPLIT_SALT = "lev-split-v1"

# A temperature needs far fewer rows than a model, and test only has to separate
# two runs.
DEFAULT_FRACTIONS: dict[Split, float] = {
    Split.TRAIN: 0.80,
    Split.CALIBRATION: 0.10,
    Split.TEST: 0.10,
}


def row_key(source: str, index: int, text: str) -> str:
    """A stable identity for a row: its source, its position within that source,
    and the first 512 characters of its text."""
    return f"{source}|{index}|{text[:512]}"


def hash_position(key: str, salt: str = SPLIT_SALT) -> float:
    """Map a key to a uniform float in [0, 1). Stable across processes.

    Not `hash()`, which Python salts per process.
    """
    digest = hashlib.blake2b(f"{salt}|{key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def assign(
    key: str,
    fractions: dict[Split, float] | None = None,
    salt: str = SPLIT_SALT,
) -> Split:
    fractions = fractions or DEFAULT_FRACTIONS
    position = hash_position(key, salt)
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
    salt: str = SPLIT_SALT,
) -> dict[Split, list[Example]]:
    by_split: dict[Split, list[Example]] = {split: [] for split in Split}
    rows_seen_per_source: dict[str, int] = defaultdict(int)
    for example in examples:
        index = rows_seen_per_source[example.source]
        rows_seen_per_source[example.source] += 1
        key = row_key(example.source, index, str(example.state))
        by_split[assign(key, fractions, salt)].append(example)
    return by_split


def check_coverage(splits: dict[Split, list[Example]], strict: bool = True) -> SplitReport:
    """Two coverage checks.

    1. Every label observed anywhere also appears in train; one seen only in
       calibration or test measures nothing.
    2. Every label the question offers is observed at all. This catches a
       truncated or label-sorted corpus, which check 1 cannot: a sample holding
       3 of banking77's 77 intents passes check 1.
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

    missing: dict[str, list[int]] = {}
    for source, labels in seen.items():
        if absent := labels - in_train[source]:
            missing[source] = sorted(absent)

    unseen: dict[str, list[int]] = {}
    for source, n_offered in offered.items():
        if never_seen := set(range(n_offered)) - seen[source]:
            unseen[source] = sorted(never_seen)

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
