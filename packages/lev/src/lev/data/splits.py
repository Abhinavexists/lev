"""Deterministic splits keyed by source and row position; input order matters (ADR-014)."""

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

DEFAULT_FRACTIONS: dict[Split, float] = {
    Split.TRAIN: 0.80,
    Split.CALIBRATION: 0.10,
    Split.TEST: 0.10,
}


def row_key(source: str, index: int, text: str) -> str:
    return f"{source}|{index}|{text[:512]}"


def hash_position(key: str, salt: str = SPLIT_SALT) -> float:
    """Map a key to a process-stable float in [0, 1)."""
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
    """Check observed labels appear in train and offered labels appear in the data."""
    seen: dict[str, set[int]] = defaultdict(set)
    in_train: dict[str, set[int]] = defaultdict(set)
    offered: dict[str, int] = {}
    for split, items in splits.items():
        for example in items:
            seen[example.source].add(example.target)
            # Binary Noul data only covers the rating scale's endpoints.
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
