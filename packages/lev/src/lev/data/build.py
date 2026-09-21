"""Materialise the mixture to disk as three JSONL splits.

Training reads from disk, not from the network. Two reasons, both learned the
expensive way by other people: a run that re-downloads its corpus is not
reproducible, and an H100 that stalls on a rate-limited dataset server is still
being billed.

    lev data build --out data/mixture --limit-per-source 20000

Writes `train.jsonl`, `calibration.jsonl`, `test.jsonl` and a `manifest.json`
recording exactly what went into them -- source ids, row counts, the split salt
and the resolved weights -- so a later run can prove it trained on the same data.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path

from pydantic import TypeAdapter

from ..prompt import Layout
from ..types import Question
from .mixture import Example, MixtureSpec, build_mixture
from .sources import REGISTRY, adjacency_map, default_weights, load_source
from .splits import SPLIT_SALT, Split, check_coverage, split_examples

_QUESTION = TypeAdapter(Question)

SPLIT_FILES = {s: f"{s.value}.jsonl" for s in Split}
MANIFEST = "manifest.json"


def to_json(example: Example) -> dict:
    return {
        "state": example.state,
        "name": example.name,
        "question": _QUESTION.dump_python(example.question, mode="json"),
        "target": example.target,
        "layout": example.layout.value,
        "source": example.source,
        "abstain": example.abstain,
        "soft_target": example.soft_target,
    }


def _question_from(payload: dict, cache: dict[str, Question]) -> Question:
    """Validate a question once per distinct schema, not once per row.

    The mixture has nine distinct questions and 200,000 rows. Building a
    pydantic model per row costs both the validation time and a live object for
    each -- hundreds of MB resident before training starts, for nine values.
    """
    key = json.dumps(payload, sort_keys=True)
    if key not in cache:
        cache[key] = _QUESTION.validate_python(payload)
    return cache[key]


def from_json(row: dict, question_cache: dict[str, Question] | None = None) -> Example:
    return Example(
        state=row["state"],
        name=row["name"],
        question=(
            _question_from(row["question"], question_cache)
            if question_cache is not None
            else _QUESTION.validate_python(row["question"])
        ),
        target=row["target"],
        layout=Layout(row["layout"]),
        source=row["source"],
        abstain=row.get("abstain", False),
        soft_target=row.get("soft_target"),
    )


def write_jsonl(path: Path, examples: Iterable[Example]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(to_json(example), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[Example]:
    cache: dict[str, Question] = {}
    with path.open(encoding="utf-8") as handle:
        return [from_json(json.loads(line), cache) for line in handle if line.strip()]


def build_dataset(
    out_dir: str | Path,
    limit_per_source: int | None = 20_000,
    n_examples: int = 200_000,
    sources: dict[str, float] | None = None,
    schema_first_fraction: float = 0.5,
    abstain_fraction: float = 0.1,
    seed: int = 17,
    cache_dir: str | None = None,
    loader: Callable | None = None,
) -> dict:
    """Load every source, split it, then draw the mixture from the train split.

    Order matters: **split first, mix second.** Mixing first and splitting after
    would let the same underlying row appear in train and in test wearing two
    different layouts, which is a contamination leak with our own data rather
    than S1Bench's -- subtler, and it would flatter the result the same way.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    weights = sources or default_weights()

    pools: dict[str, list[Example]] = {}
    for name in weights:
        pools[name] = list(
            load_source(
                REGISTRY[name], limit=limit_per_source, cache_dir=cache_dir, load_dataset=loader
            )
        )
        if not pools[name]:
            raise ValueError(f"source {name!r} loaded zero rows")

    all_examples = [example for pool in pools.values() for example in pool]
    splits = split_examples(all_examples)
    report = check_coverage(splits)
    by_source = {split: _group_by_source(rows) for split, rows in splits.items()}

    counts: dict[str, int] = {}
    for split in Split:
        available = {name: weight for name, weight in weights.items() if by_source[split].get(name)}
        if not available:
            raise ValueError(f"split {split.value} has no examples from any source")
        total_weight = sum(available.values())
        is_train = split is Split.TRAIN

        mixture = MixtureSpec(
            sources={name: weight / total_weight for name, weight in available.items()},
            # Held-out splits are drawn at their natural size; only train is
            # oversampled to the configured budget.
            n_examples=(
                n_examples if is_train else sum(len(rows) for rows in by_source[split].values())
            ),
            schema_first_fraction=schema_first_fraction,
            # Abstain augmentation is a training device. Unanswerable rows in the
            # test split would measure our abstention, not our accuracy.
            abstain_fraction=abstain_fraction if is_train else 0.0,
            # A different stream per split, so the three do not replay the same
            # layout and abstain decisions in lockstep.
            seed=seed + list(Split).index(split),
            adjacent=adjacency_map(),
        )
        # `name=name` binds the loop variable at definition time; without it every
        # loader would close over the last source in the dict.
        loaders = {
            name: (lambda name=name, split=split: by_source[split][name]) for name in available
        }
        counts[split.value] = write_jsonl(out / SPLIT_FILES[split], build_mixture(mixture, loaders))

    unique_train = sum(len(rows) for rows in by_source[Split.TRAIN].values())
    manifest = {
        "sources": {name: REGISTRY[name].hf_id for name in weights},
        # `build_mixture` draws with replacement, so asking for more examples
        # than the train split holds oversamples it. That is legitimate -- each
        # draw gets its own layout and abstain roll, so the examples differ even
        # when the underlying row repeats -- but the ratio is worth recording,
        # because at 3 epochs on top of it an oversample of 1.4x means the model
        # sees each row about four times.
        "unique_train_rows": unique_train,
        "oversample_ratio": round(n_examples / max(1, unique_train), 2),
        "weights": weights,
        "limit_per_source": limit_per_source,
        "rows_loaded": {name: len(pool) for name, pool in pools.items()},
        "split_counts": counts,
        "raw_split_counts": {split.value: n for split, n in report.counts.items()},
        "schema_first_fraction": schema_first_fraction,
        "abstain_fraction": abstain_fraction,
        "seed": seed,
        "split_salt": SPLIT_SALT,
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _group_by_source(examples: Iterable[Example]) -> dict[str, list[Example]]:
    grouped: dict[str, list[Example]] = {}
    for example in examples:
        grouped.setdefault(example.source, []).append(example)
    return grouped


def load_split(data_dir: str | Path, split: Split) -> list[Example]:
    path = Path(data_dir) / SPLIT_FILES[split]
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. Build it first:\n    uv run lev data build --out {data_dir}"
        )
    return read_jsonl(path)
