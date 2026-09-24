"""Write the held-out test split out as levbench task files.

levbench does not import `lev` (ADR-010), so the handoff is a file written here.

    uv run lev data eval --data data/mixture --out data/eval

One file per source, because each source has its own question; a combined file
would ask every question of every item.

Truth values are written in the shape levbench compares against per primitive:
a Choice yields its option string, a Score its level index, a Noul a bool.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..labels import noul_probability
from ..types import Noul, Score, question_payload
from .build import _group_by_source, load_split
from .splits import Split

# Below this an accuracy is not a measurement: at n=24 the 95% interval is
# about +/-16 points.
MIN_USEFUL_ITEMS = 100


def truth_for(question, target: int):
    if isinstance(question, Noul):
        # The target is a rating index; collapse it as the readout does.
        return noul_probability({target: 1.0}) >= 0.5
    if isinstance(question, Score):
        return target
    return list(question.criteria)[target]


def export(
    data_dir: str | Path,
    out_dir: str | Path,
    split: Split = Split.TEST,
    limit_per_source: int | None = None,
) -> dict:
    """Write `<source>.json` per source, plus an `index.json` describing them."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Scoring abstain rows would measure abstention, not accuracy.
    answerable = [row for row in load_split(data_dir, split) if not row.abstain]
    by_source = _group_by_source(answerable)

    index: dict[str, dict] = {}
    variable: list[str] = []
    for source, rows in sorted(by_source.items()):
        if limit_per_source:
            rows = rows[:limit_per_source]
        question = rows[0].question
        # A task file has one question for all items, so QA sources with per-row
        # options (race, sciq, ...) are skipped; `evaluate_split` still scores them.
        payloads = {json.dumps(question_payload(r.question), sort_keys=True) for r in rows}
        if len(payloads) != 1:
            variable.append(source)
            continue
        payload = {
            "questions": {source: question_payload(question)},
            "items": [
                {"state": row.state, "labels": {source: truth_for(row.question, row.target)}}
                for row in rows
            ],
        }
        (out / f"{source}.json").write_text(json.dumps(payload, indent=2) + "\n")
        index[source] = {"items": len(rows), "type": question.type}

    total = sum(entry["items"] for entry in index.values())
    if total < MIN_USEFUL_ITEMS:
        raise ValueError(
            f"only {total} eval items across {len(index)} sources. Below "
            f"~{MIN_USEFUL_ITEMS} the accuracy interval is wider than any "
            f"improvement training would produce, so the number cannot tell you "
            f"whether the run helped. Rebuild the mixture with a larger "
            f"`--limit-per-source`."
        )

    summary = {
        "split": split.value,
        "total_items": total,
        "sources": index,
        "skipped_variable_question": variable,
    }
    (out / "index.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
