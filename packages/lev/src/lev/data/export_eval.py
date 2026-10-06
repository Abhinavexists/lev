"""Export held-out rows as levbench task files, one question per source."""

from __future__ import annotations

import json
from pathlib import Path

from ..labels import noul_probability
from ..types import Noul, Question, Score, question_payload
from .build import _group_by_source, load_split
from .splits import Split

# Require enough rows for a useful accuracy estimate.
MIN_USEFUL_ITEMS = 100


def truth_for(question: Question, target: int) -> bool | int | str:
    """Convert a target to a Choice key, Score index, or Noul bool."""
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
        # Skip per-row option sets: each task file has one shared question.
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
