"""Write the held-out test split out as levbench task files.

This lives on the model side of the ADR-010 fence on purpose. levbench must not
import `lev` -- an instrument that imports what it measures is not an instrument
-- so the handoff is a file, and this is the thing that writes it.

    uv run lev data eval --data data/mixture --out data/eval

One file per source, because each source carries exactly one question. A single
combined file would make the harness ask every question of every item, i.e. ask
"how positive is this review?" of a banking ticket, and score the answer.

Truth values are written in the shape levbench compares against per primitive:
a Choice yields its option string, a Score its level index, a Noul a bool.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..labels import noul_probability
from ..types import Choice, Noul, Score
from .build import _group_by_source, load_split
from .splits import Split

# Below this, an accuracy number is not a measurement. At n=24 the 95% interval
# is about +/-16 points, so a 5-point improvement is invisible.
MIN_USEFUL_ITEMS = 100


def truth_for(question, target: int):
    if isinstance(question, Noul):
        # The stored target is a rating index, not a class. Collapse it the same
        # way the readout will, so the comparison is like-for-like.
        return noul_probability({target: 1.0}) >= 0.5
    if isinstance(question, Score):
        return target
    return list(question.criteria)[target]


def question_payload(question) -> dict:
    """The question as levbench will rebuild it. `criteria` is required on a
    Choice and a Score and optional on a Noul, so an absent Noul criteria map
    must stay absent rather than round-tripping as an empty one."""
    payload: dict = {"type": question.type, "instructions": question.instructions}
    if isinstance(question, Choice | Score) or question.criteria:
        payload["criteria"] = question.criteria
    return payload


def export(
    data_dir: str | Path,
    out_dir: str | Path,
    split: Split = Split.TEST,
    limit_per_source: int | None = None,
) -> dict:
    """Write `<source>.json` per source, plus an `index.json` describing them."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Abstain rows are a training device: scoring them measures abstention
    # rather than accuracy, and the two are different numbers.
    answerable = [row for row in load_split(data_dir, split) if not row.abstain]
    by_source = _group_by_source(answerable)

    index: dict[str, dict] = {}
    variable: list[str] = []
    for source, rows in sorted(by_source.items()):
        if limit_per_source:
            rows = rows[:limit_per_source]
        question = rows[0].question
        # A levbench task file carries one question for every item. QA sources
        # (race, sciq, ...) supply options per row, so they cannot be written in
        # this format; they are still scored in-process by `evaluate_split`.
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
