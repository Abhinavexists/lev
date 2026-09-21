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
from .build import load_split
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

    examples = [e for e in load_split(data_dir, split) if not e.abstain]
    by_source: dict[str, list] = {}
    for example in examples:
        by_source.setdefault(example.source, []).append(example)

    index: dict[str, dict] = {}
    for source, rows in sorted(by_source.items()):
        if limit_per_source:
            rows = rows[:limit_per_source]
        question = rows[0].question
        payload = {
            "questions": {source: question_payload(question)},
            "items": [
                {"state": r.state, "labels": {source: truth_for(r.question, r.target)}}
                for r in rows
            ],
        }
        (out / f"{source}.json").write_text(json.dumps(payload, indent=2) + "\n")
        index[source] = {"items": len(rows), "type": question.type}

    total = sum(v["items"] for v in index.values())
    if total < MIN_USEFUL_ITEMS:
        raise ValueError(
            f"only {total} eval items across {len(index)} sources. Below "
            f"~{MIN_USEFUL_ITEMS} the accuracy interval is wider than any "
            f"improvement training would produce, so the number cannot tell you "
            f"whether the run helped. Rebuild the mixture with a larger "
            f"`--limit-per-source`."
        )

    (out / "index.json").write_text(
        json.dumps({"split": split.value, "total_items": total, "sources": index}, indent=2) + "\n"
    )
    return {"split": split.value, "total_items": total, "sources": index}
