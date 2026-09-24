"""Load S1Bench evaluation subsets, for evaluation and nothing else.

The one place in `lev` allowed to read the 13 subsets `contamination.assert_clean`
refuses. The separation is structural: this module never imports `mixture` or
`build`, never produces an `Example`, and writes only levbench task files, so no
training entry point can route a blocked subset into the mixture.
`tests/test_s1bench.py` asserts that import closure. Every subset is read through
`assert_eval_only`.

The task files match `export_eval.py`'s, so one `levbench eval --tasks` run scores
Jev and lev on identical files; a gap against Jev's published numbers means
little until Jev has run the same files.

Only the 6 subsets that ran in `s1-fast` have a measured Jev accuracy to validate
a loader against, so only they are in `EVAL_SUBSETS`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from ..types import Choice, Noul, Question, Score, question_payload
from .contamination import assert_eval_only


@dataclass(frozen=True)
class EvalSubset:
    """One S1Bench subset: where to read it, and what Jev scored on it.

    `jev_measured` is the `s1-fast` accuracy in `data/s1bench-snapshot.json`;
    `jev_published` is TypeSafe's figure. They agree within 0.7 pp on five of the
    six; `aegis2` differs by 3.2 pp, so only its measured number is a target.
    """

    name: str
    hf_id: str
    primitive: str
    items: int
    jev_measured: float
    jev_published: float
    hf_config: str | None = None
    split: str = "validation"
    # Script-backed repos have no parquet in `main` and `datasets>=5` will not run
    # the script, so this names the auto-converted branch.
    revision: str | None = None
    # `massive`'s converted branch exposes locales as directories, not configs.
    data_files: str | None = None

    @property
    def agreement_pp(self) -> float:
        """How far the published and measured numbers sit apart, in points."""
        return abs(self.jev_measured - self.jev_published) * 100


EVAL_SUBSETS: dict[str, EvalSubset] = {
    "vitaminc-dev": EvalSubset(
        name="vitaminc-dev",
        hf_id="tals/vitaminc",
        primitive="choice",
        items=599,
        jev_measured=0.8030,
        jev_published=0.8010,
    ),
    "massive-en-US": EvalSubset(
        name="massive-en-US",
        hf_id="AmazonScience/massive",
        primitive="choice",
        items=350,
        jev_measured=0.8743,
        jev_published=0.8740,
        revision="refs/convert/parquet",
        data_files="en-US/validation/0000.parquet",
    ),
    "boolq": EvalSubset(
        name="boolq",
        hf_id="google/boolq",
        primitive="noul",
        items=300,
        jev_measured=0.8933,
        jev_published=0.8970,
    ),
    "aegis2": EvalSubset(
        name="aegis2",
        hf_id="nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
        primitive="noul",
        items=250,
        jev_measured=0.8360,
        jev_published=0.8040,
    ),
    "paws": EvalSubset(
        name="paws",
        hf_id="google-research-datasets/paws",
        hf_config="labeled_final",
        primitive="noul",
        items=250,
        jev_measured=0.8960,
        jev_published=0.8920,
    ),
    "helpsteer2": EvalSubset(
        name="helpsteer2",
        hf_id="nvidia/HelpSteer2",
        primitive="score",
        items=250,
        jev_measured=0.3480,
        jev_published=0.3410,
    ),
}


def subset_names() -> list[str]:
    """The subsets this harness can load, in snapshot order."""
    return list(EVAL_SUBSETS)


def get_subset(name: str) -> EvalSubset:
    """Resolve a subset by name or alias. Raises unless it is S1Bench evaluation data."""
    canonical = assert_eval_only(name)
    if canonical not in EVAL_SUBSETS:
        raise KeyError(
            f"{name!r} resolves to the blocked subset {canonical!r}, which has no "
            f"loader: it never ran in `s1-fast`, so there is no measured Jev "
            f"accuracy to validate one against. Loadable subsets: "
            f"{', '.join(subset_names())}."
        )
    return EVAL_SUBSETS[canonical]


# Each reader returns `(question, items)`, with questions built as `lev.types`
# models so a malformed schema fails on load, not mid-run.


@dataclass(frozen=True)
class EvalItem:
    """One scored row: the state to send, and the answer to compare against.

    `truth` is written in the shape levbench compares per primitive -- an option
    string for Choice, a level index for Score, a bool for Noul -- matching
    `export_eval.truth_for`.
    """

    state: dict
    truth: bool | int | str


def humanise(label: str) -> str:
    """`iot_hue_lightchange` -> `iot hue lightchange`, for option descriptions."""
    return re.sub(r"[_\-]+", " ", label).strip()


def _sample(dataset, limit: int, seed: int):
    """Shuffle before limiting, always.

    A head slice is not a sample: the first N rows of `massive`'s validation
    split under-cover its 60 intents.
    """
    if limit < len(dataset):
        return dataset.shuffle(seed=seed).select(range(limit))
    return dataset


def _read_boolq(dataset) -> tuple[Noul, list[EvalItem]]:
    question = Noul(
        instructions="Does the passage support answering yes to the question?",
        criteria={"true": "the passage supports yes", "false": "the passage supports no"},
    )
    items = [
        EvalItem(
            state={"passage": r["passage"], "question": r["question"]}, truth=bool(r["answer"])
        )
        for r in dataset
    ]
    return question, items


def _read_paws(dataset) -> tuple[Noul, list[EvalItem]]:
    question = Noul(
        instructions="Do the two sentences mean the same thing?",
        criteria={"true": "they are paraphrases", "false": "they differ in meaning"},
    )
    items = [
        EvalItem(
            state={"sentence1": r["sentence1"], "sentence2": r["sentence2"]},
            truth=bool(r["label"]),
        )
        for r in dataset
    ]
    return question, items


def _read_aegis2(dataset) -> tuple[Noul, list[EvalItem]]:
    question = Noul(
        instructions="Is this prompt unsafe?",
        criteria={"true": "unsafe", "false": "safe"},
    )
    items = [
        EvalItem(state={"prompt": r["prompt"]}, truth=r["prompt_label"] == "unsafe")
        for r in dataset
        if r["prompt_label"] in ("safe", "unsafe")
    ]
    return question, items


def _read_vitaminc(dataset) -> tuple[Choice, list[EvalItem]]:
    options = {
        "SUPPORTS": "the evidence supports the claim",
        "REFUTES": "the evidence contradicts the claim",
        "NOT ENOUGH INFO": "the evidence neither supports nor contradicts the claim",
    }
    question = Choice(
        instructions="Does the evidence support or refute the claim?", criteria=options
    )
    items = [
        EvalItem(state={"claim": r["claim"], "evidence": r["evidence"]}, truth=r["label"])
        for r in dataset
        if r["label"] in options
    ]
    return question, items


def _read_massive(dataset) -> tuple[Choice, list[EvalItem]]:
    """60 intents, the largest option set here; within the 68-code Mode A limit."""
    names = dataset.features["intent"].names
    question = Choice(
        instructions="What is the user's intent?",
        criteria={name: humanise(name) for name in names},
    )
    items = [EvalItem(state={"utterance": r["utt"]}, truth=names[r["intent"]]) for r in dataset]
    return question, items


# HelpSteer2 rates on 0-4; the levels are the dataset's own rubric.
_HELPSTEER_LEVELS = [
    "not helpful at all",
    "borderline unhelpful",
    "partially helpful",
    "mostly helpful",
    "fully helpful",
]


def _read_helpsteer2(dataset) -> tuple[Score, list[EvalItem]]:
    question = Score(
        instructions="How helpful is the response to the prompt?", criteria=_HELPSTEER_LEVELS
    )
    items = [
        EvalItem(
            state={"prompt": r["prompt"], "response": r["response"]}, truth=int(r["helpfulness"])
        )
        for r in dataset
        if r["helpfulness"] is not None
    ]
    return question, items


_READERS: dict[str, Callable] = {
    "boolq": _read_boolq,
    "paws": _read_paws,
    "aegis2": _read_aegis2,
    "vitaminc-dev": _read_vitaminc,
    "massive-en-US": _read_massive,
    "helpsteer2": _read_helpsteer2,
}


# Fixed, so two harness runs score the same rows.
SAMPLE_SEED = 20260922


def load_eval_subset(
    name: str, limit: int | None = None, seed: int = SAMPLE_SEED
) -> tuple[Question, list[EvalItem]]:
    """Load one S1Bench subset as a question plus scored items.

    Raises `ContaminationError` for anything that is not S1Bench evaluation
    data. `limit` defaults to the item count S1Bench itself ran, so accuracy is
    comparable to Jev's figure over the same nominal N.
    """
    from datasets import load_dataset  # heavy, and not needed to inspect the registry

    spec = get_subset(name)
    kwargs: dict = {"path": spec.hf_id, "split": spec.split}
    if spec.hf_config:
        kwargs["name"] = spec.hf_config
    if spec.revision:
        kwargs["revision"] = spec.revision
    if spec.data_files:
        kwargs["data_files"] = {spec.split: spec.data_files}

    dataset = load_dataset(**kwargs)
    question, items = _READERS[spec.name](_sample(dataset, limit or spec.items, seed))
    return question, items


def export(
    out_dir: str | Path, subsets: Iterable[str] | None = None, limit: int | None = None
) -> dict:
    """Write `<subset>.json` levbench task files, plus an `index.json`.

    The same format as `export_eval.export`, so one `levbench eval --tasks`
    run scores Jev and lev through identical code.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    index: dict[str, dict] = {}
    for name in subsets or subset_names():
        spec = get_subset(name)
        question, items = load_eval_subset(name, limit)
        payload = {
            "questions": {spec.name: question_payload(question)},
            "items": [{"state": item.state, "labels": {spec.name: item.truth}} for item in items],
        }
        (out / f"{spec.name}.json").write_text(json.dumps(payload, indent=2) + "\n")
        index[spec.name] = {
            "items": len(items),
            "type": question.type,
            "jev_measured": spec.jev_measured,
            "jev_published": spec.jev_published,
        }

    summary = {
        "suite": "s1bench",
        "seed": SAMPLE_SEED,
        "total_items": sum(entry["items"] for entry in index.values()),
        "subsets": index,
    }
    (out / "index.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
