"""Export the 13 S1Bench subsets as levbench task files, for evaluation only.

This module must not import training data builders or produce an `Example`;
`test_s1bench.py` checks that separation. Every subset passes `assert_eval_only`.
Ids, questions and labels come from Nimble's pinned manifests in
`s1bench_subsets/`. Conversion details and the multinli metadata exception are
recorded in docs/FINDINGS.md §17.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:  # `datasets` is heavy and only imported where it is used.
    from datasets import Dataset

from ..types import Choice, Noul, Question, Score, question_payload
from .contamination import assert_eval_only

SUBSET_DIR = Path(__file__).with_name("s1bench_subsets")
QUESTION_NAME = "decision"  # S1Bench asks every record one question under this name


@dataclass(frozen=True)
class EvalItem:
    """One record: its S1Bench id, the state to send, and the answer to compare
    against (an option string for Choice, a level index for Score, a bool for Noul)."""

    id: str
    state: dict | str
    truth: bool | int | str


@dataclass(frozen=True)
class EvalSubset:
    """One S1Bench subset: its upstream dataset, and what Jev scored on it.

    `jev_published` is Jev 1.13's figure on this subset; `jev_measured` is the
    board's own `s1-fast` run in `data/s1bench-snapshot.json`, which covered only
    six subsets.
    """

    name: str
    hf_id: str
    jev_published: float
    jev_measured: float | None = None


EVAL_SUBSETS: dict[str, EvalSubset] = {
    subset.name: subset
    for subset in (
        EvalSubset("vitaminc-dev", "tals/vitaminc", 0.801, 0.8030),
        EvalSubset("massive-en-US", "AmazonScience/massive", 0.874, 0.8743),
        EvalSubset("massive-de-DE", "AmazonScience/massive", 0.869),
        EvalSubset("boolq", "google/boolq", 0.897, 0.8933),
        EvalSubset("squad2", "rajpurkar/squad_v2", 0.829),
        EvalSubset("paws", "google-research-datasets/paws", 0.892, 0.8960),
        EvalSubset("multinli", "nyu-mll/multi_nli", 0.829),
        EvalSubset("civil_comments", "google/civil_comments", 0.810),
        EvalSubset("aegis2", "nvidia/Aegis-AI-Content-Safety-Dataset-2.0", 0.804, 0.8360),
        EvalSubset("helpsteer2", "nvidia/HelpSteer2", 0.341, 0.3480),
        EvalSubset("summeval-relevance", "mteb/summeval", 0.350),
        EvalSubset("summeval-consistency", "mteb/summeval", 0.812),
        EvalSubset("pubmedqa", "qiaojin/PubMedQA", 0.772),
    )
}


def subset_names() -> list[str]:
    return list(EVAL_SUBSETS)


def get_subset(name: str) -> EvalSubset:
    """Resolve a subset by name or alias. Raises unless it is S1Bench evaluation data."""
    canonical = name if name in EVAL_SUBSETS else assert_eval_only(name)
    assert_eval_only(canonical)
    return EVAL_SUBSETS[canonical]


def definition(name: str) -> dict:
    """The vendored definition: `instructions`, `criteria`, `ids`, `labels`, provenance."""
    return json.loads((SUBSET_DIR / f"{name}.json").read_text(encoding="utf-8"))


def question_for(spec: dict) -> Question:
    criteria = spec["criteria"]
    if isinstance(criteria, list):
        return Score(instructions=spec["instructions"], criteria=criteria)
    if set(criteria) == {"false", "true"}:
        return Noul(instructions=spec["instructions"], criteria=criteria)
    return Choice(instructions=spec["instructions"], criteria=criteria)


# Readers: every upstream row as an EvalItem, before selection by id.


def _download(repo: str, filename: str, revision: str | None = None) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type="dataset", revision=revision)


def _parquet(repo: str, filename: str, revision: str | None = None) -> Dataset:
    from datasets import load_dataset  # heavy, and only needed to export

    # `split=` narrows load_dataset's return union to one Dataset; the annotation
    # says so, since callers index rows by column name and read `.features`.
    return load_dataset("parquet", data_files=_download(repo, filename, revision), split="train")


def _rows(dataset: Dataset) -> Iterator[dict]:
    """The dataset's rows as dicts.

    `Dataset.__iter__` carries no annotation upstream, so a type checker infers
    `list` for the element and rejects every column lookup. It yields dicts.
    """
    return cast("Iterator[dict]", iter(dataset))


def _jsonl(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="\n") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _read_boolq() -> Iterator[EvalItem]:
    for row in _rows(_parquet("google/boolq", "data/validation-00000-of-00001.parquet")):
        yield EvalItem(
            "boolq-" + _digest(row["question"] + "\n" + row["passage"]),
            {"passage": row["passage"], "question": row["question"]},
            row["answer"],
        )


def _read_paws() -> Iterator[EvalItem]:
    for row in _rows(
        _parquet("google-research-datasets/paws", "labeled_final/test-00000-of-00001.parquet")
    ):
        yield EvalItem(
            f"paws-{row['id']}",
            {"sentence_1": row["sentence1"], "sentence_2": row["sentence2"]},
            row["label"] == 1,
        )


def _read_vitaminc() -> Iterator[EvalItem]:
    for row in _jsonl(_download("tals/vitaminc", "dev.jsonl")):
        yield EvalItem(
            "vitaminc-" + row["unique_id"],
            {"evidence": row["evidence"], "claim": row["claim"]},
            row["label"],
        )


def _read_massive(locale: str) -> Callable[[], Iterator[EvalItem]]:
    def read() -> Iterator[EvalItem]:
        rows = _parquet(
            "AmazonScience/massive", f"{locale}/test/0000.parquet", "refs/convert/parquet"
        )
        scenarios = rows.features["scenario"].names
        for row in _rows(rows):
            yield EvalItem(
                f"massive-{row['id']}",
                {"utterance": row["utt"], "locale": locale},
                scenarios[row["scenario"]],
            )

    return read


def _read_squad2() -> Iterator[EvalItem]:
    for row in _rows(_parquet("rajpurkar/squad_v2", "squad_v2/validation-00000-of-00001.parquet")):
        answerable = any(text.strip() for text in row["answers"]["text"])
        yield EvalItem(
            "squad2-" + row["id"],
            {"paragraph": row["context"], "question": row["question"]},
            answerable,
        )


def _read_multinli() -> Iterator[EvalItem]:
    labels = ("entailment", "neutral", "contradiction")
    for row in _rows(
        _parquet("nyu-mll/multi_nli", "data/validation_matched-00000-of-00001.parquet")
    ):
        if row["label"] in (0, 1, 2):  # -1: no majority label
            yield EvalItem(
                "multinli-" + row["pairID"],
                {"premise": row["premise"].strip(), "hypothesis": row["hypothesis"].strip()},
                labels[row["label"]],
            )


def _read_civil_comments() -> Iterator[EvalItem]:
    # Ids are row positions in the test split, which has no id column.
    rows = _parquet("google/civil_comments", "data/test-00000-of-00001.parquet")
    for index, row in enumerate(_rows(rows)):
        if isinstance(row["text"], str) and row["text"].strip():
            yield EvalItem(f"civil_comments-{index}", row["text"].strip(), row["toxicity"] >= 0.5)


def _read_aegis2() -> Iterator[EvalItem]:
    path = _download("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "test.json")
    for row in json.loads(Path(path).read_text(encoding="utf-8")):
        human = row.get("prompt_label_source") == "human"
        prompt = row.get("prompt")
        if human and row.get("reconstruction_id_if_redacted") is None and prompt and prompt.strip():
            yield EvalItem(
                "aegis2-" + row["id"],
                {"user_message": prompt.strip()},
                row["prompt_label"] == "unsafe",
            )


def _read_helpsteer2() -> Iterator[EvalItem]:
    rows = _jsonl(_download("nvidia/HelpSteer2", "validation.jsonl.gz"))
    for index, row in enumerate(rows):
        yield EvalItem(
            f"helpsteer2-{index}",
            {"prompt": row["prompt"], "response": row["response"]},
            row["helpfulness"],
        )


def _read_summeval(dimension: str) -> Callable[[], Iterator[EvalItem]]:
    def read() -> Iterator[EvalItem]:
        rows = _parquet("mteb/summeval", "data/test-00000-of-00001-35901af5f6649399.parquet")
        for row in _rows(rows):
            for index, summary in enumerate(row["machine_summaries"]):
                # The three-expert 1-5 mean, rounded half up, as a 0-4 level index.
                level = min(4, max(0, math.floor(row[dimension][index] + 0.5) - 1))
                yield EvalItem(
                    f"summeval-{dimension}-{row['id']}-{index}",
                    {"article": row["text"], "summary": summary},
                    level,
                )

    return read


def _read_pubmedqa() -> Iterator[EvalItem]:
    for row in _rows(_parquet("qiaojin/PubMedQA", "pqa_labeled/train-00000-of-00001.parquet")):
        contexts = [c for c in row["context"]["contexts"] if isinstance(c, str) and c.strip()]
        yield EvalItem(
            f"pubmedqa-{row['pubid']}",
            {"question": row["question"], "abstract_context": " ".join(contexts)},
            row["final_decision"].strip().lower(),
        )


_READERS: dict[str, Callable[[], Iterator[EvalItem]]] = {
    "vitaminc-dev": _read_vitaminc,
    "massive-en-US": _read_massive("en-US"),
    "massive-de-DE": _read_massive("de-DE"),
    "boolq": _read_boolq,
    "squad2": _read_squad2,
    "paws": _read_paws,
    "multinli": _read_multinli,
    "civil_comments": _read_civil_comments,
    "aegis2": _read_aegis2,
    "helpsteer2": _read_helpsteer2,
    "summeval-relevance": _read_summeval("relevance"),
    "summeval-consistency": _read_summeval("consistency"),
    "pubmedqa": _read_pubmedqa,
}


def load_eval_subset(name: str) -> tuple[Question, list[EvalItem]]:
    """One S1Bench subset as its question plus exactly its records, sorted by id.

    Raises `ContaminationError` for anything that is not S1Bench evaluation data,
    and `ValueError` if the upstream data no longer yields the pinned ids and labels.
    """
    subset = get_subset(name)
    spec = definition(subset.name)
    by_id = {item.id: item for item in _READERS[subset.name]()}
    missing = [i for i in spec["ids"] if i not in by_id]
    if missing:
        raise ValueError(
            f"{subset.name}: {len(missing)} pinned ids not in the upstream data, e.g. {missing[:3]}"
        )
    items = sorted((by_id[i] for i in spec["ids"]), key=lambda item: item.id)
    labels = dict(Counter(str(item.truth) for item in items))
    if labels != spec["labels"]:
        raise ValueError(
            f"{subset.name}: label counts {labels} differ from the manifest's {spec['labels']}"
        )
    return question_for(spec), items


def export(out_dir: str | Path, subsets: list[str] | None = None) -> dict:
    """Write `<subset>.json` levbench task files, plus an `index.json`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    index: dict[str, dict] = {}
    for name in subsets or subset_names():
        subset = get_subset(name)
        question, items = load_eval_subset(subset.name)
        payload = {
            "questions": {QUESTION_NAME: question_payload(question)},
            "items": [
                {"state": item.state, "labels": {QUESTION_NAME: item.truth}} for item in items
            ],
        }
        (out / f"{subset.name}.json").write_text(json.dumps(payload, indent=2) + "\n")
        index[subset.name] = {
            "items": len(items),
            "type": question.type,
            "jev_published": subset.jev_published,
            "jev_measured": subset.jev_measured,
        }

    summary = {
        "suite": "s1bench",
        "total_items": sum(e["items"] for e in index.values()),
        "subsets": index,
    }
    (out / "index.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
