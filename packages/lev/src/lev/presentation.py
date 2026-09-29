"""Presentation checks: does an answer move with where an option is shown?

The design of NandhaKishorM/laya#259, over `system_one` (ADR-029). Per question type
(Score and Choice), on any list of states, with no labels:

- **identical**: every option carries the same text, so options differ only by
  position and code. Metric: slot 0's log-probability minus the mean over the slots,
  averaged over states and configurations. 0 is no position effect; below 0, slot 0
  is disfavoured. Choice keys must be unique, so they are numbered (`1`, `2`, ...)
  and share one description, as Score levels share one text.
- **first_slot**: three real options in all 3! = 6 orders per state. Every option
  sits in every slot exactly twice, so an order-free readout picks slot 0 in exactly
  1/3 of the decisions.
- **packed**: the probe asks its questions in as few `system_one` calls as fit
  `MAX_SCORE_ROWS` (one, by default); this asks each question alone on a few states and
  reports the largest probability difference, so what is measured is the presentation
  and not the packing.

Probabilities are whatever `system_one` returns: calibrated, unless the engine carries
an empty `CalibrationProfile`. Within one question, calibrated log-probabilities are the
raw scores divided by the bucket's temperature, so the identical metric shrinks by 1/T
and the first-slot rate does not change. The proposed gate, -0.20, is Laya's, set on
raw scores.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

KINDS = ("score", "choice")
IDENTICAL_KS = (3, 4, 5)
SLOT0_MIN = -0.20
FIRST_SLOT_MIN = 0.15
# Score rows per `system_one` call. The default probe is 12 rows off and 24 reversed, one call
# either way; cyclic reads a K-level question K times (42 rows), which filled a 32 GB GPU.
MAX_SCORE_ROWS = 24

# en follows Laya's presentation_checks.py; ja the same wording in Japanese.
TEXT: dict[str, dict[str, Any]] = {
    "en": {
        "score": {
            "instructions": "How urgent is this request?",
            "identical": ("moderate", "a request"),
            "options": ("Not urgent", "Soon", "Work is blocked"),
        },
        "choice": {
            "instructions": "Which team should handle this message?",
            "identical": ("a team", "a request"),
            "options": {
                "Billing": "payments, refunds, invoices",
                "Technical": "bugs, outages, errors",
                "Sales": "plans, contracts, quotes",
            },
        },
    },
    "ja": {
        "score": {
            "instructions": "この依頼の緊急度は",
            "identical": ("中程度", "依頼"),
            "options": ("急がない", "早めに", "業務が止まっている"),
        },
        "choice": {
            "instructions": "この問い合わせはどの部署が担当すべきか",
            "identical": ("部署", "依頼"),
            "options": {
                "請求": "支払い・返金・請求書",
                "技術": "不具合・障害・エラー",
                "営業": "料金プラン・契約・見積もり",
            },
        },
    },
}

SystemOne = Callable[[Any, Mapping[str, dict]], Any]


def identical_question(kind: str, text: str, k: int, instructions: str) -> dict:
    if kind == "score":
        return {"type": "score", "instructions": instructions, "criteria": [text] * k}
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": {str(i + 1): text for i in range(k)},
    }


def probe_questions(lang: str = "en", kinds: Sequence[str] = KINDS) -> dict[str, dict]:
    """Every probe question for one state, keyed `kind|identical|text|k` or
    `kind|perm|012`. One `system_one` call answers them all."""
    questions: dict[str, dict] = {}
    for kind in kinds:
        spec = TEXT[lang][kind]
        for text in spec["identical"]:
            for k in IDENTICAL_KS:
                questions[f"{kind}|identical|{text}|{k}"] = identical_question(
                    kind, text, k, spec["instructions"]
                )
        options = list(spec["options"])
        for order in itertools.permutations(range(len(options))):
            shown = [options[i] for i in order]
            criteria = shown if kind == "score" else {key: spec["options"][key] for key in shown}
            questions[f"{kind}|perm|{''.join(map(str, order))}"] = {
                "type": kind,
                "instructions": spec["instructions"],
                "criteria": criteria,
            }
    return questions


def slot_probs(answer: Any, question: dict) -> list[float]:
    """An answer's probabilities in the order the options were shown."""
    probs = answer.probabilities
    if question["type"] == "score":
        return [float(probs[i]) for i in range(len(question["criteria"]))]
    return [float(probs[key]) for key in question["criteria"]]


def leave_one_out(per_state: Sequence[float]) -> list[float]:
    n, total = len(per_state), sum(per_state)
    loo = [(total - x) / (n - 1) for x in per_state] if n > 1 else list(per_state)
    return [min(loo), max(loo)]


def score_rows(question: dict, score_order_average: str) -> int:
    """Batch rows a Score question adds: one per order it is read in (`_orders`)."""
    k = len(question["criteria"])
    return {"off": 1, "reversed": 2, "cyclic": k}[score_order_average] if k >= 2 else 1


def call_groups(
    questions: dict[str, dict],
    split_checks: bool = False,
    max_score_rows: int = MAX_SCORE_ROWS,
    score_order_average: str = "off",
) -> list[dict[str, dict]]:
    """The probe split into `system_one` calls. By default one call, cut only where the
    Score rows would pass `max_score_rows`, which bounds memory under `cyclic`; with
    `split_checks` the identical and permuted questions never share a call."""
    groups: list[dict[str, dict]] = []
    for check in ("identical", "perm") if split_checks else (None,):
        current: dict[str, dict] = {}
        rows = 0
        for qid, question in questions.items():
            if check is not None and qid.split("|")[1] != check:
                continue
            added = score_rows(question, score_order_average) if question["type"] == "score" else 0
            if current and rows + added > max_score_rows:
                groups.append(current)
                current, rows = {}, 0
            current[qid] = question
            rows += added
        if current:
            groups.append(current)
    return groups


def run(
    system_one: SystemOne,
    states: Sequence[Any],
    lang: str = "en",
    kinds: Sequence[str] = KINDS,
    slot0_min: float = SLOT0_MIN,
    split_checks: bool = False,
    max_score_rows: int = MAX_SCORE_ROWS,
    score_order_average: str = "off",
) -> dict[str, dict]:
    """Both checks for each kind, from the `call_groups` calls per state (one by default)."""
    questions = probe_questions(lang, kinds)
    groups = call_groups(questions, split_checks, max_score_rows, score_order_average)
    slot0: dict[str, list[float]] = {kind: [] for kind in kinds}
    first: dict[str, list[float]] = {kind: [] for kind in kinds}
    by_config: dict[str, dict[str, list[float]]] = {kind: {} for kind in kinds}
    by_slot = {kind: [0] * len(TEXT[lang][kind]["options"]) for kind in kinds}
    for state in states:
        answers = {}
        for group in groups:
            answers.update(system_one(state, group).answers)
        centred: dict[str, list[float]] = {kind: [] for kind in kinds}
        firsts: dict[str, list[int]] = {kind: [] for kind in kinds}
        for qid, question in questions.items():
            kind, check, *rest = qid.split("|")
            p = slot_probs(answers[qid], question)
            if check == "identical":
                logp = [math.log(max(x, 1e-12)) for x in p]
                value = logp[0] - sum(logp) / len(logp)
                centred[kind].append(value)
                by_config[kind].setdefault("|".join(rest), []).append(value)
            else:
                slot = max(range(len(p)), key=p.__getitem__)
                firsts[kind].append(int(slot == 0))
                by_slot[kind][slot] += 1
        for kind in kinds:
            slot0[kind].append(sum(centred[kind]) / len(centred[kind]))
            first[kind].append(sum(firsts[kind]) / len(firsts[kind]))
    report: dict[str, dict] = {}
    for kind in kinds:
        identical = sum(slot0[kind]) / len(slot0[kind])
        rate = sum(first[kind]) / len(first[kind])
        report[kind] = {
            "identical": {
                "metric": identical,
                "leave_one_out": leave_one_out(slot0[kind]),
                "gate": slot0_min,
                "passed": identical >= slot0_min,
                "by_config": {c: sum(v) / len(v) for c, v in by_config[kind].items()},
            },
            "first_slot": {
                "metric": rate,
                "leave_one_out": leave_one_out(first[kind]),
                "gate": FIRST_SLOT_MIN,
                "passed": rate >= FIRST_SLOT_MIN,
                "decisions": len(states) * math.factorial(len(TEXT[lang][kind]["options"])),
                "argmax_by_slot": by_slot[kind],
            },
        }
    return report


def packed_consistency(
    system_one: SystemOne,
    states: Sequence[Any],
    lang: str = "en",
    kinds: Sequence[str] = KINDS,
    split_checks: bool = False,
    max_score_rows: int = MAX_SCORE_ROWS,
    score_order_average: str = "off",
) -> float:
    """Largest |p| difference between the probe as `run` asks it and each question alone."""
    questions = probe_questions(lang, kinds)
    groups = call_groups(questions, split_checks, max_score_rows, score_order_average)
    worst = 0.0
    for state in states:
        packed = {}
        for group in groups:
            packed.update(system_one(state, group).answers)
        for qid, question in questions.items():
            alone = system_one(state, {qid: question}).answers[qid]
            for a, b in zip(
                slot_probs(packed[qid], question), slot_probs(alone, question), strict=True
            ):
                worst = max(worst, abs(a - b))
    return worst


def drift(report: dict, snapshot: dict, tolerance: float) -> list[str]:
    """Every metric that moved more than `tolerance` from the committed report."""
    moved = []
    for kind, checks in snapshot.items():
        for check, committed in checks.items():
            now = report.get(kind, {}).get(check, {}).get("metric")
            if now is None:
                moved.append(f"{kind}.{check}: missing from this run")
            elif abs(now - committed["metric"]) > tolerance:
                moved.append(f"{kind}.{check}: {committed['metric']:+.3f} -> {now:+.3f}")
    return moved


def to_markdown(report: dict) -> str:
    lines = [
        "| kind | identical (slot 0 - mean) | leave-one-out | first-slot rate | leave-one-out |",
        "|---|---|---|---|---|",
    ]
    for kind, r in report.items():
        ident, first = r["identical"], r["first_slot"]
        lines.append(
            f"| {kind} | {ident['metric']:+.3f} {'PASS' if ident['passed'] else 'FAIL'} "
            f"| {ident['leave_one_out'][0]:+.3f} .. {ident['leave_one_out'][1]:+.3f} "
            f"| {first['metric']:.3f} {'PASS' if first['passed'] else 'FAIL'} "
            f"| {first['leave_one_out'][0]:.3f} .. {first['leave_one_out'][1]:.3f} |"
        )
    return "\n".join(lines)


def read_states(path: str) -> list[Any]:
    """`state` from each line of a JSONL file (bench_en.jsonl, bench_ja.jsonl or your own)."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line)["state"] for line in fh if line.strip()]
