"""Presentation checks: does an answer move with where an option is shown?

The design of NandhaKishorM/laya#259, over `system_one` (ADR-029). Per question type
(Score and Choice), on any list of states, with no labels:

- **identical**: the position bias of one read, before any order averaging. Every
  option carries the same text, so options differ only by position and code. Metric:
  slot 0's log-probability minus the mean over the slots, averaged over states and
  configurations; 0 is no position effect. It is read with `order_average=False`,
  because averaged orders cancel it by construction (cyclic) or hide the middle slots
  (reversed): it measures the bias averaging has to remove, not what is left after it.
  Choice keys must be unique, so they are shapes with no order (`KEYS`), and every
  assignment of keys to slots is read, so a key's own pull cancels.
- **first_slot**: three real options in all 3! = 6 orders per state, read as the engine
  is configured. An order-free readout picks slot 0 in exactly 1/3 of the decisions; a
  tie gives each of its m slots 1/m. `consistent` is the share of states whose six orders
  all give one answer. In the chat style a Score line names its level by its place in
  the request (`(level i of K)`), so a permuted listing renumbers the levels too.
- **packed**: the largest probability difference between the probe's answers and each
  question asked alone, so what is measured is the presentation and not the packing.

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
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

KINDS = ("score", "choice")
IDENTICAL_KS = (3, 4, 5)
KEYS = ("■", "●", "◆", "▲", "◎")
SLOT0_MIN = -0.20
FIRST_SLOT_MIN = 0.15

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

# Settings a report records. The first group is known before the model loads.
PRE_LOAD = ("n_states", "lang", "raw", "kinds", "score_order_average", "max_rows")
POST_LOAD = ("checkpoint_revision", "dtype")
SETTINGS = PRE_LOAD + POST_LOAD


def identical_questions(kind: str, text: str, k: int, instructions: str) -> dict[str, dict]:
    """The identical-option questions of one configuration, keyed by key assignment."""
    if kind == "score":
        return {"": {"type": "score", "instructions": instructions, "criteria": [text] * k}}
    return {
        "".join(map(str, keys)): {
            "type": "choice",
            "instructions": instructions,
            "criteria": {KEYS[i]: text for i in keys},
        }
        for keys in itertools.permutations(range(k))
    }


def probe_questions(lang: str = "en", kinds: Sequence[str] = KINDS) -> dict[str, dict]:
    """Every probe question for one state, keyed `kind|identical|text|k|keys` or
    `kind|perm|012`."""
    questions: dict[str, dict] = {}
    for kind in kinds:
        spec = TEXT[lang][kind]
        for text in spec["identical"]:
            for k in IDENTICAL_KS:
                for keys, q in identical_questions(kind, text, k, spec["instructions"]).items():
                    questions[f"{kind}|identical|{text}|{k}|{keys}"] = q
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


def is_identical(qid: str) -> bool:
    return qid.split("|")[1] == "identical"


@contextmanager
def single_order(engine) -> Iterator[None]:
    """Read every question once, in the order given."""
    config = engine.config
    engine.config = replace(config, order_average=False)
    try:
        yield
    finally:
        engine.config = config


def call_groups(engine, questions: dict[str, dict], max_rows: int) -> list[dict[str, dict]]:
    """`questions` split into `system_one` calls of at most `max_rows` batch rows, as the
    engine counts them under its current config."""
    rows = engine.question_rows(questions)
    groups: list[dict[str, dict]] = []
    current: dict[str, dict] = {}
    total = 0
    for qid, question in questions.items():
        if current and total + rows[qid] > max_rows:
            groups.append(current)
            current, total = {}, 0
        current[qid] = question
        total += rows[qid]
    if current:
        groups.append(current)
    return groups


def ask(
    engine,
    states: Sequence[Any],
    lang: str = "en",
    kinds: Sequence[str] = KINDS,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Every probe answer for every state: the identical questions in one order, the
    permuted ones as the engine is configured. Calls hold at most `max_rows` batch rows,
    the engine's `max_request_rows` by default."""
    max_rows = max_rows or engine.config.max_request_rows
    questions = probe_questions(lang, kinds)
    identical = {q: v for q, v in questions.items() if is_identical(q)}
    permuted = {q: v for q, v in questions.items() if not is_identical(q)}
    with single_order(engine):
        identical_groups = call_groups(engine, identical, max_rows)
    permuted_groups = call_groups(engine, permuted, max_rows)
    answers = []
    for state in states:
        got: dict[str, Any] = {}
        with single_order(engine):
            for group in identical_groups:
                got.update(engine.system_one(state, group).answers)
        for group in permuted_groups:
            got.update(engine.system_one(state, group).answers)
        answers.append(got)
    return answers


def slot_probs(answer: Any, question: dict) -> list[float]:
    """An answer's probabilities in the order the options were shown."""
    probs = answer.probabilities
    if question["type"] == "score":
        return [float(probs[i]) for i in range(len(question["criteria"]))]
    return [float(probs[key]) for key in question["criteria"]]


def top_slots(p: Sequence[float]) -> list[int]:
    best = max(p)
    return [i for i, x in enumerate(p) if x == best]


def leave_one_out(per_state: Sequence[float]) -> list[float]:
    n, total = len(per_state), sum(per_state)
    loo = [(total - x) / (n - 1) for x in per_state] if n > 1 else list(per_state)
    return [min(loo), max(loo)]


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def report(
    answers: Sequence[Mapping[str, Any]],
    lang: str = "en",
    kinds: Sequence[str] = KINDS,
    slot0_min: float = SLOT0_MIN,
) -> dict[str, dict]:
    """Both checks for each kind from `ask`'s answers."""
    questions = probe_questions(lang, kinds)
    out: dict[str, dict] = {}
    for kind in kinds:
        options = TEXT[lang][kind]["options"]
        slot0, first, consistent = [], [], 0
        by_config: dict[str, list[float]] = {}
        by_slot = [0.0] * len(options)
        for got in answers:
            centred: dict[str, list[float]] = {}
            firsts, picks = [], set()
            for qid, question in questions.items():
                if not qid.startswith(kind + "|"):
                    continue
                p = slot_probs(got[qid], question)
                if is_identical(qid):
                    logp = [math.log(max(x, 1e-12)) for x in p]
                    config = "|".join(qid.split("|")[2:4])
                    centred.setdefault(config, []).append(logp[0] - mean(logp))
                    continue
                order = [int(c) for c in qid.split("|")[2]]
                top = top_slots(p)
                for slot in top:
                    by_slot[slot] += 1 / len(top)
                firsts.append(1 / len(top) if 0 in top else 0.0)
                picks.add(frozenset(order[slot] for slot in top))
            per_config = {c: mean(v) for c, v in centred.items()}
            for c, v in per_config.items():
                by_config.setdefault(c, []).append(v)
            slot0.append(mean(list(per_config.values())))
            first.append(mean(firsts))
            consistent += int(len(picks) == 1)
        identical, rate = mean(slot0), mean(first)
        out[kind] = {
            "identical": {
                "metric": identical,
                "leave_one_out": leave_one_out(slot0),
                "gate": slot0_min,
                "passed": identical >= slot0_min,
                "read": "single order",
                "by_config": {c: mean(v) for c, v in by_config.items()},
            },
            "first_slot": {
                "metric": rate,
                "leave_one_out": leave_one_out(first),
                "gate": FIRST_SLOT_MIN,
                "passed": rate >= FIRST_SLOT_MIN,
                "decisions": len(answers) * math.factorial(len(options)),
                "argmax_by_slot": by_slot,
                "consistent": consistent / len(answers),
            },
        }
    return out


def packed_consistency(
    engine,
    states: Sequence[Any],
    answers: Sequence[Mapping[str, Any]],
    lang: str = "en",
    kinds: Sequence[str] = KINDS,
) -> float:
    """Largest |p| difference between `ask`'s answers for `states` and each question asked
    alone, under the same order setting."""
    questions = probe_questions(lang, kinds)
    worst = 0.0
    for state, packed in zip(states, answers, strict=False):
        for qid, question in questions.items():
            if is_identical(qid):
                with single_order(engine):
                    alone = engine.system_one(state, {qid: question}).answers[qid]
            else:
                alone = engine.system_one(state, {qid: question}).answers[qid]
            for a, b in zip(
                slot_probs(packed[qid], question), slot_probs(alone, question), strict=True
            ):
                worst = max(worst, abs(a - b))
    return worst


def settings_mismatch(saved: dict, run: dict, keys: Sequence[str] = SETTINGS) -> list[str]:
    """Every setting among `keys` that `saved` recorded and this run does not share, as
    `<name> (report: X, run: Y)`. A setting the report never recorded is not checked."""
    return [
        f"{key} (report: {saved[key]!r}, run: {run.get(key)!r})"
        for key in keys
        if key in saved and saved[key] != run.get(key)
    ]


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
        "| kind | identical, one read (slot 0 - mean) | leave-one-out "
        "| first-slot rate | leave-one-out | consistent |",
        "|---|---|---|---|---|---|",
    ]
    for kind, r in report.items():
        ident, first = r["identical"], r["first_slot"]
        lines.append(
            f"| {kind} | {ident['metric']:+.3f} {'PASS' if ident['passed'] else 'FAIL'} "
            f"| {ident['leave_one_out'][0]:+.3f} .. {ident['leave_one_out'][1]:+.3f} "
            f"| {first['metric']:.3f} {'PASS' if first['passed'] else 'FAIL'} "
            f"| {first['leave_one_out'][0]:.3f} .. {first['leave_one_out'][1]:.3f} "
            f"| {first['consistent']:.3f} |"
        )
    return "\n".join(lines)


def read_states(path: str) -> list[Any]:
    """`state` from each line of a JSONL file (bench_en.jsonl, bench_ja.jsonl or your own)."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line)["state"] for line in fh if line.strip()]
