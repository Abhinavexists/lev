"""Label-free checks for option-position bias and packing consistency (ADR-029).

Identical-option probes use one order; real-option probes use configured averaging.
Scores use returned probabilities: calibration scales the identical metric by 1/T.
The -0.20 gate comes from Laya's raw-score checks (NandhaKishorM/laya#259)."""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

KINDS = ("score", "choice")
IDENTICAL_KS = (3, 4, 5)
KEYS = ("■", "●", "◆", "▲", "◎")
SLOT0_MIN = -0.20
FIRST_SLOT_MIN = 0.15
# Limit probe rows to bound GPU memory use.
MAX_ROWS = 32

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

# Check pre-load settings before spending time loading weights.
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
    """Build identical-option and permutation probes with structured question ids."""
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
def configured(engine, **changes) -> Generator[None, None, None]:
    """Run the block with `engine.config` changed as given, then restore it."""
    config = engine.config
    engine.config = replace(config, **changes)
    try:
        yield
    finally:
        engine.config = config


def single_order(engine):
    return configured(engine, order_average=False)


def call_groups(engine, questions: dict[str, dict], max_rows: int) -> list[dict[str, dict]]:
    """Group questions into calls within the engine's rendered-row budget."""
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
    """Run single-order identical probes and configured permutation probes within max_rows."""
    max_rows = max_rows or MAX_ROWS
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
        # Probes enforce their own row budget instead of the serving token limit.
        with configured(engine, max_batch_tokens=None):
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
    """Return the largest probability difference between packed and individual probes."""
    questions = probe_questions(lang, kinds)
    worst = 0.0
    for state, packed in zip(states, answers, strict=False):
        for qid, question in questions.items():
            if is_identical(qid):
                with single_order(engine):
                    alone = engine.system_one(state, {qid: question}).answers[qid]
            else:
                with configured(engine, max_batch_tokens=None):
                    alone = engine.system_one(state, {qid: question}).answers[qid]
            for a, b in zip(
                slot_probs(packed[qid], question), slot_probs(alone, question), strict=True
            ):
                worst = max(worst, abs(a - b))
    return worst


def settings_mismatch(saved: dict, run: dict, keys: Sequence[str] = SETTINGS) -> list[str]:
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
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line)["state"] for line in fh if line.strip()]
