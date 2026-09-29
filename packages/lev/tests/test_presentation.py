"""Presentation checks (ADR-029) and opt-in Score order averaging.

The checks run against a fake `system_one` that validates every request with
lev's own `SystemOneRequest` and answers with lev's answer types, so a probe lev
would reject, or an answer shape the checks misread, fails here without weights.
"""

from __future__ import annotations

import json
import math

import pytest
from lev import (
    ChoiceAnswer,
    Score,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
)
from lev.model import DecisionEngine, EngineConfig, average_orders
from lev.presentation import (
    IDENTICAL_KS,
    KINDS,
    SLOT0_MIN,
    TEXT,
    drift,
    packed_consistency,
    probe_questions,
    read_states,
    run,
    to_markdown,
)
from lev.router import route
from lev.types import Choice


def softmax(z: list[float]) -> list[float]:
    m = max(z)
    e = [math.exp(v - m) for v in z]
    return [v / sum(e) for v in e]


class FakeEngine:
    """logit = content[option text] + slot_bias[slot]. `packing` adds a term that
    depends on how many questions share the call, which a packed readout must not."""

    def __init__(self, slot0: float = 0.0, content: dict | None = None, packing: float = 0.0):
        self.slot0, self.content, self.packing, self.calls = slot0, content or {}, packing, 0

    def system_one(self, state, questions):
        request = SystemOneRequest(state=state, questions=questions)
        self.calls += 1
        answers = {}
        for name, q in request.questions.items():
            texts = list(q.criteria)
            z = [
                self.content.get(t, 0.0)
                + (self.slot0 if i == 0 else 0.0)
                + self.packing * len(request.questions) * i
                for i, t in enumerate(texts)
            ]
            p = softmax(z)
            if q.type == "score":
                answers[name] = ScoreAnswer(
                    score=0.0,
                    probabilities=dict(enumerate(p)),
                    legend=dict(enumerate(q.criteria)),
                    confidence=max(p),
                )
            else:
                keys = list(q.criteria)
                answers[name] = ChoiceAnswer(
                    choice=keys[p.index(max(p))],
                    probabilities=dict(zip(keys, p, strict=True)),
                    confidence=max(p),
                )
        return SystemOneResponse(model="fake", answers=answers, usage=Usage(input_tokens=1))


class TestProbeQuestions:
    @pytest.mark.parametrize("lang", sorted(TEXT))
    def test_every_probe_is_a_valid_request(self, lang):
        questions = probe_questions(lang)
        assert SystemOneRequest(state="s", questions=questions)
        per_kind = 2 * len(IDENTICAL_KS) + 6
        assert len(questions) == per_kind * len(KINDS)

    @pytest.mark.parametrize("lang", sorted(TEXT))
    def test_identical_options_differ_only_by_position(self, lang):
        for qid, q in probe_questions(lang).items():
            if "|identical|" not in qid:
                continue
            texts = q["criteria"] if q["type"] == "score" else list(q["criteria"].values())
            assert len(set(texts)) == 1
            if q["type"] == "choice":
                assert list(q["criteria"]) == [str(i + 1) for i in range(len(q["criteria"]))]

    @pytest.mark.parametrize("lang", sorted(TEXT))
    def test_every_option_sits_in_every_slot_twice(self, lang):
        for kind in KINDS:
            perms = [
                list(q["criteria"])
                for qid, q in probe_questions(lang, [kind]).items()
                if "|perm|" in qid
            ]
            options = list(TEXT[lang][kind]["options"])
            for slot in range(3):
                assert sorted(p[slot] for p in perms) == sorted(options * 2)


class TestChecks:
    def test_a_flat_readout_sits_at_zero_and_one_third(self):
        content = {"Soon": 3.0, "Technical": 3.0}
        report = run(FakeEngine(content=content).system_one, ["a", "b"])
        for kind in KINDS:
            assert report[kind]["identical"]["metric"] == pytest.approx(0.0)
            assert report[kind]["identical"]["passed"]
            assert report[kind]["first_slot"]["metric"] == pytest.approx(1 / 3)

    def test_a_planted_slot_zero_penalty_is_measured_exactly(self):
        report = run(FakeEngine(slot0=-2.0).system_one, ["a", "b", "c"])
        # centred slot 0 = -2 - (-2/K), averaged over K = 3, 4, 5 and both texts
        expected = sum(-2.0 * (1 - 1 / k) for k in IDENTICAL_KS) / len(IDENTICAL_KS)
        for kind in KINDS:
            assert report[kind]["identical"]["metric"] == pytest.approx(expected)
            assert not report[kind]["identical"]["passed"]
            assert report[kind]["first_slot"]["argmax_by_slot"][0] == 0
            assert report[kind]["first_slot"]["decisions"] == 18

    def test_one_call_per_state(self):
        engine = FakeEngine()
        run(engine.system_one, ["a", "b", "c"])
        assert engine.calls == 3

    def test_the_gate_is_the_proposal_and_can_be_moved(self):
        engine = FakeEngine(slot0=-0.3)
        assert not run(engine.system_one, ["a"])["score"]["identical"]["passed"]
        assert run(engine.system_one, ["a"], slot0_min=-1.0)["score"]["identical"]["passed"]
        assert SLOT0_MIN == -0.20


class TestPackedConsistency:
    def test_an_isolated_readout_agrees_with_itself_alone(self):
        assert packed_consistency(FakeEngine(slot0=-1.0).system_one, ["a"]) == pytest.approx(0.0)

    def test_a_readout_that_depends_on_packing_is_caught(self):
        assert packed_consistency(FakeEngine(packing=0.01).system_one, ["a"]) > 0.01


class TestReport:
    def test_drift_flags_only_what_moved_past_the_tolerance(self):
        now = run(FakeEngine(slot0=-2.0).system_one, ["a"])
        committed = json.loads(json.dumps(now))
        assert drift(now, committed, 0.05) == []
        committed["score"]["identical"]["metric"] += 0.2
        assert drift(now, committed, 0.05) == [
            f"score.identical: {committed['score']['identical']['metric']:+.3f}"
            f" -> {now['score']['identical']['metric']:+.3f}"
        ]
        del now["choice"]
        assert "choice.identical: missing from this run" in drift(now, committed, 0.05)

    def test_markdown_has_a_row_per_kind(self):
        text = to_markdown(run(FakeEngine().system_one, ["a"]))
        assert text.count("\n") == 1 + len(KINDS)

    def test_states_are_read_from_jsonl(self, tmp_path):
        path = tmp_path / "s.jsonl"
        path.write_text('{"state": "one"}\n\n{"state": "two", "x": 1}\n', "utf-8")
        assert read_states(str(path)) == ["one", "two"]

    def test_the_cli_is_registered(self):
        from lev.cli import main

        with pytest.raises(SystemExit) as exc:
            main(["presentation-checks", "--help"])
        assert exc.value.code == 0


class TestScoreOrderAverage:
    """Off by default: a Score keeps its level order unless the config opts in."""

    def orders(self, question, tokenizer, **config):
        engine = DecisionEngine(model=None, tokenizer=tokenizer, config=EngineConfig(**config))
        return engine._orders(question, route(question, tokenizer))

    def test_off_by_default(self, rich_tokenizer):
        assert EngineConfig().score_order_average == "off"
        assert self.orders(Score(criteria=["low", "mid", "high"]), rich_tokenizer) == [None]

    def test_reversed_is_two_rows(self, rich_tokenizer):
        q = Score(criteria=["low", "mid", "high"])
        assert self.orders(q, rich_tokenizer, score_order_average="reversed") == [None, [2, 1, 0]]

    def test_cyclic_is_every_rotation(self, rich_tokenizer):
        q = Score(criteria=["low", "mid", "high"])
        assert self.orders(q, rich_tokenizer, score_order_average="cyclic") == [
            None,
            [1, 2, 0],
            [2, 0, 1],
        ]

    def test_independent_of_the_choice_switch(self, rich_tokenizer):
        q = Score(criteria=["low", "high"])
        assert self.orders(
            q, rich_tokenizer, score_order_average="reversed", order_average=False
        ) == [None, [1, 0]]
        c = Choice(criteria={"a": None, "b": None, "c": None})
        for mode in ("off", "reversed", "cyclic"):
            assert self.orders(c, rich_tokenizer, score_order_average=mode) == [None, [2, 1, 0]]

    def test_schema_first_never_doubles_the_prefix(self, rich_tokenizer):
        from lev.prompt import Layout

        q = Score(criteria=["low", "high"])
        assert self.orders(
            q, rich_tokenizer, score_order_average="cyclic", layout=Layout.SCHEMA_FIRST
        ) == [None]

    def test_cyclic_averaging_cancels_a_slot_bias_on_equal_content(self):
        """Each level sits in each slot once, so a pure position bias averages out."""
        k = 4
        orders = [None] + [[(j + r) % k for j in range(k)] for r in range(1, k)]
        biased = softmax([2.0, 0.0, 0.0, 0.0])  # slot 0 favoured, content equal
        assert average_orders([biased] * k, orders) == pytest.approx([1 / k] * k)


class TestCallGroups:
    """How the probe is split into calls: one by default, bounded Score rows under cyclic."""

    def rows(self, group, mode):
        from lev.presentation import score_rows

        return sum(score_rows(q, mode) for q in group.values() if q["type"] == "score")

    def test_off_and_reversed_stay_one_call(self):
        from lev.presentation import call_groups

        questions = probe_questions("en", ["score"])
        assert len(call_groups(questions)) == 1
        assert self.rows(questions, "reversed") == 24
        assert len(call_groups(questions, score_order_average="reversed")) == 1

    def test_cyclic_is_cut_at_the_row_limit_and_asks_everything_once(self):
        from lev.presentation import call_groups

        questions = probe_questions("en")
        assert self.rows(questions, "cyclic") == 42
        groups = call_groups(questions, score_order_average="cyclic")
        assert len(groups) > 1
        assert all(self.rows(g, "cyclic") <= 24 for g in groups)
        asked = [qid for g in groups for qid in g]
        assert sorted(asked) == sorted(questions)

    def test_split_checks_never_mixes_the_two_checks(self):
        from lev.presentation import call_groups

        for group in call_groups(probe_questions("ja"), split_checks=True):
            assert len({qid.split("|")[1] for qid in group}) == 1

    def test_the_split_does_not_change_an_isolated_readout(self):
        engine = FakeEngine(slot0=-1.0, content={"Soon": 1.0})
        one = run(engine.system_one, ["a", "b"])
        split = run(
            engine.system_one,
            ["a", "b"],
            split_checks=True,
            max_score_rows=5,
            score_order_average="cyclic",
        )
        assert split == one
        assert packed_consistency(engine.system_one, ["a"], split_checks=True) == pytest.approx(0.0)


class TestCommittedReport:
    """`data/presentation-checks.json` is what `--compare` reads in CI."""

    def test_it_parses_and_carries_every_metric(self):
        from pathlib import Path

        path = Path(__file__).resolve().parents[3] / "data" / "presentation-checks.json"
        committed = json.loads(path.read_text("utf-8"))
        assert committed["score_order_average"] == "off" and committed["n_states"] == 290
        for kind in KINDS:
            for check in ("identical", "first_slot"):
                assert isinstance(committed["checks"][kind][check]["metric"], float)
        assert drift(committed["checks"], committed["checks"], 0.0) == []
