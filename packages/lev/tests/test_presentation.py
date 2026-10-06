"""Presentation checks (ADR-029), opt-in Score order averaging and the request row limit.

The checks run against `FakeEngine`: a real `DecisionEngine` without weights routes,
renders and counts every request, as serving does, and only the logits are scripted.
So a probe lev would reject, a row count that disagrees with the engine, or an answer
shape the checks misread fails here without weights.
"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from string import ascii_uppercase

import pytest
from lev import ChoiceAnswer, Score, ScoreAnswer, SystemOneRequest, SystemOneResponse, Usage
from lev.model import MAX_BATCH_TOKENS, DecisionEngine, EngineConfig, average_orders
from lev.presentation import (
    IDENTICAL_KS,
    KEYS,
    KINDS,
    MAX_ROWS,
    POST_LOAD,
    PRE_LOAD,
    SLOT0_MIN,
    TEXT,
    ask,
    call_groups,
    drift,
    packed_consistency,
    probe_questions,
    read_states,
    report,
    single_order,
    to_markdown,
)
from lev.router import route
from lev.types import Choice, Noul


def softmax(z: list[float]) -> list[float]:
    m = max(z)
    e = [math.exp(v - m) for v in z]
    return [v / sum(e) for v in e]


class Tokenizer:
    """Space-prefixed A-Z and 0-8 are single tokens, as `rich_tokenizer`."""

    single = {f" {c}" for c in ascii_uppercase} | {f" {i}" for i in range(9)}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [1] if text in self.single else [1] * (len(text) + 1)


class FakeEngine:
    """logit = content[text] + key_pull[key] + slot0 (in slot 0) + a packing term that
    grows with the questions per call, read in every order the engine renders and
    averaged as lev does. Routing, orders, row counts and the row limit are the real
    engine's."""

    def __init__(self, slot0=0.0, content=None, packing=0.0, key_pull=None, **config):
        self.real = DecisionEngine(model=None, tokenizer=Tokenizer(), config=EngineConfig(**config))
        self.slot0, self.content, self.packing = slot0, content or {}, packing
        self.key_pull = key_pull or {}
        self.calls = 0
        self.checkpoint = None

    @property
    def config(self) -> EngineConfig:
        return self.real.config

    @config.setter
    def config(self, value: EngineConfig) -> None:
        self.real.config = value

    def question_rows(self, questions):
        return self.real.question_rows(questions)

    def system_one(self, state, questions):
        prepared = self.real.prepare(state, questions)
        SystemOneRequest(state=state, questions=questions)
        self.calls += 1
        rows: dict[str, list] = {}
        for variant in prepared.variants:
            q = prepared.questions[variant.name]
            items = list(q.criteria.items()) if q.type == "choice" else list(enumerate(q.criteria))
            shown = items if variant.order is None else [items[i] for i in variant.order]
            z = [
                self.content.get(text, 0.0)
                + self.key_pull.get(key, 0.0)
                + (self.slot0 if j == 0 else 0.0)
                + self.packing * len(questions) * j
                for j, (key, text) in enumerate(shown)
            ]
            rows.setdefault(variant.name, []).append((softmax(z), variant.order))
        answers = {}
        for name, q in prepared.questions.items():
            p = average_orders([r[0] for r in rows[name]], [r[1] for r in rows[name]])
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


def run(engine, states, lang="en", kinds=KINDS, max_rows=MAX_ROWS, slot0_min=SLOT0_MIN):
    return report(ask(engine, states, lang, kinds, max_rows), lang, kinds, slot0_min)


class TestProbeQuestions:
    @pytest.mark.parametrize("lang", sorted(TEXT))
    def test_every_probe_is_a_valid_request(self, lang):
        questions = probe_questions(lang)
        assert SystemOneRequest(state="s", questions=questions)
        score = 2 * len(IDENTICAL_KS) + 6
        choice = 2 * sum(math.factorial(k) for k in IDENTICAL_KS) + 6
        assert len(questions) == score + choice

    @pytest.mark.parametrize("lang", sorted(TEXT))
    def test_identical_options_differ_only_by_position(self, lang):
        for qid, q in probe_questions(lang).items():
            if "|identical|" not in qid:
                continue
            texts = q["criteria"] if q["type"] == "score" else list(q["criteria"].values())
            assert len(set(texts)) == 1
            if q["type"] == "choice":
                assert set(q["criteria"]) <= set(KEYS)
                assert not any(c.isdigit() for key in q["criteria"] for c in key)

    def test_every_choice_key_sits_in_every_slot_equally_often(self):
        for k in IDENTICAL_KS:
            listings = [
                list(q["criteria"])
                for qid, q in probe_questions("en", ["choice"]).items()
                if qid.startswith(f"choice|identical|a team|{k}|")
            ]
            assert len(listings) == math.factorial(k)
            for slot in range(k):
                counts = {key: sum(lst[slot] == key for lst in listings) for key in KEYS[:k]}
                assert set(counts.values()) == {math.factorial(k - 1)}

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
        content = {"Soon": 3.0, "payments, refunds, invoices": 3.0}
        r = run(FakeEngine(content=content), ["a", "b"])
        for kind in KINDS:
            assert r[kind]["identical"]["metric"] == pytest.approx(0.0)
            assert r[kind]["identical"]["passed"]
            assert r[kind]["first_slot"]["metric"] == pytest.approx(1 / 3)
            assert r[kind]["first_slot"]["consistent"] == 1.0

    def test_a_planted_slot_zero_penalty_is_measured_exactly(self):
        r = run(FakeEngine(slot0=-2.0, order_average=False), ["a", "b", "c"])
        # centred slot 0 = -2 - (-2/K), averaged over K = 3, 4, 5 and both texts
        expected = sum(-2.0 * (1 - 1 / k) for k in IDENTICAL_KS) / len(IDENTICAL_KS)
        for kind in KINDS:
            assert r[kind]["identical"]["metric"] == pytest.approx(expected)
            assert not r[kind]["identical"]["passed"]
            assert r[kind]["first_slot"]["argmax_by_slot"][0] == 0
            assert r[kind]["first_slot"]["decisions"] == 18

    def test_ties_are_split_between_the_tied_slots(self):
        """A flat readout ties all three slots; argmax would credit slot 0 every time."""
        r = run(FakeEngine(), ["a", "b"])
        for kind in KINDS:
            assert r[kind]["first_slot"]["metric"] == pytest.approx(1 / 3)
            assert r[kind]["first_slot"]["argmax_by_slot"] == pytest.approx([4.0, 4.0, 4.0])

    def test_a_choice_key_pull_cancels_and_a_position_effect_does_not(self):
        pulled = run(FakeEngine(key_pull={KEYS[0]: 2.0}), ["a"], kinds=["choice"])
        assert pulled["choice"]["identical"]["metric"] == pytest.approx(0.0, abs=1e-12)
        both = run(FakeEngine(key_pull={KEYS[0]: 2.0}, slot0=-1.0, order_average=False), ["a"])
        alone = run(FakeEngine(slot0=-1.0, order_average=False), ["a"])
        assert both["choice"]["identical"]["metric"] == pytest.approx(
            alone["choice"]["identical"]["metric"]
        )

    @pytest.mark.parametrize("mode", ["reversed", "cyclic"])
    def test_the_identical_check_reads_one_order_whatever_the_engine_averages(self, mode):
        """Averaged, a pure slot bias cancels (cyclic) or hides the middle slots (reversed)."""
        single = run(FakeEngine(slot0=-2.0, order_average=False), ["a"])
        averaged = run(FakeEngine(slot0=-2.0, score_order_average=mode), ["a"])
        for kind in KINDS:
            assert averaged[kind]["identical"]["metric"] == pytest.approx(
                single[kind]["identical"]["metric"]
            )
        engine = FakeEngine(score_order_average=mode)
        with single_order(engine):
            assert engine.config.order_average is False
        assert engine.config.order_average and engine.config.score_order_average == mode

    def test_the_gate_is_the_proposal_and_can_be_moved(self):
        engine = FakeEngine(slot0=-0.3, order_average=False)
        assert not run(engine, ["a"])["score"]["identical"]["passed"]
        assert run(engine, ["a"], slot0_min=-1.0)["score"]["identical"]["passed"]
        assert SLOT0_MIN == -0.20


class TestCallGroups:
    """Calls hold at most `max_rows` rows, counted by the engine for every question."""

    @pytest.mark.parametrize("mode", ["off", "reversed", "cyclic"])
    def test_every_call_fits_the_limit_and_asks_everything_once(self, mode):
        engine = FakeEngine(score_order_average=mode)
        questions = probe_questions("en")
        groups = call_groups(engine, questions, 24)
        rows = engine.question_rows(questions)
        assert all(sum(rows[q] for q in g) <= 24 for g in groups)
        assert sorted(q for g in groups for q in g) == sorted(questions)

    def test_choice_rows_count(self):
        engine = FakeEngine()
        rows = engine.question_rows(probe_questions("en", ["choice"]))
        assert set(rows.values()) == {2}
        assert len(call_groups(engine, probe_questions("en", ["choice"]), 24)) > 1

    def test_the_grouping_does_not_change_an_isolated_readout(self):
        engine = FakeEngine(slot0=-1.0, content={"Soon": 1.0}, score_order_average="cyclic")
        assert run(engine, ["a", "b"], max_rows=5) == run(engine, ["a", "b"])

    def test_identical_and_permuted_questions_never_share_a_call(self):
        engine = FakeEngine()
        seen = []
        real = engine.system_one
        engine.system_one = lambda state, qs: seen.append(set(qs)) or real(state, qs)
        ask(engine, ["a"])
        for qids in seen:
            assert len({q.split("|")[1] for q in qids}) == 1


class TestPackedConsistency:
    def test_an_isolated_readout_agrees_with_itself_alone(self):
        engine = FakeEngine(slot0=-1.0)
        answers = ask(engine, ["a"])
        assert packed_consistency(engine, ["a"], answers) == pytest.approx(0.0)

    def test_a_readout_that_depends_on_packing_is_caught(self):
        engine = FakeEngine(packing=0.01)
        assert packed_consistency(engine, ["a"], ask(engine, ["a"])) > 0.01

    def test_the_packed_answers_are_reused_not_asked_again(self):
        engine = FakeEngine()
        answers = ask(engine, ["a", "b"])
        before = engine.calls
        packed_consistency(engine, ["a"], answers)
        assert engine.calls - before == len(probe_questions("en"))


class TestReport:
    def test_drift_flags_only_what_moved_past_the_tolerance(self):
        now = run(FakeEngine(slot0=-2.0), ["a"])
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
        text = to_markdown(run(FakeEngine(), ["a"]))
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

    @pytest.mark.parametrize("mode", ["off", "reversed", "cyclic"])
    def test_order_average_off_reads_every_question_once(self, rich_tokenizer, mode):
        for q in (
            Score(criteria=["low", "mid", "high"]),
            Choice(criteria={"a": None, "b": None, "c": None}),
        ):
            assert self.orders(
                q, rich_tokenizer, score_order_average=mode, order_average=False
            ) == [None]

    def test_choice_does_not_follow_the_score_switch(self, rich_tokenizer):
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
        biased = softmax([2.0, 0.0, 0.0, 0.0])
        assert average_orders([biased] * k, orders) == pytest.approx([1 / k] * k)


QUESTIONS = {
    "team": Choice(criteria={"a": None, "b": None, "c": None}),
    "urgency": Score(criteria=["low", "mid", "high", "blocked"]),
    "churn": Noul(instructions="leaving?"),
    "many": Choice(criteria={f"o{i}": None for i in range(40)}),
}


class TestQuestionRows:
    """`question_rows` is what `prepare` renders, so callers never recount rows."""

    @pytest.mark.parametrize("mode", ["off", "reversed", "cyclic"])
    @pytest.mark.parametrize("order_average", [True, False])
    @pytest.mark.parametrize("noul_readout", ["rating", "binary"])
    def test_it_matches_prepare(self, poor_tokenizer, mode, order_average, noul_readout):
        engine = DecisionEngine(
            model=None,
            tokenizer=poor_tokenizer,
            config=EngineConfig(
                score_order_average=mode,
                order_average=order_average,
                noul_readout=noul_readout,
            ),
            mode_b_head=object(),
        )
        rows = engine.question_rows(QUESTIONS)
        assert sum(rows.values()) == engine.prepare("s", QUESTIONS).rows
        assert rows["many"] == 1  # Mode B: one row whatever the averaging

    def test_schema_first_is_one_row_per_question(self, rich_tokenizer):
        from lev.prompt import Layout

        engine = DecisionEngine(
            model=None,
            tokenizer=rich_tokenizer,
            config=EngineConfig(layout=Layout.SCHEMA_FIRST, score_order_average="cyclic"),
            mode_b_head=object(),
        )
        assert set(engine.question_rows(QUESTIONS).values()) == {1}
        assert engine.prepare("s", QUESTIONS).rows == len(QUESTIONS)


BIG = {
    **{f"c{i}": Choice(criteria={"a": None, "b": None, "c": None, "d": None}) for i in range(20)},
    **{f"s{i}": Score(criteria=["low", "mid", "high"]) for i in range(40)},
}


LONG = "x" * 6000


class TestAveragedTokenLimit:
    """Only a request Score averaging grew can be refused, and only past the batch budget."""

    def engine(self, tokenizer, **config) -> DecisionEngine:
        return DecisionEngine(model=None, tokenizer=tokenizer, config=EngineConfig(**config))

    def test_a_long_state_with_few_averaged_rows_is_refused(self, rich_tokenizer):
        question = {"urgency": Score(criteria=["low", "mid", "high"])}
        engine = self.engine(rich_tokenizer, score_order_average="cyclic")
        with pytest.raises(
            ValueError, match="2 of the rows from score_order_average='cyclic', above"
        ):
            engine.prepare(LONG, question)
        assert engine.prepare("short", question).rows == 3
        assert self.engine(rich_tokenizer).prepare(LONG, question).rows == 1

    def test_without_score_averaging_a_large_request_is_never_refused(self, rich_tokenizer):
        """#4's batcher runs an oversized request alone and unsplit, so main accepts it."""
        prepared = self.engine(rich_tokenizer).prepare("x" * 400, BIG)
        assert prepared.rows == 80 and prepared.rows * prepared.width > MAX_BATCH_TOKENS
        assert [v.order for v in prepared.variants] == [None, [3, 2, 1, 0]] * 20 + [None] * 40

    def test_only_a_request_score_averaging_grew_can_be_refused(self, rich_tokenizer):
        engine = self.engine(rich_tokenizer, score_order_average="cyclic")
        with pytest.raises(ValueError, match="80 of the rows from"):
            engine.prepare("x" * 400, BIG)
        choices = {k: q for k, q in BIG.items() if k.startswith("c")}
        assert engine.prepare("x" * 400, choices).rows == 40

    def test_none_is_no_limit_and_the_default_is_the_batchers(self, rich_tokenizer):
        from lev.batcher import Batcher

        question = {"urgency": Score(criteria=["low", "mid", "high"])}
        engine = self.engine(rich_tokenizer, score_order_average="cyclic", max_batch_tokens=None)
        assert engine.prepare(LONG, question).rows == 3
        default = inspect.signature(Batcher).parameters["max_batch_tokens"].default
        assert EngineConfig().max_batch_tokens == MAX_BATCH_TOKENS == default == 16384

    def serve(self, monkeypatch, tokenizer, **create):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient
        from lev import server

        seen = {}

        def load(*a, **kwargs):
            seen["engine"] = self.engine(
                tokenizer, score_order_average="cyclic", max_batch_tokens=kwargs["max_batch_tokens"]
            )
            return seen["engine"]

        def batcher(engine, **kwargs):
            seen["batcher"] = kwargs["max_batch_tokens"]
            return real_batcher(engine, **kwargs)

        real_batcher = server.Batcher
        monkeypatch.setattr(server, "load", load)
        monkeypatch.setattr(server, "Batcher", batcher)
        return TestClient(server.create_app(**create)), seen

    def test_the_server_gives_its_budget_to_the_engine(self, monkeypatch, rich_tokenizer):
        client, seen = self.serve(monkeypatch, rich_tokenizer, max_batch_tokens=1234)
        with client:
            health = client.get("/health").json()
        assert seen["batcher"] == seen["engine"].config.max_batch_tokens == 1234
        assert health["max_batch_tokens"] == 1234

    def test_the_server_answers_422(self, monkeypatch, rich_tokenizer):
        client, _ = self.serve(monkeypatch, rich_tokenizer, max_batch_tokens=500)
        questions = {f"q{i}": {"type": "score", "criteria": ["a", "b", "c"]} for i in range(2)}
        with client:
            response = client.post(
                "/v1/systemone", json={"state": "x" * 200, "questions": questions}
            )
        assert response.status_code == 422
        assert "max_batch_tokens=500" in response.json()["detail"]


class TestCommittedReport:
    """`data/presentation-checks.json` is what `--compare` reads in CI."""

    def test_it_parses_and_carries_every_metric(self):

        path = Path(__file__).resolve().parents[3] / "data" / "presentation-checks.json"
        committed = json.loads(path.read_text("utf-8"))
        assert committed["score_order_average"] == "off" and committed["n_states"] == 290
        assert committed["max_rows"] == MAX_ROWS
        for kind in KINDS:
            assert committed["checks"][kind]["identical"]["read"] == "single order"
            for check in ("identical", "first_slot"):
                assert isinstance(committed["checks"][kind][check]["metric"], float)
        assert drift(committed["checks"], committed["checks"], 0.0) == []


class TestCompareGuardsItsSettings:
    """`--compare` compares like with like: settings first (exit 2), then metrics (exit 1)."""

    def cli(self, monkeypatch, tmp_path, engine, *extra):
        from lev.cli import main

        monkeypatch.setattr("lev.model.load", lambda *a, **k: engine)
        monkeypatch.setattr("lev.cli._cuda", lambda: None)
        states = tmp_path / "states.jsonl"
        states.write_text('{"state": "a"}\n{"state": "b"}\n', "utf-8")
        main(["presentation-checks", str(states), "--packed", "0", *extra])

    def committed(self, monkeypatch, tmp_path, engine, **edit) -> str:
        out = tmp_path / "report.json"
        self.cli(monkeypatch, tmp_path, engine, "--out", str(out))
        saved = json.loads(out.read_text("utf-8")) | edit
        path = tmp_path / "committed.json"
        path.write_text(json.dumps(saved), "utf-8")
        return str(path)

    def test_the_weights_dtype_comes_from_the_engine_model(self):
        torch = pytest.importorskip("torch")
        from types import SimpleNamespace

        from lev.cli import _weights_dtype

        model = torch.nn.Linear(2, 2).to(torch.bfloat16)
        assert _weights_dtype(SimpleNamespace(model=model)) == "torch.bfloat16"
        assert _weights_dtype(SimpleNamespace(model=None)) is None

    def test_the_settings_split_at_the_load(self):
        assert set(PRE_LOAD).isdisjoint(POST_LOAD)
        assert set(POST_LOAD) == {"checkpoint_revision", "dtype"}

    def test_matching_settings_are_compared_metric_by_metric(self, monkeypatch, tmp_path):
        engine = FakeEngine(slot0=-1.0)
        path = Path(self.committed(monkeypatch, tmp_path, engine))
        saved = json.loads(path.read_text("utf-8"))
        assert saved["score_order_average"] == "off" and saved["max_rows"] == MAX_ROWS
        saved["checks"]["score"]["identical"]["metric"] += 0.5
        path.write_text(json.dumps(saved), "utf-8")
        with pytest.raises(SystemExit) as exc:
            self.cli(monkeypatch, tmp_path, engine, "--compare", str(path))
        assert "score.identical" in str(exc.value.code)

    def test_a_model_free_setting_is_refused_before_the_load(self, monkeypatch, tmp_path, capsys):
        path = self.committed(monkeypatch, tmp_path, FakeEngine(), score_order_average="reversed")

        def no_load(*a, **k):
            raise AssertionError("loaded a model for a run it had to refuse")

        monkeypatch.setattr("lev.model.load", no_load)
        from lev.cli import main

        states = tmp_path / "states.jsonl"
        with pytest.raises(SystemExit) as exc:
            main(["presentation-checks", str(states), "--packed", "0", "--compare", path])
        assert exc.value.code == 2
        assert (
            "presentation settings differ from the committed report: "
            "score_order_average (report: 'reversed', run: 'off')"
        ) in capsys.readouterr().err

    def test_another_revision_is_refused_after_the_load_and_before_any_run(
        self, monkeypatch, tmp_path, capsys
    ):
        engine = FakeEngine()
        path = self.committed(monkeypatch, tmp_path, engine, checkpoint_revision="abc123")
        calls = engine.calls
        with pytest.raises(SystemExit) as exc:
            self.cli(monkeypatch, tmp_path, engine, "--compare", str(path))
        assert exc.value.code == 2
        assert engine.calls == calls
        assert "checkpoint_revision (report: 'abc123', run: None)" in capsys.readouterr().err
