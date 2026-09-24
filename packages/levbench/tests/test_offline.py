"""End-to-end plumbing check against the fake transport. No API key needed."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = next(p for p in PKG.parents if (p / "data").is_dir())
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(PKG / "tests"))

import pytest  # noqa: E402
from fake_transport import transport  # noqa: E402
from levbench import batching, confidence_id, metrics, runner  # noqa: E402
from levbench.tasks import dataset  # noqa: E402
from typesafe_sdk import TypeSafeClient  # noqa: E402


def fake_client() -> TypeSafeClient:
    return TypeSafeClient(api_key="offline-test", transport=transport())


def test_metrics_are_arithmetically_right() -> None:
    """Check the metric maths against values computed by hand."""
    import math

    dist = {"a": 0.7, "b": 0.2, "c": 0.1}
    assert abs(metrics.log_loss(dist, "a") - (-math.log(0.7))) < 1e-12
    # Brier: (0.7-1)^2 + 0.2^2 + 0.1^2 = 0.09 + 0.04 + 0.01
    assert abs(metrics.brier(dist, "a") - 0.14) < 1e-12

    # A perfectly calibrated set: 70% confident, 70% accurate -> ECE 0.
    recs = [({True: 0.7, False: 0.3}, True, True, 0.7)] * 7
    recs += [({True: 0.7, False: 0.3}, True, False, 0.7)] * 3
    cal = metrics.calibration(recs)
    assert abs(cal.accuracy - 0.7) < 1e-12
    assert cal.ece < 1e-12, f"expected zero ECE, got {cal.ece}"

    # A badly overconfident set: 100% confident, 50% accurate -> ECE 0.5.
    bad = [({True: 1.0, False: 0.0}, True, True, 1.0)] * 5
    bad += [({True: 1.0, False: 0.0}, True, False, 1.0)] * 5
    assert abs(metrics.calibration(bad).ece - 0.5) < 1e-12


def test_every_primitive_flattens_to_a_distribution() -> None:
    """Every primitive must flatten, including the fieldless Noul."""
    items, questions = dataset()
    result = runner.call_once(fake_client(), items[0].state, questions)

    noul = result.answers["is_urgent"]
    assert not hasattr(noul, "confidence") or noul.confidence is None
    dist = metrics.to_distribution(noul)
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert metrics.top_probability(noul) == max(dist.values())

    choice = result.answers["department"]
    assert metrics.predicted_label(choice) == choice.choice
    assert abs(sum(metrics.to_distribution(choice).values()) - 1.0) < 1e-3

    score = result.answers["frustration"]
    by_level = metrics.to_distribution(score)
    expected = sum(level * p for level, p in by_level.items())
    assert abs(score.score - expected) < 1e-2, "score must equal the probability-weighted mean"


def test_ece_bins_on_the_top_probability_not_the_vendor_confidence_field() -> None:
    """lev's `confidence` is Gini concentration and Jev's is chance-corrected max
    probability, so ECE bins on the answer's own top probability instead."""
    from types import SimpleNamespace

    from levbench.runner import run_eval
    from levbench.tasks import FileItem
    from typesafe_sdk import Choice

    class Server:
        def system_one(self, state, questions):
            # 90% on the right answer, but a vendor confidence of 0.2.
            answer = SimpleNamespace(
                type="choice", choice="a", probabilities={"a": 0.9, "b": 0.1}, confidence=0.2
            )
            return SimpleNamespace(
                answers={"q": answer},
                usage=SimpleNamespace(input_tokens=1, output_tokens=0),
                model="stub",
            )

    items = [FileItem(f"s{i}", {"q": "a" if i < 9 else "b"}) for i in range(10)]
    questions = {"q": Choice(instructions="?", criteria={"a": None, "b": None})}
    report = run_eval(Server(), "lev", "stub", items, questions)
    # Every answer is 0.9 confident and 9 of 10 are right: perfectly calibrated.
    assert report.per_question["q"].ece == pytest.approx(0.0, abs=1e-9)


def test_eval_runs_end_to_end_over_the_built_in_fixture() -> None:
    items, questions = dataset()
    report = runner.run_eval(fake_client(), "jev", "jev-latest", items, questions)
    assert len(report.calls) == len(items)
    assert set(report.per_question) == set(questions)
    assert report.total_cost > 0
    text = runner.format_report(report)
    assert "accuracy" in text and "ECE" in text


def test_sweep_arithmetic_over_a_fixed_billing_shape() -> None:
    """SweepRow arithmetic over the fake transport's per-request billing, not
    Jev's real billing, which only a live `levbench sweep` can measure."""
    state = (ROOT / "data" / "sample_policy.md").read_text()
    rows = batching.sweep(fake_client(), "jev-latest", state, [1, 2, 4, 8, 13])

    assert rows[0].cost_ratio == 1.0 or abs(rows[0].cost_ratio - 1.0) < 0.01, (
        "at N=1 batched and split are the same call; ratio must be ~1"
    )
    ratios = [r.cost_ratio for r in rows]
    assert ratios == sorted(ratios), f"saving must grow with N, got {ratios}"
    assert rows[-1].cost_ratio > 5, (
        f"at N=13 expected a large saving, got {rows[-1].cost_ratio:.1f}x"
    )
    assert "batching sweep" in batching.format_sweep(rows, "jev-latest", len(state))


def test_confidence_identifier_recovers_a_planted_formula() -> None:
    """Known-answer test: plant each formula, demand it be identified uniquely.

    If planted formulas were ambiguous, a match against a real server would
    prove nothing.
    """
    import json

    import httpx2
    from typesafe_sdk import Choice, TypeSafeClient

    def server(conf_fn):
        def handle(request):
            body = json.loads(request.content)
            answers = {}
            for key, question in body["questions"].items():
                options = list(question["criteria"])
                # Varied, non-uniform distributions, so candidate statistics
                # actually separate instead of coinciding.
                raw = [1 + hashlib.sha256(f"{key}:{o}".encode()).digest()[0] % 97 for o in options]
                total = sum(raw)
                probs = {o: r / total for o, r in zip(options, raw, strict=True)}
                answers[key] = {
                    "type": "choice",
                    "choice": max(probs, key=probs.get),
                    "probabilities": probs,
                    "confidence": conf_fn(list(probs.values())),
                }
            return httpx2.Response(
                200,
                json={
                    "model": "planted",
                    "answers": answers,
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            )

        return TypeSafeClient(api_key="x", model="planted", transport=httpx2.MockTransport(handle))

    questions = {
        f"q{i}": Choice(instructions="pick", criteria={c: c for c in "abcde"[: 2 + i % 4]})
        for i in range(6)
    }

    for planted, fn in confidence_id.CANDIDATES.items():
        client = server(fn)
        answers = []
        for state in ("alpha", "beta", "gamma", "delta"):
            answers.extend(runner.call_once(client, state, questions).answers.values())
        fits = confidence_id.identify(confidence_id.collect(answers))
        winners = [f.name for f in fits if f.matches]
        assert winners == [planted], f"planted {planted}, identified {winners}"


def test_base_url_routes_to_a_local_clone() -> None:
    """`--base-url` must actually change where requests go."""
    import httpx2
    from typesafe_sdk import TypeSafeClient

    seen: dict[str, str] = {}

    def handle(request):
        seen["url"] = str(request.url)
        return httpx2.Response(
            200,
            json={
                "model": "litjev",
                "answers": {"q": {"type": "noul", "noul": 0.5}},
                "usage": {"input_tokens": 5, "output_tokens": 0},
            },
        )

    from typesafe_sdk import Noul

    client = TypeSafeClient(
        api_key="local",
        model="litjev",
        base_url="http://127.0.0.1:8000",
        transport=httpx2.MockTransport(handle),
    )
    runner.call_once(client, "hi", {"q": Noul(instructions="test")})
    assert seen["url"] == "http://127.0.0.1:8000/v1/systemone", seen["url"]


def test_local_base_url_never_forwards_the_real_key(monkeypatch) -> None:
    """A `--base-url` server must not receive TYPESAFE_API_KEY (the SDK always
    sends `Authorization: Bearer <key>`); the hosted path still authenticates."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-REAL-SECRET-KEY")
    monkeypatch.delenv("LEVBENCH_LOCAL_API_KEY", raising=False)
    local, _ = runner.build_client("jev", "litjev", "http://127.0.0.1:8000")
    assert "REAL-SECRET" not in str(local._config.api_key), "leaked the real key"
    assert local._config.api_key == "local"

    hosted, _ = runner.build_client("jev", "jev-latest")
    assert hosted._config.api_key == "sk-REAL-SECRET-KEY", "hosted path lost its key"


def test_empty_local_key_env_var_falls_back(monkeypatch) -> None:
    """`LEVBENCH_LOCAL_API_KEY=` (as copied from `.env.example`) must fall back.

    `os.environ.get(var, "local")` returns "" there, and the SDK then sends a
    malformed `Authorization: Bearer ` header (`LocalProtocolError: Illegal
    header value`).
    """
    monkeypatch.setenv("LEVBENCH_LOCAL_API_KEY", "")
    client, _ = runner.build_client("lev")
    assert client._config.api_key == "local", (
        f"empty env var leaked through as {client._config.api_key!r}"
    )


def _write_task_file(path, question_type="choice", n=5):
    import json

    question = {
        "choice": {
            "type": "choice",
            "instructions": "which team",
            "criteria": {"a": None, "b": None},
        },
        "score": {"type": "score", "instructions": "how much", "criteria": ["low", "high"]},
        "noul": {"type": "noul", "instructions": "is it urgent"},
    }[question_type]
    truth = {"choice": "a", "score": 0, "noul": True}[question_type]
    path.write_text(
        json.dumps(
            {
                "questions": {"q": question},
                "items": [{"state": f"state {i}", "labels": {"q": truth}} for i in range(n)],
            }
        )
    )
    return path


def test_eval_runs_over_a_generated_task_file(tmp_path) -> None:
    """The lev -> levbench seam: the harness reads back and scores the kind of
    file `lev data eval` writes."""
    from levbench.tasks import dataset as load

    items, questions = load(_write_task_file(tmp_path / "gen.json"))
    report = runner.run_eval(fake_client(), "jev", "jev-latest", items, questions)
    assert len(report.calls) == 5
    assert set(report.per_question) == {"q"}
    assert "accuracy" in runner.format_report(report)


@pytest.mark.parametrize("question_type", ["choice", "score", "noul"])
def test_every_primitive_round_trips_through_a_task_file(tmp_path, question_type) -> None:
    from levbench.tasks import load_task_file

    items, questions = load_task_file(_write_task_file(tmp_path / "g.json", question_type))
    assert questions["q"].type == question_type
    report = runner.run_eval(fake_client(), "jev", "jev-latest", items, questions)
    assert report.per_question["q"] is not None


def test_a_task_file_labelling_an_undefined_question_raises(tmp_path) -> None:
    """Otherwise the label is silently dropped and the eval scores fewer items
    than it was given."""
    import json

    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "questions": {"q": {"type": "noul", "instructions": "x"}},
                "items": [{"state": "s", "labels": {"q": True, "ghost": False}}],
            }
        )
    )
    from levbench.tasks import load_task_file

    with pytest.raises(ValueError, match="does not define"):
        load_task_file(path)


def test_a_missing_task_file_says_how_to_make_one(tmp_path) -> None:
    from levbench.tasks import load_task_file

    with pytest.raises(FileNotFoundError, match="lev data eval"):
        load_task_file(tmp_path / "nope.json")


def test_detectable_difference_shrinks_with_n() -> None:
    """The number that says whether an accuracy delta means anything."""
    from levbench.tasks import detectable_difference

    assert detectable_difference(24) > 0.15, "24 items cannot resolve a 5-point gain"
    assert detectable_difference(2000) < 0.02
    assert detectable_difference(0) == 1.0


def test_pooling_can_hide_a_formula_that_holds_per_candidate_set_size() -> None:
    """Two formulas, one per size, look like "no match" when pooled.

    This is the shape the live Jev run produced: nothing matched overall, and
    the pooled view could not say which samples broke it.
    """
    four = [0.7, 0.1, 0.1, 0.1]
    three = [0.5, 0.3, 0.2]
    samples = [(four, confidence_id.max_prob(four))] * 6
    samples += [(three, confidence_id.gini(three))] * 6

    assert not any(f.matches for f in confidence_id.identify(samples)), (
        "pooled fit should fail; that is the situation this diagnosis exists for"
    )

    by_size = confidence_id.identify_by_size(samples)
    assert by_size[4][0].name == "max_prob" and by_size[4][0].matches
    assert by_size[3][0].name == "gini" and by_size[3][0].matches


def test_worst_residuals_surfaces_the_samples_that_break_a_candidate() -> None:
    fits_well = [0.9, 0.1]
    breaks_it = [0.5, 0.5]
    samples = [(fits_well, confidence_id.max_prob(fits_well)), (breaks_it, 0.99)]

    worst = confidence_id.worst_residuals(samples, "max_prob", limit=1)
    distribution, reported, predicted = worst[0]
    assert distribution == breaks_it
    assert reported == 0.99
    assert abs(predicted - 0.5) < 1e-9


def test_diagnosis_reports_whether_distributions_are_complete() -> None:
    """A truncated distribution makes every candidate wrong for the same reason."""
    truncated = [(([0.6, 0.3]), 0.6)]
    assert "sum to 1 within 0.1" in confidence_id.format_diagnosis(truncated)


def test_tolerance_accounts_for_how_much_a_statistic_amplifies_rounding() -> None:
    """A flat threshold rejects correct formulas that divide by (K-1).

    Jev rounds to 2 dp. `max_prob` passes that error through, but
    `norm_max_prob` multiplies it by K/(K-1) to 0.010, twice a flat 5e-3.
    """
    import random

    rng = random.Random(0)
    samples = []
    for size in (4, 3):
        for _ in range(24):
            raw = [rng.random() for _ in range(size)]
            total = sum(raw)
            exact = [x / total for x in raw]
            samples.append(
                ([round(x, 2) for x in exact], round(confidence_id.normalized_max_prob(exact), 2))
            )

    fits = {f.name: f for f in confidence_id.identify(samples)}
    assert fits["norm_max_prob"].matches, fits["norm_max_prob"]
    assert fits["norm_max_prob"].tolerance > 5e-3, "tolerance did not widen for amplification"
    assert not fits["gini"].matches, "tolerance widened so far it accepts the wrong formula"


def test_an_unamplified_statistic_keeps_a_tight_tolerance() -> None:
    """`max_prob` does not divide by (K-1), so its tolerance stays near eps."""
    exact = [[0.62, 0.21, 0.17], [0.44, 0.33, 0.23]]
    samples = [(p, round(max(p), 2)) for p in exact]
    fits = {f.name: f for f in confidence_id.identify(samples)}
    assert fits["max_prob"].matches
    assert fits["max_prob"].tolerance < fits["norm_max_prob"].tolerance


def test_detect_precision_reads_the_quantisation_off_the_data() -> None:
    assert confidence_id.detect_precision([0.25, 0.5, 0.75]) == pytest.approx(0.01)
    assert confidence_id.detect_precision([0.125, 0.5]) == pytest.approx(0.001)
    assert confidence_id.detect_precision([0.1234567]) == 0.0


class TestConcurrentEval:
    def test_parallel_requests_keep_item_order_and_report_wall_time(self):
        from threading import Barrier
        from types import SimpleNamespace

        from levbench.runner import run_eval
        from levbench.tasks import FileItem
        from typesafe_sdk import Noul

        barrier = Barrier(4)

        class ConcurrentStub:
            def system_one(self, state, questions):
                # All four workers must enter before any request can finish.
                barrier.wait(timeout=5)
                return SimpleNamespace(
                    answers={
                        "q": SimpleNamespace(type="noul", noul=0.9 if state.endswith("1") else 0.1)
                    },
                    usage=SimpleNamespace(input_tokens=3, output_tokens=0),
                    model="stub",
                )

        items = [FileItem(f"state {i % 2}", {"q": i % 2 == 1}) for i in range(8)]
        questions = {"q": Noul(instructions="one?")}
        report = run_eval(ConcurrentStub(), "lev", "stub", items, questions, concurrency=4)
        assert len(report.calls) == 8 and report.wall_seconds > 0
        preds = [rec[1] for rec in report.records["q"]]
        assert preds == [i % 2 == 1 for i in range(8)], "results must stay in item order"


class TestCostAccounting:
    def test_cost_is_per_million_tokens_at_list_price(self) -> None:
        from levbench.pricing import cost_usd

        assert cost_usd("jev-latest", 1_000_000, 0) == pytest.approx(0.042)
        assert cost_usd("claude-opus-5", 1_000_000, 1_000_000) == pytest.approx(30.0)
        assert cost_usd("Qwen/Qwen3.5-4B", 1_000_000, 1_000_000) == 0.0, "self-hosted is unmetered"

    def test_billed_totals_win_over_base_counts(self) -> None:
        """The adapter's `*_total` fields include retried attempts -- what is billed."""
        from types import SimpleNamespace

        usage = SimpleNamespace(
            input_tokens=10, input_tokens_total=25, output_tokens=2, output_tokens_total=4
        )
        assert runner._usage_ints(usage)[:2] == (25, 4)

    def test_missing_token_counts_raise_rather_than_count_as_free(self) -> None:
        from types import SimpleNamespace

        with pytest.raises(ValueError, match="refusing to assume 0 tokens"):
            runner._usage_ints(SimpleNamespace(input_tokens=None, output_tokens=None))

    def test_a_substituted_model_is_flagged_in_the_report(self) -> None:
        """Pricing uses the requested model; if the server answered with another,
        the report has to say so."""
        report = runner.EvalReport(backend="jev", model="jev-latest")
        report.calls.append(runner.CallResult({}, 0.3, "jev-1.13.0", 10, 1))
        assert "requested 'jev-latest' but server served ['jev-1.13.0']" in runner.format_report(
            report
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
