"""End-to-end plumbing check against the fake transport. No API key needed."""

from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = next(p for p in PKG.parents if (p / "data").is_dir())
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(PKG / "tests"))

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
    print("metrics maths OK")


def test_answer_flattening() -> None:
    """Every primitive must flatten, including the fieldless Noul."""
    items, questions = dataset()
    result = runner.call_once(fake_client(), items[0].state, questions)

    noul = result.answers["is_urgent"]
    assert not hasattr(noul, "confidence") or noul.confidence is None
    dist = metrics.to_distribution(noul)
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert metrics.confidence(noul) >= 0.5, "synthesised Noul confidence must be >= 0.5"

    choice = result.answers["department"]
    assert metrics.predicted_label(choice) == choice.choice
    assert abs(sum(metrics.to_distribution(choice).values()) - 1.0) < 1e-3

    score = result.answers["frustration"]
    sdist = metrics.to_distribution(score)
    expected = sum(k * v for k, v in sdist.items())
    assert abs(score.score - expected) < 1e-2, "score must equal the probability-weighted mean"
    print("answer flattening OK (all three primitives)")


def test_full_eval_runs() -> None:
    items, questions = dataset()
    report = runner.run_eval(fake_client(), "jev", "jev-latest", items, questions)
    assert len(report.calls) == len(items)
    assert set(report.per_question) == set(questions)
    assert report.total_cost > 0
    text = runner.format_report(report)
    assert "accuracy" in text and "ECE" in text
    print(f"eval ran over {len(items)} items, cost ${report.total_cost:.6f}")


def test_sweep_arithmetic_over_a_fixed_billing_shape() -> None:
    """Validate SweepRow arithmetic, NOT Jev's real billing behaviour.

    The fake transport hardcodes per-request input billing, so the growing
    cost ratio below is baked in by construction and this assertion cannot
    fail on a real-world billing change. It checks that `batching.py` divides
    and accumulates correctly given a known billing shape. Whether Jev
    actually bills that way is only answerable by a live `levbench sweep`.
    """
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
    print(batching.format_sweep(rows, "jev-latest (FAKE transport)", len(state)))


def test_confidence_identifier_recovers_a_planted_formula() -> None:
    """Known-answer test: plant each formula, demand it be identified uniquely.

    Validates the identifier before it is pointed at a real server, where the
    answer is unknown. Ambiguity here would mean a match against Jev proves
    nothing.
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
                raw = [1 + (hash((key, o)) % 97) for o in options]
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
    print(f"confidence identifier uniquely recovers all {len(confidence_id.CANDIDATES)} formulas")


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
    print(f"base_url routes to {seen['url']}")


def test_local_base_url_never_forwards_the_real_key() -> None:
    """Pointing at a third-party server must not hand it your Jev credential.

    The SDK always sends `Authorization: Bearer <key>`, so reusing
    TYPESAFE_API_KEY for a `--base-url` run would leak it to whatever host the
    user named. The hosted path must still authenticate normally.
    """
    import os

    previous = os.environ.get("TYPESAFE_API_KEY")
    os.environ["TYPESAFE_API_KEY"] = "sk-REAL-SECRET-KEY"
    try:
        local, _ = runner.build_client("jev", "litjev", "http://127.0.0.1:8000")
        assert "REAL-SECRET" not in str(local._config.api_key), "leaked the real key"
        assert local._config.api_key == "local"

        hosted, _ = runner.build_client("jev", "jev-latest")
        assert hosted._config.api_key == "sk-REAL-SECRET-KEY", "hosted path lost its key"
    finally:
        if previous is None:
            del os.environ["TYPESAFE_API_KEY"]
        else:
            os.environ["TYPESAFE_API_KEY"] = previous
    print("local base_url uses a placeholder; hosted path still authenticates")


if __name__ == "__main__":
    test_metrics_are_arithmetically_right()
    test_answer_flattening()
    test_full_eval_runs()
    test_sweep_arithmetic_over_a_fixed_billing_shape()
    test_confidence_identifier_recovers_a_planted_formula()
    test_base_url_routes_to_a_local_clone()
    test_local_base_url_never_forwards_the_real_key()
    print("\nAll offline checks passed.")
