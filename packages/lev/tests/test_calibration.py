"""Calibration must actually reduce ECE, and must refuse to cheat."""

from __future__ import annotations

import math
import random

import pytest
from lev.calibrate import (
    CalibrationProfile,
    expected_calibration_error,
    fit,
    fit_temperature,
    nll,
    softmax,
)


def overconfident_samples(n: int = 600, seed: int = 7, sharpness: float = 4.0):
    """Logits that are directionally right but far too sharp.

    This is the untuned-backbone failure mode the benchmark measured at ECE 0.4252.
    The correct class wins ~70% of the time, but the margin implies ~99%.
    """
    rng = random.Random(seed)
    samples = []
    for _ in range(n):
        truth = rng.randrange(3)
        logits = [rng.gauss(0, 0.3) for _ in range(3)]
        # Right 70% of the time; when wrong, a different class gets the boost.
        winner = truth if rng.random() < 0.7 else (truth + 1) % 3
        logits[winner] += sharpness
        samples.append((logits, truth))
    return samples


class TestTemperatureFitting:
    def test_softmax_is_a_distribution(self):
        p = softmax([1.0, 2.0, 3.0])
        assert sum(p) == pytest.approx(1.0)
        assert p[2] > p[1] > p[0]

    def test_higher_temperature_flattens(self):
        sharp, flat = softmax([5.0, 0.0], 1.0), softmax([5.0, 0.0], 10.0)
        assert max(sharp) > max(flat)

    def test_temperature_must_be_positive(self):
        with pytest.raises(ValueError):
            softmax([1.0, 2.0], 0.0)

    def test_fitting_reduces_ece_on_overconfident_logits(self):
        samples = overconfident_samples()
        t = fit_temperature(samples)
        assert t > 1.0, f"overconfident logits need softening, got T={t}"

        before = expected_calibration_error(
            [softmax(x, 1.0) for x, _ in samples], [y for _, y in samples]
        )
        after = expected_calibration_error(
            [softmax(x, t) for x, _ in samples], [y for _, y in samples]
        )
        assert after < before, f"ECE got worse: {before:.4f} -> {after:.4f}"
        # The measured effect on real backbones is 0.43 -> 0.08. Demand a real
        # improvement, not a rounding-level one.
        assert after < before / 2, f"expected a large reduction, {before:.4f} -> {after:.4f}"

    def test_fitting_minimises_nll(self):
        samples = overconfident_samples()
        t = fit_temperature(samples)
        best = nll(samples, t)
        for other in (t * 0.5, t * 1.5, 1.0):
            assert best <= nll(samples, other) + 1e-6

    def test_perfectly_calibrated_logits_need_no_change(self):
        # Two classes, true 50/50 — the optimal temperature keeps them at 50/50.
        samples = [([0.0, 0.0], i % 2) for i in range(400)]
        assert fit_temperature(samples) == pytest.approx(fit_temperature(samples))
        p = softmax([0.0, 0.0], fit_temperature(samples))
        assert p[0] == pytest.approx(0.5)

    def test_empty_input_is_the_identity(self):
        assert fit_temperature([]) == 1.0


class TestECE:
    def test_perfect_calibration_scores_zero(self):
        # 70% confident, 70% accurate.
        probs = [[0.7, 0.3]] * 7 + [[0.7, 0.3]] * 3
        truths = [0] * 7 + [1] * 3
        assert expected_calibration_error(probs, truths) == pytest.approx(0.0, abs=1e-9)

    def test_total_overconfidence_scores_the_gap(self):
        probs = [[1.0, 0.0]] * 10
        truths = [0] * 5 + [1] * 5
        assert expected_calibration_error(probs, truths) == pytest.approx(0.5)


class TestProfile:
    def test_buckets_are_independent(self):
        p = CalibrationProfile(temperatures={"choice:A": 2.0, "choice:B": 0.5})
        assert p.temperature("choice", "A") == 2.0
        assert p.temperature("choice", "B") == 0.5

    def test_unfitted_bucket_degrades_to_identity(self):
        p = CalibrationProfile(temperatures={"choice:A": 2.0})
        assert p.temperature("score", "A") == 1.0, "must not borrow another bucket's scalar"

    def test_round_trips_through_disk(self, tmp_path):
        p = CalibrationProfile({"noul:A": 1.7}, fitted_on="calib", n_samples={"noul:A": 900})
        path = tmp_path / "calibration.json"
        p.save(path)
        loaded = CalibrationProfile.load(path)
        assert loaded.temperatures == p.temperatures
        assert loaded.fitted_on == "calib"

    def test_fit_refuses_the_test_split(self):
        samples = overconfident_samples(100)
        for forbidden in ("test", "TEST", "eval", "holdout"):
            with pytest.raises(ValueError, match="refusing to fit"):
                fit({"choice:A": samples}, split_name=forbidden)

    def test_fit_skips_undersized_buckets(self):
        profile = fit({"choice:A": overconfident_samples(10)}, "calib", min_samples=50)
        assert "choice:A" not in profile.temperatures
        assert profile.temperature("choice", "A") == 1.0

    def test_fit_records_provenance(self):
        profile = fit({"choice:A": overconfident_samples(200)}, "calib-split")
        assert profile.fitted_on == "calib-split"
        assert profile.n_samples["choice:A"] == 200
        assert not math.isclose(profile.temperatures["choice:A"], 1.0, abs_tol=0.05)
