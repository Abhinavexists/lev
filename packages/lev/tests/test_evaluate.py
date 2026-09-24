import math

import pytest


class TestEvalScoring:
    """`score` turns raw logits into the numbers a run is judged on.

    Temperature must move calibration and leave accuracy alone.
    """

    def confident_but_wrong(self):
        # Right 50% of the time, always at ~0.95 confidence: badly overconfident.
        return [([3.0, 0.0], 0), ([3.0, 0.0], 1), ([3.0, 0.0], 0), ([3.0, 0.0], 1)]

    def test_temperature_does_not_change_accuracy(self):
        from lev.train.evaluate import score

        samples = self.confident_but_wrong()
        assert score(samples, 1.0)[0] == score(samples, 4.0)[0]

    def test_softening_an_overconfident_model_lowers_ece(self):
        from lev.train.evaluate import score

        samples = self.confident_but_wrong()
        assert score(samples, 4.0)[1] < score(samples, 1.0)[1]

    def test_perfect_predictions_score_accuracy_one(self):
        from lev.train.evaluate import score

        accuracy, _, mean_brier, mean_ll = score([([9.0, 0.0], 0), ([0.0, 9.0], 1)], 1.0)
        assert accuracy == 1.0
        assert mean_brier < 1e-3
        assert mean_ll < 1e-3


class TestAccuracyInterval:
    """The report must state its own resolution: the eval is not reproducible."""

    def test_interval_shrinks_with_n(self):
        from lev.train.evaluate import accuracy_interval

        assert accuracy_interval(0.86, 200) > accuracy_interval(0.86, 1800)

    def test_matches_the_normal_approximation(self):
        from lev.train.evaluate import accuracy_interval

        assert accuracy_interval(0.86, 200) == pytest.approx(0.048, abs=0.001)

    def test_a_certain_estimate_still_reports_a_finite_interval(self):
        """p=1.0 gives zero variance; the formula must not claim infinite precision."""
        from lev.train.evaluate import accuracy_interval

        assert 0.0 <= accuracy_interval(1.0, 200) < 0.01

    def test_no_items_means_no_information(self):
        from lev.train.evaluate import accuracy_interval

        assert accuracy_interval(0.5, 0) == 1.0


def test_mixed_temperatures_are_applied_per_row():
    from lev.train.evaluate import score_mixed

    # Both rows assign 0.75 to class 0 after their respective temperatures.
    rows = [([math.log(3), 0.0], 0, 1.0), ([math.log(9), 0.0], 1, 2.0)]
    accuracy, ece, mean_brier, mean_log_loss = score_mixed(rows)
    assert accuracy == 0.5
    assert ece == pytest.approx(0.25)
    assert mean_brier == pytest.approx(0.625)
    assert mean_log_loss == pytest.approx(-math.log(0.75 * 0.25) / 2)
