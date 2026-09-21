"""Progress reporting: what the log says while a run is in flight.

torch is in the `train` extra, so these skip on a bare `uv sync`.
"""

from __future__ import annotations

import re

import pytest

torch = pytest.importorskip("torch")


class TestProgressLog:
    """A silent run is indistinguishable from a hung one."""

    def make(self, capsys, total=100, every=10):
        from lev.train.progress import ProgressLog

        return ProgressLog(total, every)

    def test_logs_only_on_the_interval(self, capsys):
        log = self.make(capsys)
        for step in range(9):
            log.record(step, 1.0, "A", 1e-4, tokens=100)
        assert capsys.readouterr().out == ""
        log.record(9, 1.0, "A", 1e-4, tokens=100)
        assert "step     10/100" in capsys.readouterr().out

    def test_always_logs_the_final_step(self, capsys):
        """An interval that does not divide the total must not swallow the end."""
        log = self.make(capsys, total=7, every=10)
        for step in range(7):
            log.record(step, 1.0, "A", 1e-4, tokens=10)
        assert "step      7/7" in capsys.readouterr().out

    def test_modes_are_reported_separately(self, capsys):
        """Mode B starts near ln(K); blending it with Mode A hides both."""
        log = self.make(capsys, total=2, every=2)
        log.record(0, 1.0, "A", 1e-4, tokens=10)
        log.record(1, 5.0, "B", 1e-4, tokens=10)
        out = capsys.readouterr().out
        assert "A=1.0000" in out and "B=5.0000" in out
        assert "3.0000" not in out, "the two modes were averaged together"

    def test_window_resets_between_reports(self, capsys):
        log = self.make(capsys, total=4, every=2)
        log.record(0, 10.0, "A", 1e-4)
        log.record(1, 10.0, "A", 1e-4)
        capsys.readouterr()
        log.record(2, 1.0, "A", 1e-4)
        log.record(3, 1.0, "A", 1e-4)
        assert "A=1.0000" in capsys.readouterr().out, "stale window carried forward"

    def test_throughput_counts_real_tokens(self, capsys):
        """Derived from the attention mask, not from the config's average.

        That constant was wrong by 10x once (ADR-016); a readout derived from
        it would have agreed with the mistake rather than exposed it.
        """
        log = self.make(capsys, total=1, every=1)
        log.record(0, 1.0, "A", 1e-4, tokens=4096)
        out = capsys.readouterr().out
        # Parse the value rather than substring-match it: a fast window makes
        # the rate large, and "4096000000 tok/s" contains "0 tok/s".
        match = re.search(r"([\d,]+) tok/s", out)
        assert match, out
        assert int(match.group(1).replace(",", "")) > 0

    def test_throughput_is_zero_when_nothing_was_counted(self, capsys):
        log = self.make(capsys, total=1, every=1)
        log.record(0, 1.0, "A", 1e-4)
        assert re.search(r"\b0 tok/s", capsys.readouterr().out)

    def test_hms_formats_hours(self):
        from lev.train.progress import hms

        assert hms(0) == "0:00:00"
        assert hms(59) == "0:00:59"
        assert hms(3661) == "1:01:01"


class TestWindowedRate:
    """Startup cost is a one-off; a cumulative average never stops paying it.

    Measured on the 0.8B Modal smoke: 4.68 s/step for the first 25 steps
    (weight load + Triton JIT), 0.40 s/step after. Reported cumulatively that
    is 0.33 it/s against a true 2.50 -- and the ETA is wrong by the same 7.5x,
    which is how a healthy run gets abandoned for being slow.
    """

    def test_rate_reflects_the_window_not_the_startup_stall(self, capsys, monkeypatch):
        import lev.train.progress as progress

        clock = {"t": 0.0}
        monkeypatch.setattr(progress.time, "monotonic", lambda: clock["t"])

        log = progress.ProgressLog(total_steps=40, log_every=20)
        # First window: a 100 s stall, then 20 quick steps.
        clock["t"] = 100.0
        for step in range(20):
            clock["t"] += 0.4
            log.record(step, 1.0, "A", 1e-4, tokens=100)
        first = capsys.readouterr().out

        # Second window: steady state only.
        for step in range(20, 40):
            clock["t"] += 0.4
            log.record(step, 1.0, "A", 1e-4, tokens=100)
        second = capsys.readouterr().out

        rate = lambda out: float(re.search(r"([\d.]+) it/s", out).group(1))  # noqa: E731
        assert rate(second) == pytest.approx(2.5, abs=0.05), second
        # The steady-state window must not be dragged by the earlier stall.
        assert rate(second) > 5 * rate(first), f"{rate(first)} -> {rate(second)}"

    def test_eta_uses_the_windowed_rate(self, capsys, monkeypatch):
        import lev.train.progress as progress

        clock = {"t": 0.0}
        monkeypatch.setattr(progress.time, "monotonic", lambda: clock["t"])
        log = progress.ProgressLog(total_steps=1000, log_every=10)

        clock["t"] = 500.0  # a long stall before the first step
        for step in range(10):
            clock["t"] += 1.0
            log.record(step, 1.0, "A", 1e-4, tokens=10)
        capsys.readouterr()
        for step in range(10, 20):
            clock["t"] += 1.0
            log.record(step, 1.0, "A", 1e-4, tokens=10)

        out = capsys.readouterr().out
        # 980 steps left at 1 s/step = 0:16:20, not inflated by the 500 s stall.
        assert "eta 0:16:20" in out, out

    def test_throughput_is_also_windowed(self, capsys, monkeypatch):
        import lev.train.progress as progress

        clock = {"t": 0.0}
        monkeypatch.setattr(progress.time, "monotonic", lambda: clock["t"])
        log = progress.ProgressLog(total_steps=4, log_every=2)

        clock["t"] = 100.0
        for step in range(2):
            clock["t"] += 1.0
            log.record(step, 1.0, "A", 1e-4, tokens=1000)
        capsys.readouterr()
        for step in range(2, 4):
            clock["t"] += 1.0
            log.record(step, 1.0, "A", 1e-4, tokens=1000)

        out = capsys.readouterr().out
        # 2000 tokens over 2 s, not 4000 over 102 s.
        assert "1,000 tok/s" in out, out
