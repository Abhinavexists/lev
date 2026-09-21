"""The training loop's control flow, with the backbone stubbed out.

torch is in the `train` extra, so these skip on a bare `uv sync`.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from conftest import a_choice, an_example  # noqa: E402


class StubModel:
    """Just enough surface for the loop: parameters, train(), no real forward."""

    def __init__(self):
        self._p = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._p])

    def train(self):
        return self

    def __call__(self, **kwargs):
        raise AssertionError("candidate_logits is stubbed; the model is never called")


def make_batching_tokenizer():
    from string import ascii_uppercase

    from conftest import BatchingTokenizer

    return BatchingTokenizer({f" {c}" for c in ascii_uppercase} | {f" {i}" for i in range(9)})


class TestCheckpointCadence:
    """Drives the real `run_training` loop with the expensive parts stubbed.

    Only the backbone, the forward pass and the checkpoint write are replaced;
    the batching, the mode routing, the step counter and the checkpoint
    decisions are the shipping code.
    """

    def run(self, tmp_path, monkeypatch, *, max_steps, checkpoint_every, epochs=1):
        from dataclasses import replace

        import lev.train.loop as loop
        from lev.train.config import PRESETS

        rows = [an_example(a_choice(4), target=i % 4, state=f"row {i}") for i in range(64)]
        saved: list[int] = []

        monkeypatch.setattr(loop, "prepare_data", lambda c, d: {"train": rows, "calibration": []})
        monkeypatch.setattr(
            loop, "build_model", lambda c, mc=None: (StubModel(), make_batching_tokenizer())
        )
        monkeypatch.setattr(loop, "build_head", lambda c, m=None: None)
        monkeypatch.setattr(loop, "device_of", lambda m: torch.device("cpu"))
        monkeypatch.setattr(loop, "to_device", lambda b, d: b)
        monkeypatch.setattr(
            loop,
            "candidate_logits",
            lambda m, b, h=None: torch.zeros(
                b.size, b.candidate_mask.size(1), requires_grad=True
            ).masked_fill(b.candidate_mask, float("-inf")),
        )
        monkeypatch.setattr(
            loop,
            "save_checkpoint",
            lambda m, h, tk, out, step, cb=None: saved.append(step),
        )

        config = replace(
            PRESETS["smoke"],
            epochs=epochs,
            per_device_batch=2,
            checkpoint_every=checkpoint_every,
            output_dir=str(tmp_path / "out"),
        )
        loop.run_training(config, str(tmp_path), max_steps=max_steps)
        return saved

    def test_the_final_step_is_not_checkpointed_twice(self, tmp_path, monkeypatch):
        """A total that is a multiple of `checkpoint_every` used to save twice.

        Locally that is a wasted write; on Modal it is a second multi-hundred-MB
        adapter dump and a second Volume commit for an identical result.
        """
        saved = self.run(tmp_path, monkeypatch, max_steps=12, checkpoint_every=6)
        assert saved == [6, 12], f"expected [6, 12], got {saved}"

    def test_an_uneven_total_still_saves_at_the_end(self, tmp_path, monkeypatch):
        """Dedup must not swallow the final save when it is not on the interval."""
        saved = self.run(tmp_path, monkeypatch, max_steps=10, checkpoint_every=4)
        assert saved == [4, 8, 10], f"expected [4, 8, 10], got {saved}"

    def test_checkpointing_off_still_saves_once(self, tmp_path, monkeypatch):
        saved = self.run(tmp_path, monkeypatch, max_steps=5, checkpoint_every=0)
        assert saved == [5], f"expected a single final save, got {saved}"
