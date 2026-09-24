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

    return BatchingTokenizer(
        set(ascii_uppercase)
        | {f" {c}" for c in ascii_uppercase}
        | {str(i) for i in range(9)}
        | {f" {i}" for i in range(9)}
    )


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
            lambda m, h, tk, out, step, cb=None, state=None: saved.append(step),
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
        """A total that is a multiple of `checkpoint_every` saves once; on Modal a
        duplicate is a second multi-hundred-MB adapter write and Volume commit."""
        saved = self.run(tmp_path, monkeypatch, max_steps=12, checkpoint_every=6)
        assert saved == [6, 12], f"expected [6, 12], got {saved}"

    def test_an_uneven_total_still_saves_at_the_end(self, tmp_path, monkeypatch):
        """Dedup must not swallow the final save when it is not on the interval."""
        saved = self.run(tmp_path, monkeypatch, max_steps=10, checkpoint_every=4)
        assert saved == [4, 8, 10], f"expected [4, 8, 10], got {saved}"

    def test_checkpointing_off_still_saves_once(self, tmp_path, monkeypatch):
        saved = self.run(tmp_path, monkeypatch, max_steps=5, checkpoint_every=0)
        assert saved == [5], f"expected a single final save, got {saved}"


class TestResume:
    """A preempted run continues at the step it stopped, through the batches it
    had not seen, on the schedule it had reached -- driven through the real loop
    with only the backbone, forward and weight I/O stubbed."""

    def run(
        self, tmp_path, monkeypatch, *, max_steps, checkpoint_every=3, fresh=False, persist=True
    ):
        from dataclasses import replace

        import lev.train.loop as loop
        from lev.train.checkpoints import TRAINING_STATE
        from lev.train.collate import DecisionCollator
        from lev.train.config import PRESETS

        rows = [an_example(a_choice(4), target=i % 4, state=f"row {i}") for i in range(64)]
        saved: list[int] = []
        seen: list[tuple[str, ...]] = []

        class RecordingCollator(DecisionCollator):
            def __call__(self, examples):
                seen.append(tuple(e.state for e in examples))
                return super().__call__(examples)

        def fake_save(m, h, tk, out, step, cb=None, state=None):
            saved.append(step)
            path = out / f"step-{step}"
            path.mkdir(parents=True, exist_ok=True)
            (path / "adapter_model.safetensors").write_bytes(b"")
            if state is not None and persist:
                torch.save(state, path / TRAINING_STATE)

        monkeypatch.setattr(loop, "prepare_data", lambda c, d: {"train": rows, "calibration": []})
        monkeypatch.setattr(
            loop, "build_model", lambda c, mc=None: (StubModel(), make_batching_tokenizer())
        )
        monkeypatch.setattr(loop, "build_head", lambda c, m=None: None)
        monkeypatch.setattr(loop, "device_of", lambda m: torch.device("cpu"))
        monkeypatch.setattr(loop, "to_device", lambda b, d: b)
        monkeypatch.setattr(loop, "load_checkpoint", lambda m, h, p: None)
        monkeypatch.setattr(loop, "DecisionCollator", RecordingCollator)
        monkeypatch.setattr(
            loop,
            "candidate_logits",
            lambda m, b, h=None: torch.zeros(
                b.size, b.candidate_mask.size(1), requires_grad=True
            ).masked_fill(b.candidate_mask, float("-inf")),
        )
        monkeypatch.setattr(loop, "save_checkpoint", fake_save)

        config = replace(
            PRESETS["smoke"],
            epochs=2,
            per_device_batch=2,
            checkpoint_every=checkpoint_every,
            output_dir=str(tmp_path / "out"),
        )
        summary = loop.run_training(config, str(tmp_path), max_steps=max_steps, fresh=fresh)
        return saved, seen, summary

    def test_resumes_at_the_saved_step_and_sees_only_the_remaining_batches(
        self, tmp_path, monkeypatch
    ):
        _, full, _ = self.run(tmp_path / "ref", monkeypatch, max_steps=10)
        # The same run, stopped at 6 and restarted.
        saved_a, seen_a, _ = self.run(tmp_path, monkeypatch, max_steps=6)
        saved_b, seen_b, summary = self.run(tmp_path, monkeypatch, max_steps=10)

        assert saved_a == [3, 6]
        assert saved_b == [9, 10], f"resumed run should save at 9 and 10, got {saved_b}"
        assert summary["steps"] == 10
        assert seen_b == full[6:10], "the resumed run must process exactly the unseen batches"
        assert [h["step"] for h in summary["history"]] == list(range(10)), (
            "history must extend the saved curve, not restart it"
        )

    def test_saved_state_records_the_schedule_position(self, tmp_path, monkeypatch):
        from lev.train.checkpoints import load_training_state

        self.run(tmp_path, monkeypatch, max_steps=6)
        state = load_training_state(tmp_path / "out" / "step-6")
        assert state["step"] == 6 and state["epoch"] == 0 and state["step_in_epoch"] == 6
        assert state["scheduler"]["last_epoch"] == 6

    def test_fresh_ignores_the_checkpoint(self, tmp_path, monkeypatch):
        self.run(tmp_path, monkeypatch, max_steps=6)
        saved, _, summary = self.run(tmp_path, monkeypatch, max_steps=6, fresh=True)
        assert saved == [3, 6]
        assert summary["history"][0]["step"] == 0

    def test_weights_only_checkpoint_restarts_the_schedule(self, tmp_path, monkeypatch):
        """A checkpoint without training state (as before ADR-021) restores its
        weights and counts from zero."""
        self.run(tmp_path, monkeypatch, max_steps=6, persist=False)
        saved, _, summary = self.run(tmp_path, monkeypatch, max_steps=6)
        assert saved == [3, 6]
        assert summary["history"][0]["step"] == 0

    def test_resume_across_an_epoch_boundary(self, tmp_path, monkeypatch):
        """64 rows at batch 2 is 32 steps per epoch; stopping at 33 is one step
        into epoch 1, and the resumed run must skip exactly that one."""
        _, full, _ = self.run(tmp_path / "ref", monkeypatch, max_steps=36)
        self.run(tmp_path, monkeypatch, max_steps=33)
        _, seen, summary = self.run(tmp_path, monkeypatch, max_steps=36)
        assert summary["steps"] == 36
        assert seen == full[33:36]


class TestFreshSetsAsideThePreviousRun:
    def test_stale_checkpoints_and_calibration_are_moved_not_deleted(self, tmp_path):
        from lev.train.loop import set_aside_previous_run

        out = tmp_path / "out"
        (out / "step-18750").mkdir(parents=True)
        (out / "step-18750" / "adapter_model.safetensors").write_bytes(b"old")
        (out / "calibration.json").write_text("{}")
        (out / "history.json").write_text("{}")

        moved = set_aside_previous_run(out)
        assert moved is not None and moved.parent == out and moved.name.startswith("superseded-")
        assert not (out / "step-18750").exists() and not (out / "calibration.json").exists()
        assert (moved / "step-18750" / "adapter_model.safetensors").read_bytes() == b"old"
        assert (moved / "calibration.json").is_file()
        assert set_aside_previous_run(out) is None

    def test_fresh_run_does_not_resume_from_a_higher_stale_step(self, tmp_path, monkeypatch):
        """The hazard: a fresh run writes step-3, step-6 beside a stale step-18750;
        a preemption's auto-resume would take the stale one. `--fresh` moves it."""
        from lev.train.checkpoints import latest_checkpoint

        runner = TestResume()
        stale = tmp_path / "out" / "step-18750"
        stale.mkdir(parents=True)
        (stale / "adapter_model.safetensors").write_bytes(b"corrupted run")
        saved, _, summary = runner.run(tmp_path, monkeypatch, max_steps=6, fresh=True)
        assert saved == [3, 6] and summary["history"][0]["step"] == 0
        assert latest_checkpoint(tmp_path / "out").name == "step-6"


class TestModeAProjectsOnlyTheAnswerPositions:
    def test_logits_to_keep_selects_distinct_positions_and_rows_read_their_own(self):
        """Mode A must not materialise (B, seq, V): the model is asked for the
        distinct answer positions only, and each row reads its own column."""
        from types import SimpleNamespace

        from lev.router import Mode
        from lev.train.loop import candidate_logits

        seen = {}

        class FakeModel:
            def __call__(self, input_ids, attention_mask, logits_to_keep):
                seen["keep"] = logits_to_keep.tolist()
                b, v = input_ids.shape[0], 10
                # logits[row, j, tok] = 100*row + 10*keep[j] + tok, so the value
                # read identifies which row and position it came from.
                keep = logits_to_keep.float()
                rows = torch.arange(b).float()[:, None, None] * 100
                pos = keep[None, :, None] * 10
                tok = torch.arange(v).float()[None, None, :]
                return SimpleNamespace(logits=rows + pos + tok)

        batch = SimpleNamespace(
            mode=Mode.LABEL_TOKEN,
            size=3,
            input_ids=torch.zeros(3, 8, dtype=torch.long),
            attention_mask=torch.ones(3, 8, dtype=torch.long),
            last_positions=torch.tensor([7, 4, 7]),
            candidate_token_ids=torch.tensor([[1, 2], [1, 2], [3, 4]]),
            candidate_mask=torch.zeros(3, 2, dtype=torch.bool),
        )
        out = candidate_logits(FakeModel(), batch)
        assert seen["keep"] == [4, 7], "only the distinct answer positions are projected"
        assert out.tolist() == [[71.0, 72.0], [141.0, 142.0], [273.0, 274.0]]
