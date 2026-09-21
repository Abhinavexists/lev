"""The objective and the collator, on tensors rather than a backbone.

torch is in the `train` extra, so these skip on a bare `uv sync`.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lev.data.mixture import Example  # noqa: E402
from lev.prompt import Layout  # noqa: E402
from lev.router import Mode  # noqa: E402
from lev.train.collate import IGNORE_INDEX, DecisionCollator, ModeBatcher  # noqa: E402
from lev.train.config import PRESETS  # noqa: E402
from lev.train.loop import decision_loss  # noqa: E402
from lev.types import Choice, Noul, Score  # noqa: E402


@pytest.fixture
def config():
    from dataclasses import replace

    return replace(PRESETS["smoke"])


def choice(n=4):
    return Choice(instructions="pick", criteria={f"option {i}": None for i in range(n)})


def example(question, target=0, source="s", state="a state", abstain=False, soft=None):
    return Example(
        state=state,
        name=source,
        question=question,
        target=target,
        layout=Layout.STATE_FIRST,
        source=source,
        abstain=abstain,
        soft_target=soft,
    )


class TestDecisionLoss:
    def test_padded_slots_do_not_produce_nan(self, config):
        """The regression from the first smoke run: every loss was `nan`.

        Padded candidates carry a logit of `-inf` so they win no softmax mass,
        and their target probability is 0, so a plain product is `0 * -inf`.
        One padded slot anywhere poisons the batch mean, and because backward
        still runs the symptom is a nan loss rather than a crash.
        """
        logits = torch.tensor(
            [[1.0, 2.0, 0.5, 0.1], [0.3, 1.2, float("-inf"), float("-inf")]],
            requires_grad=True,
        )
        loss = decision_loss(logits, torch.tensor([1, 0]), config)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(logits.grad).all()

    def test_matches_cross_entropy_when_only_ce_is_enabled(self, config):
        """Supporting soft targets must not change the hard-target answer."""
        import torch.nn.functional as F

        config.brier_weight = 0.0
        config.ordinal_weight = 0.0
        logits, targets = torch.randn(8, 5), torch.randint(0, 5, (8,))
        assert torch.allclose(
            decision_loss(logits, targets, config), F.cross_entropy(logits, targets), atol=1e-6
        )

    def test_brier_penalises_overconfidence_at_equal_accuracy(self, config):
        """Why Brier is in the loss at all: CE alone optimises the argmax."""
        config.brier_weight = 1.0
        config.ordinal_weight = 0.0
        targets = torch.tensor([0])
        confident = decision_loss(torch.tensor([[8.0, 0.0, 0.0]]), targets, config)
        modest = decision_loss(torch.tensor([[2.0, 0.0, 0.0]]), targets, config)
        # Both are correct. The overconfident one must still be preferred *less*
        # than it would be under CE alone -- checked by comparing the gap.
        config.brier_weight = 0.0
        ce_gap = modest - decision_loss(torch.tensor([[8.0, 0.0, 0.0]]), targets, config)
        config.brier_weight = 1.0
        assert (modest - confident) < ce_gap

    def test_uniform_soft_target_is_minimised_by_a_uniform_prediction(self, config):
        config.brier_weight = 0.0
        config.ordinal_weight = 0.0
        soft = torch.full((1, 4), 0.25)
        targets = torch.tensor([IGNORE_INDEX])
        flat = decision_loss(torch.zeros(1, 4), targets, config, soft_targets=soft)
        peaked = decision_loss(
            torch.tensor([[5.0, 0.0, 0.0, 0.0]]), targets, config, soft_targets=soft
        )
        assert flat < peaked

    def test_hard_and_abstain_rows_coexist_in_one_batch(self, config):
        logits = torch.tensor([[1.0, 2.0, 0.5, 0.1], [0.3, 1.2, 0.0, 0.0]], requires_grad=True)
        soft = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]])
        loss = decision_loss(logits, torch.tensor([1, IGNORE_INDEX]), config, soft_targets=soft)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()

    def test_ordinal_term_prefers_a_near_miss(self, config):
        config.brier_weight = 0.0
        config.ordinal_weight = 2.0
        targets = torch.tensor([4])
        near = decision_loss(torch.tensor([[0.0, 0.0, 0.0, 5.0, 0.0]]), targets, config, True)
        far = decision_loss(torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0]]), targets, config, True)
        assert near < far, "an ordered scale must not treat every wrong level alike"

    def test_ordinal_applies_only_to_the_rows_flagged(self, config):
        """`ordinal` is per-row: a Score and a Choice can share a batch."""
        config.ordinal_weight = 5.0
        logits = torch.randn(4, 5)
        targets = torch.tensor([0, 1, 2, 3])
        none = decision_loss(logits, targets, config, torch.zeros(4, dtype=torch.bool))
        some = decision_loss(logits, targets, config, torch.tensor([True, False, False, False]))
        assert not torch.isclose(none, some)


class TestCollator:
    @pytest.fixture
    def collator(self, batching_tokenizer):
        return DecisionCollator(batching_tokenizer, max_seq_len=512)

    def test_last_position_is_the_final_real_token_not_seq_minus_one(self, collator):
        """Rows are right-padded, so `seq - 1` reads a pad token's logits.

        That trains on noise and never raises.
        """
        batch = collator([example(choice(), state="short"), example(choice(), state="x" * 200)])
        assert batch.last_positions[0] < batch.last_positions[1]
        for row, position in enumerate(batch.last_positions):
            assert batch.attention_mask[row, position] == 1
            if position + 1 < batch.attention_mask.size(1):
                assert batch.attention_mask[row, position + 1] == 0

    def test_ragged_candidate_sets_are_masked_not_truncated(self, collator):
        batch = collator([example(choice(2)), example(choice(5))])
        assert batch.candidate_mask.shape == (2, 5)
        assert batch.candidate_mask[0].tolist() == [False, False, True, True, True]
        assert batch.candidate_mask[1].tolist() == [False] * 5

    def test_noul_gets_nine_rating_slots_not_two(self, collator):
        batch = collator([example(Noul(instructions="is it?"), target=8)])
        assert batch.candidate_mask.shape[1] == 9
        assert batch.targets.tolist() == [8]

    def test_abstain_rows_carry_ignore_index_and_a_soft_row(self, collator):
        batch = collator(
            [
                example(choice(4), target=1),
                example(choice(4), abstain=True, soft=[0.25] * 4),
            ]
        )
        assert batch.targets.tolist() == [1, IGNORE_INDEX]
        assert batch.soft_targets[1].tolist() == [0.25] * 4

    def test_no_soft_tensor_is_built_when_nothing_abstains(self, collator):
        assert collator([example(choice())]).soft_targets is None

    def test_soft_target_of_the_wrong_width_raises(self, collator):
        with pytest.raises(ValueError, match="soft_target has"):
            collator([example(choice(4), abstain=True, soft=[0.5, 0.5])])

    def test_score_rows_are_flagged_ordinal(self, collator):
        score = Score(instructions="how much", criteria=["low", "mid", "high"])
        batch = collator([example(score), example(choice())])
        assert batch.ordinal.tolist() == [True, False]

    def test_a_small_option_set_routes_to_mode_a_with_token_ids(self, collator):
        batch = collator([example(choice(4))])
        assert batch.mode is Mode.LABEL_TOKEN
        assert batch.candidate_token_ids is not None
        assert batch.candidate_input_ids is None

    def test_a_large_option_set_routes_to_mode_b_with_a_candidate_pool(self, collator):
        batch = collator([example(choice(40))])
        assert batch.mode is Mode.CANDIDATE_PATH
        assert batch.candidate_token_ids is None
        assert batch.candidate_index.shape == (1, 40)
        # 40 distinct option strings, pooled once rather than per row.
        assert batch.candidate_input_ids.shape[0] == 40

    def test_the_candidate_pool_is_shared_across_rows(self, collator):
        """A 77-option question must not cost 8x77 encodes in a batch of 8."""
        rows = [example(choice(40), target=i) for i in range(4)]
        batch = collator(rows)
        assert batch.candidate_input_ids.shape[0] == 40, "pool grew with batch size"

    def test_mixing_modes_in_one_batch_raises(self, collator):
        with pytest.raises(ValueError, match="mixes readout modes"):
            collator([example(choice(4)), example(choice(40))])

    def test_empty_batch_raises(self, collator):
        with pytest.raises(ValueError, match="empty batch"):
            collator([])


class TestModeBatcher:
    def test_batches_are_homogeneous_and_correctly_sized(self, batching_tokenizer):
        batcher = ModeBatcher(batching_tokenizer, batch_size=3)
        rows = [example(choice(4 if i % 2 else 40), source=f"s{i % 2}") for i in range(12)]
        groups = list(batcher(rows))
        assert all(len(g) <= 3 for g in groups)
        for group in groups:
            assert len({batcher.route_for(e).mode for e in group}) == 1
        assert sum(len(g) for g in groups) == 12

    def test_routes_are_cached_per_question_not_per_row(self, batching_tokenizer):
        """Re-tokenising an option set for each of 200k rows is the slowest
        thing in the pipeline, and it produces the same answer every time."""
        batcher = ModeBatcher(batching_tokenizer, batch_size=2)
        rows = [example(choice(4), source="same") for _ in range(50)]
        list(batcher(rows))
        assert len(batcher._routes) == 1


class TestOrdinalCoverage:
    def test_noul_rows_are_flagged_ordinal(self, batching_tokenizer):
        """A Noul's scale is the most explicitly ordered thing in the schema.

        Its prompt says "0 = certainly no, 8 = certainly yes" and its target is
        one of the two ends. Leaving it out of the ordinal term makes rating 4
        as wrong as rating 0 when the truth is 8.
        """
        collator = DecisionCollator(batching_tokenizer, max_seq_len=512)
        batch = collator([example(Noul(instructions="urgent?"), target=8)])
        assert batch.ordinal.tolist() == [True]

    def test_choice_rows_are_not_flagged_ordinal(self, batching_tokenizer):
        """Choice options are unordered symbols; a distance term is meaningless."""
        collator = DecisionCollator(batching_tokenizer, max_seq_len=512)
        assert collator([example(choice(4))]).ordinal.tolist() == [False]

    def test_a_near_miss_on_a_noul_costs_less_than_the_opposite(self, config):
        config.brier_weight = 0.0
        config.ordinal_weight = 1.0
        truth = torch.tensor([8])
        logits = lambda peak: torch.tensor(  # noqa: E731
            [[10.0 if i == peak else 0.0 for i in range(9)]]
        )
        near = decision_loss(logits(6), truth, config, ordinal=torch.tensor([True]))
        far = decision_loss(logits(0), truth, config, ordinal=torch.tensor([True]))
        assert near < far


class TestProgressLog:
    """A silent run is indistinguishable from a hung one."""

    def make(self, capsys, total=100, every=10):
        from lev.train.loop import ProgressLog

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
        assert "tok/s" in out
        assert "0 tok/s" not in out.replace(",", "")

    def test_hms_formats_hours(self):
        from lev.train.loop import _hms

        assert _hms(0) == "0:00:00"
        assert _hms(59) == "0:00:59"
        assert _hms(3661) == "1:01:01"


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

        rows = [example(choice(4), target=i % 4, state=f"row {i}") for i in range(64)]
        saved: list[int] = []

        monkeypatch.setattr(loop, "prepare_data", lambda c, d: {"train": rows, "calibration": []})
        monkeypatch.setattr(
            loop, "build_model", lambda c, mc=None: (StubModel(), make_batching_tokenizer())
        )
        monkeypatch.setattr(loop, "build_head", lambda c, m=None: None)
        monkeypatch.setattr(loop, "_device_of", lambda m: torch.device("cpu"))
        monkeypatch.setattr(loop, "_to_device", lambda b, d: b)
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
