"""Collation: padding, candidate sets, mode routing and length bucketing.

torch is in the `train` extra, so these skip on a bare `uv sync`.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from conftest import a_choice, an_example  # noqa: E402
from lev.router import Mode  # noqa: E402
from lev.train.collate import IGNORE_INDEX, DecisionCollator, ModeBatcher  # noqa: E402
from lev.types import Noul, Score  # noqa: E402


class TestCollator:
    @pytest.fixture
    def collator(self, batching_tokenizer):
        return DecisionCollator(batching_tokenizer, max_seq_len=512)

    def test_last_position_is_the_final_real_token_not_seq_minus_one(self, collator):
        """Rows are right-padded, so `seq - 1` reads a pad token's logits.

        That trains on noise and never raises.
        """
        batch = collator(
            [an_example(a_choice(), state="short"), an_example(a_choice(), state="x" * 200)]
        )
        assert batch.last_positions[0] < batch.last_positions[1]
        for row, position in enumerate(batch.last_positions):
            assert batch.attention_mask[row, position] == 1
            if position + 1 < batch.attention_mask.size(1):
                assert batch.attention_mask[row, position + 1] == 0

    def test_ragged_candidate_sets_are_masked_not_truncated(self, collator):
        batch = collator([an_example(a_choice(2)), an_example(a_choice(5))])
        assert batch.candidate_mask.shape == (2, 5)
        assert batch.candidate_mask[0].tolist() == [False, False, True, True, True]
        assert batch.candidate_mask[1].tolist() == [False] * 5

    def test_noul_gets_nine_rating_slots_not_two(self, collator):
        batch = collator([an_example(Noul(instructions="is it?"), target=8)])
        assert batch.candidate_mask.shape[1] == 9
        assert batch.targets.tolist() == [8]

    def test_abstain_rows_carry_ignore_index_and_a_soft_row(self, collator):
        batch = collator(
            [
                an_example(a_choice(4), target=1),
                an_example(a_choice(4), abstain=True, soft_target=[0.25] * 4),
            ]
        )
        assert batch.targets.tolist() == [1, IGNORE_INDEX]
        assert batch.soft_targets[1].tolist() == [0.25] * 4

    def test_no_soft_tensor_is_built_when_nothing_abstains(self, collator):
        assert collator([an_example(a_choice())]).soft_targets is None

    def test_soft_target_of_the_wrong_width_raises(self, collator):
        with pytest.raises(ValueError, match="soft_target has"):
            collator([an_example(a_choice(4), abstain=True, soft_target=[0.5, 0.5])])

    def test_score_rows_are_flagged_ordinal(self, collator):
        score = Score(instructions="how much", criteria=["low", "mid", "high"])
        batch = collator([an_example(score), an_example(a_choice())])
        assert batch.ordinal.tolist() == [True, False]

    def test_a_small_option_set_routes_to_mode_a_with_token_ids(self, collator):
        batch = collator([an_example(a_choice(4))])
        assert batch.mode is Mode.LABEL_TOKEN
        assert batch.candidate_token_ids is not None
        assert batch.candidate_input_ids is None

    def test_a_large_option_set_routes_to_mode_b_with_a_candidate_pool(self, collator):
        batch = collator([an_example(a_choice(40))])
        assert batch.mode is Mode.CANDIDATE_PATH
        assert batch.candidate_token_ids is None
        assert batch.candidate_index.shape == (1, 40)
        # 40 distinct option strings, pooled once rather than per row.
        assert batch.candidate_input_ids.shape[0] == 40

    def test_the_candidate_pool_is_shared_across_rows(self, collator):
        """A 77-option question must not cost 8x77 encodes in a batch of 8."""
        rows = [an_example(a_choice(40), target=i) for i in range(4)]
        batch = collator(rows)
        assert batch.candidate_input_ids.shape[0] == 40, "pool grew with batch size"

    def test_mixing_modes_in_one_batch_raises(self, collator):
        with pytest.raises(ValueError, match="mixes readout modes"):
            collator([an_example(a_choice(4)), an_example(a_choice(40))])

    def test_empty_batch_raises(self, collator):
        with pytest.raises(ValueError, match="empty batch"):
            collator([])


class TestModeBatcher:
    def test_batches_are_homogeneous_and_correctly_sized(self, batching_tokenizer):
        batcher = ModeBatcher(batching_tokenizer, batch_size=3)
        rows = [an_example(a_choice(4 if i % 2 else 40), source=f"s{i % 2}") for i in range(12)]
        groups = list(batcher(rows))
        assert all(len(g) <= 3 for g in groups)
        for group in groups:
            assert len({batcher.route_for(e).mode for e in group}) == 1
        assert sum(len(g) for g in groups) == 12

    def test_routes_are_resolved_once_per_question_not_per_row(self, batching_tokenizer):
        """Re-tokenising an option set for each of 200k rows is the slowest
        thing in the pipeline, and it produces the same answer every time."""
        batcher = ModeBatcher(batching_tokenizer, batch_size=2)
        rows = [an_example(a_choice(4), source="same") for _ in range(50)]
        resolved = {id(batcher.route_for(row)) for row in rows}
        assert len(resolved) == 1, "the same question resolved to distinct Route objects"

    def test_batcher_and_collator_can_share_one_route_cache(self, batching_tokenizer):
        """Sharing means a question is routed once for the whole pipeline."""
        from lev.train.collate import RouteCache

        shared = RouteCache(batching_tokenizer)
        batcher = ModeBatcher(batching_tokenizer, batch_size=2, routes=shared)
        collator = DecisionCollator(batching_tokenizer, routes=shared)
        row = an_example(a_choice(4), source="same")
        assert batcher.route_for(row) is collator.routes.route_for(row)


class TestLengthBucketing:
    """A batch pads to its longest row, so compute is the rectangle.

    Measured on the real mixture at batch 32: random batching wastes 4.43x of
    every forward pass on padding, bucketing brings it to 1.43x. That is the
    difference between a 23-hour run and a 7-hour one.
    """

    def rows(self, n=512):
        import random as _r

        rng = _r.Random(0)
        # Bimodal, like the real mixture: short intents and long reviews.
        return [
            an_example(a_choice(4), target=i % 4, state="x" * rng.choice([20, 20, 20, 1200]))
            for i in range(n)
        ]

    def waste(self, batches):
        real = sum(sum(len(str(e.state)) for e in b) for b in batches)
        padded = sum(len(b) * max(len(str(e.state)) for e in b) for b in batches)
        return padded / real

    def test_bucketing_cuts_padding_waste(self, batching_tokenizer):
        rows = self.rows()
        unbucketed = ModeBatcher(batching_tokenizer, 32, bucket_window=1)
        bucketed = ModeBatcher(batching_tokenizer, 32, bucket_window=8)
        before = self.waste(list(unbucketed(rows)))
        after = self.waste(list(bucketed(rows)))
        assert before > 2.0, f"fixture is not bimodal enough to show the effect ({before:.2f})"
        # A relative claim, not an absolute one: the absolute floor depends on
        # how the window boundary falls, and what matters is the multiple of
        # compute saved. On the real mixture this is 4.43x -> 1.43x.
        assert after < before / 2, f"bucketing only got {before:.2f}x -> {after:.2f}x"
        assert after < 1.5, f"bucketing left {after:.2f}x waste"

    def test_no_example_is_lost_or_duplicated(self, batching_tokenizer):
        rows = self.rows(300)
        batched = [e for b in ModeBatcher(batching_tokenizer, 32, bucket_window=4)(rows) for e in b]
        assert len(batched) == len(rows)
        assert {id(e) for e in batched} == {id(e) for e in rows}

    def test_batches_stay_mode_homogeneous(self, batching_tokenizer):
        """Bucketing must not undo the mode grouping the collator relies on."""
        rows = self.rows(100) + [an_example(a_choice(40), target=0) for _ in range(60)]
        batcher = ModeBatcher(batching_tokenizer, 16, bucket_window=4)
        for group in batcher(rows):
            assert len({batcher.route_for(e).mode for e in group}) == 1

    def test_batch_order_differs_between_epochs(self, batching_tokenizer):
        """Length must not track step number, or the schedule correlates with it."""
        rows = self.rows(256)
        batcher = ModeBatcher(batching_tokenizer, 32, bucket_window=4)
        first = [len(str(b[0].state)) for b in batcher(rows, epoch=0)]
        second = [len(str(b[0].state)) for b in batcher(rows, epoch=1)]
        assert first != second

    def test_ordering_is_reproducible_for_a_given_epoch(self, batching_tokenizer):
        rows = self.rows(256)
        batcher = ModeBatcher(batching_tokenizer, 32, bucket_window=4)
        a = [[id(e) for e in b] for b in batcher(rows, epoch=3)]
        b = [[id(e) for e in b] for b in batcher(rows, epoch=3)]
        assert a == b
