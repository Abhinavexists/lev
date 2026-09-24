"""The training objective: cross-entropy, Brier and the ordinal term.

torch is in the `train` extra, so these skip on a bare `uv sync`.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from conftest import a_choice, an_example  # noqa: E402
from lev.train.collate import IGNORE_INDEX, DecisionCollator  # noqa: E402
from lev.train.loop import decision_loss  # noqa: E402
from lev.types import Noul  # noqa: E402


class TestDecisionLoss:
    def test_padded_slots_do_not_produce_nan(self, train_config):
        """The first smoke run's bug: a padded slot has logit -inf and target 0,
        so a plain product is `0 * -inf` = NaN, poisoning the batch mean."""
        logits = torch.tensor(
            [[1.0, 2.0, 0.5, 0.1], [0.3, 1.2, float("-inf"), float("-inf")]],
            requires_grad=True,
        )
        loss = decision_loss(logits, torch.tensor([1, 0]), train_config)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(logits.grad).all()

    def test_matches_cross_entropy_when_only_ce_is_enabled(self, train_config):
        """Supporting soft targets must not change the hard-target answer."""
        import torch.nn.functional as F

        train_config.brier_weight = 0.0
        train_config.ordinal_weight = 0.0
        logits, targets = torch.randn(8, 5), torch.randint(0, 5, (8,))
        assert torch.allclose(
            decision_loss(logits, targets, train_config),
            F.cross_entropy(logits, targets),
            atol=1e-6,
        )

    def test_brier_penalises_overconfidence_at_equal_accuracy(self, train_config):
        """Why Brier is in the loss at all: CE alone optimises the argmax."""
        train_config.brier_weight = 1.0
        train_config.ordinal_weight = 0.0
        targets = torch.tensor([0])
        confident = decision_loss(torch.tensor([[8.0, 0.0, 0.0]]), targets, train_config)
        modest = decision_loss(torch.tensor([[2.0, 0.0, 0.0]]), targets, train_config)
        # Both are correct. The overconfident one must still be preferred *less*
        # than it would be under CE alone -- checked by comparing the gap.
        train_config.brier_weight = 0.0
        ce_gap = modest - decision_loss(torch.tensor([[8.0, 0.0, 0.0]]), targets, train_config)
        train_config.brier_weight = 1.0
        assert (modest - confident) < ce_gap

    def test_uniform_soft_target_is_minimised_by_a_uniform_prediction(self, train_config):
        train_config.brier_weight = 0.0
        train_config.ordinal_weight = 0.0
        soft = torch.full((1, 4), 0.25)
        targets = torch.tensor([IGNORE_INDEX])
        flat = decision_loss(torch.zeros(1, 4), targets, train_config, soft_targets=soft)
        peaked = decision_loss(
            torch.tensor([[5.0, 0.0, 0.0, 0.0]]), targets, train_config, soft_targets=soft
        )
        assert flat < peaked

    def test_hard_and_abstain_rows_coexist_in_one_batch(self, train_config):
        logits = torch.tensor([[1.0, 2.0, 0.5, 0.1], [0.3, 1.2, 0.0, 0.0]], requires_grad=True)
        soft = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]])
        loss = decision_loss(
            logits, torch.tensor([1, IGNORE_INDEX]), train_config, soft_targets=soft
        )
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()

    def test_ordinal_term_prefers_a_near_miss(self, train_config):
        train_config.brier_weight = 0.0
        train_config.ordinal_weight = 2.0
        targets = torch.tensor([4])
        near = decision_loss(torch.tensor([[0.0, 0.0, 0.0, 5.0, 0.0]]), targets, train_config, True)
        far = decision_loss(torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0]]), targets, train_config, True)
        assert near < far, "an ordered scale must not treat every wrong level alike"

    def test_ordinal_applies_only_to_the_rows_flagged(self, train_config):
        """`ordinal` is per-row: a Score and a Choice can share a batch."""
        train_config.ordinal_weight = 5.0
        logits = torch.randn(4, 5)
        targets = torch.tensor([0, 1, 2, 3])
        none = decision_loss(logits, targets, train_config, torch.zeros(4, dtype=torch.bool))
        some = decision_loss(
            logits, targets, train_config, torch.tensor([True, False, False, False])
        )
        assert not torch.isclose(none, some)


class TestOrdinalCoverage:
    def test_noul_rows_are_flagged_ordinal(self, batching_tokenizer):
        """The scale runs "0 = certainly no" to "8 = certainly yes": without the
        ordinal term, rating 4 is as wrong as rating 0 when the truth is 8."""
        collator = DecisionCollator(batching_tokenizer, max_seq_len=512)
        batch = collator([an_example(Noul(instructions="urgent?"), target=8)])
        assert batch.ordinal.tolist() == [True]

    def test_choice_rows_are_not_flagged_ordinal(self, batching_tokenizer):
        """Choice options are unordered symbols; a distance term is meaningless."""
        collator = DecisionCollator(batching_tokenizer, max_seq_len=512)
        assert collator([an_example(a_choice(4))]).ordinal.tolist() == [False]

    def test_a_near_miss_on_a_noul_costs_less_than_the_opposite(self, train_config):
        train_config.brier_weight = 0.0
        train_config.ordinal_weight = 1.0
        truth = torch.tensor([8])
        logits = lambda peak: torch.tensor(  # noqa: E731
            [[10.0 if i == peak else 0.0 for i in range(9)]]
        )
        near = decision_loss(logits(6), truth, train_config, ordinal=torch.tensor([True]))
        far = decision_loss(logits(0), truth, train_config, ordinal=torch.tensor([True]))
        assert near < far
