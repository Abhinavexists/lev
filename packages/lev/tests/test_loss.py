"""Decision-loss tests; skip when Torch is unavailable."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from conftest import a_choice, an_example  # noqa: E402
from lev.train.collate import IGNORE_INDEX, DecisionCollator  # noqa: E402
from lev.train.loop import decision_loss  # noqa: E402
from lev.types import Noul  # noqa: E402


class TestDecisionLoss:
    def test_padded_slots_do_not_produce_nan(self, train_config):
        """Padded slots must avoid 0 * -inf, which produces NaN."""
        logits = torch.tensor(
            [[1.0, 2.0, 0.5, 0.1], [0.3, 1.2, float("-inf"), float("-inf")]],
            requires_grad=True,
        )
        loss = decision_loss(logits, torch.tensor([1, 0]), train_config)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(logits.grad).all()

    def test_matches_cross_entropy_when_only_ce_is_enabled(self, train_config):
        import torch.nn.functional as F

        train_config.brier_weight = 0.0
        train_config.ordinal_weight = 0.0
        logits, targets = torch.randn(8, 5), torch.randint(0, 5, (8,))
        assert torch.allclose(
            decision_loss(logits, targets, train_config),
            F.cross_entropy(logits, targets),
            atol=1e-6,
        )

    @pytest.mark.parametrize("weight", [0.5, 1.0])
    def test_brier_adds_the_weighted_multiclass_brier_score(self, train_config, weight):
        logits = torch.tensor(
            [[8.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, float("-inf")]],
        )
        targets = torch.tensor([0, 1, 0])
        train_config.ordinal_weight = 0.0
        train_config.brier_weight = 0.0
        ce_only = decision_loss(logits, targets, train_config)
        train_config.brier_weight = weight
        with_brier = decision_loss(logits, targets, train_config)

        probs = logits.softmax(dim=-1)
        one_hot = torch.nn.functional.one_hot(targets, 3).float()
        expected_brier = ((probs - one_hot) ** 2).sum(dim=-1).mean()
        assert torch.allclose(with_brier - ce_only, weight * expected_brier, atol=1e-6)
        assert expected_brier > 0.5, "the wrong rows must give the term a non-trivial size"

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
        near_logits = torch.tensor([[0.0, 0.0, 0.0, 5.0, 0.0]])
        far_logits = torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0]])
        near = decision_loss(near_logits, targets, train_config, ordinal=True)
        far = decision_loss(far_logits, targets, train_config, ordinal=True)
        assert near < far, "an ordered scale must not treat every wrong level alike"

    def test_ordinal_applies_only_to_the_rows_flagged(self, train_config):
        train_config.ordinal_weight = 5.0
        logits = torch.randn(4, 5)
        targets = torch.tensor([0, 1, 2, 3])
        none = decision_loss(
            logits, targets, train_config, ordinal=torch.zeros(4, dtype=torch.bool)
        )
        some = decision_loss(
            logits, targets, train_config, ordinal=torch.tensor([True, False, False, False])
        )
        assert not torch.isclose(none, some)


class TestOrdinalCoverage:
    def test_noul_rows_are_flagged_ordinal(self, batching_tokenizer):
        collator = DecisionCollator(batching_tokenizer, max_seq_len=512)
        batch = collator([an_example(Noul(instructions="urgent?"), target=8)])
        assert batch.ordinal.tolist() == [True]

    def test_choice_rows_are_not_flagged_ordinal(self, batching_tokenizer):
        collator = DecisionCollator(batching_tokenizer, max_seq_len=512)
        assert collator([an_example(a_choice(4))]).ordinal.tolist() == [False]

    def test_a_near_miss_on_a_noul_costs_less_than_the_opposite(self, train_config):
        train_config.brier_weight = 0.0
        train_config.ordinal_weight = 1.0
        truth = torch.tensor([8])

        def peaked_at(rating):
            return torch.tensor([[10.0 if i == rating else 0.0 for i in range(9)]])

        near = decision_loss(peaked_at(6), truth, train_config, ordinal=torch.tensor([True]))
        far = decision_loss(peaked_at(0), truth, train_config, ordinal=torch.tensor([True]))
        assert near < far
