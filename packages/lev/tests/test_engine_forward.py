"""Pin last-token projection to the full forward of a tiny offline Qwen3.5 with both layer kinds."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
qwen = pytest.importorskip("transformers.models.qwen3_5")

from lev.model import DecisionEngine, EngineConfig  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    config = qwen.Qwen3_5TextConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
    )
    return qwen.Qwen3_5ForCausalLM(config).eval()


@pytest.mark.parametrize("want_hidden", [False, True])
def test_single_forward_matches_the_full_logits_at_each_last_token(tiny_model, want_hidden):
    engine = DecisionEngine(tiny_model, SimpleNamespace(pad_token_id=0), EngineConfig())
    rows = [[5, 6, 7, 8, 9, 10], [5, 6, 7], [5, 6, 7, 8]]  # right padding differs per row

    logits, hidden = engine._forward_single(rows, want_hidden)

    for i, row in enumerate(rows):
        with torch.no_grad():
            full = tiny_model(input_ids=torch.tensor([row]), output_hidden_states=True)
        torch.testing.assert_close(logits[i], full.logits[0, -1])
        if want_hidden:
            torch.testing.assert_close(hidden[i], full.hidden_states[-1][0, -1])
    assert logits.shape == (len(rows), 512), "one vocabulary row per variant, not per position"
    assert (hidden is not None) == want_hidden
