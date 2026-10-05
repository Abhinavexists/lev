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


class TinyTokenizer:
    """Ids inside the tiny vocabulary; space-prefixed A-Z are single tokens, as label codes need."""

    pad_token_id = 0
    single = {f" {c}": 400 + i for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}

    def encode(self, text, add_special_tokens=True):
        return (
            [self.single[text]] if text in self.single else [ord(c) % 300 + 2 for c in text] or [1]
        )


def test_without_score_averaging_a_large_request_answers_as_with_no_limit(tiny_model):
    """The row limit never touches a request Score averaging did not grow (ADR-029)."""
    from lev.types import Choice, Score

    questions = {
        **{
            f"c{i}": Choice(criteria={"a": None, "b": None, "c": None, "d": None})
            for i in range(20)
        },
        **{f"s{i}": Score(criteria=["low", "mid", "high"]) for i in range(40)},
    }
    limited = DecisionEngine(tiny_model, TinyTokenizer(), EngineConfig())
    unlimited = DecisionEngine(tiny_model, TinyTokenizer(), EngineConfig(max_request_rows=None))
    assert limited.prepare("state", questions).rows == 80 > limited.config.max_request_rows
    assert limited.system_one("state", questions) == unlimited.system_one("state", questions)
