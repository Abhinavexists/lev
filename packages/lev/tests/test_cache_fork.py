"""Regression test for the hybrid cache fork (`prefix_mode="fork"`).

`Cache.batch_repeat_interleave` exists only on full-attention layers, and 24 of
Qwen3.5's 32 layers are `LinearAttentionLayer` (conv/recurrent state, not
keys/values), so a fork built on it died with

    AttributeError: 'LinearAttentionLayer' object has no attribute
                    'batch_repeat_interleave'

These tests build a mixed cache from the real layer classes, with no GPU or
model download.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
cache_utils = pytest.importorskip("transformers.cache_utils")

from lev.model import _fork  # noqa: E402

BATCH, HEADS, SEQ, DIM = 1, 2, 5, 4


def hybrid_cache():
    """A cache shaped like Qwen3.5's: full-attention and linear layers mixed."""
    full = cache_utils.DynamicLayer()
    full.update(torch.randn(BATCH, HEADS, SEQ, DIM), torch.randn(BATCH, HEADS, SEQ, DIM))

    linear = cache_utils.LinearAttentionLayer()
    linear.lazy_initialization(
        conv_states=torch.randn(BATCH, DIM, 3),
        recurrent_states=torch.randn(BATCH, HEADS, DIM, DIM),
    )
    linear.conv_states[0] = torch.randn(BATCH, DIM, 3)
    linear.recurrent_states[0] = torch.randn(BATCH, HEADS, DIM, DIM)

    return cache_utils.Cache(layers=[full, linear])


def test_fork_expands_both_layer_kinds():
    """The whole bug: a linear-attention layer must survive the fork."""
    n = 4
    forked = _fork(hybrid_cache(), n)

    full, linear = forked.layers
    assert full.keys.shape[0] == n, "full-attention keys were not expanded"
    assert full.values.shape[0] == n
    assert linear.conv_states[0].shape[0] == n, "conv state was not expanded"
    assert linear.recurrent_states[0].shape[0] == n, "recurrent state was not expanded"


def test_every_forked_row_is_a_copy_of_row_zero():
    """Rows must be identical: each question sees the same prefix."""
    original = hybrid_cache()
    source_keys = original.layers[0].keys.clone()
    source_conv = original.layers[1].conv_states[0].clone()

    forked = _fork(original, 3)
    for row in range(3):
        assert torch.equal(forked.layers[0].keys[row], source_keys[0])
        assert torch.equal(forked.layers[1].conv_states[0][row], source_conv[0])


def test_fork_does_not_mutate_the_source():
    """The prefix cache must stay reusable for the next request."""
    original = hybrid_cache()
    before_keys = original.layers[0].keys.clone()
    before_conv = original.layers[1].conv_states[0].clone()

    _fork(original, 5)

    assert original.layers[0].keys.shape[0] == BATCH, "source cache was expanded in place"
    assert torch.equal(original.layers[0].keys, before_keys)
    assert torch.equal(original.layers[1].conv_states[0], before_conv)


def test_the_old_approach_still_fails():
    """Pin the reason for the fix, so nobody 'simplifies' it back.

    If a future transformers gives `LinearAttentionLayer` a
    `batch_repeat_interleave`, this test fails and the comment in `_fork` can be
    revisited -- deliberately, rather than by accident.
    """
    linear = cache_utils.LinearAttentionLayer()
    assert not hasattr(linear, "batch_repeat_interleave"), (
        "LinearAttentionLayer now implements batch_repeat_interleave; "
        "revisit the _fork docstring in lev/model.py"
    )
