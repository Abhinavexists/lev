"""Pin the adapter-scale multiplier: relative to trained strength, and survives the merge."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
qwen = pytest.importorskip("transformers.models.qwen3_5")
peft = pytest.importorskip("peft")

from lev.model import _set_adapter_scale  # noqa: E402

TARGET = "model.layers.1.self_attn.q_proj"


def tiny_base():
    torch.manual_seed(0)
    config = qwen.Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
    )
    return qwen.Qwen3_5ForCausalLM(config).eval()


def merged_delta(scale: float | None):
    base = tiny_base()
    before = base.get_submodule(TARGET).weight.detach().clone()

    adapted = peft.get_peft_model(
        base, peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"])
    )
    # lora_B is zero-initialised, so an untouched adapter is a no-op whatever the scale.
    for name, param in adapted.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.05)

    if scale is not None:
        _set_adapter_scale(adapted, scale)
    merged = adapted.merge_and_unload()
    return merged.get_submodule(TARGET).weight.detach() - before


def test_the_merge_scales_the_adapter_linearly():
    full = merged_delta(None)
    assert full.abs().max() > 0, "the adapter must not be a no-op for this test to mean anything"
    for scale in (0.25, 0.5):
        torch.testing.assert_close(merged_delta(scale), full * scale, rtol=1e-4, atol=1e-6)


def test_scale_one_is_the_trained_strength():
    torch.testing.assert_close(merged_delta(1.0), merged_delta(None))


def test_scale_zero_leaves_the_frozen_backbone():
    assert merged_delta(0.0).abs().max() == 0


def test_scaling_a_model_without_lora_layers_raises():
    with pytest.raises(RuntimeError, match="no LoRA layers"):
        _set_adapter_scale(tiny_base(), 0.25)
