"""Packaging a checkpoint into a release, and resolving one back."""

from __future__ import annotations

import json

import pytest
from lev.release import MODEL_CARD, RELEASE_MANIFEST, build_release, model_card
from lev.train.checkpoints import TRAINING_STATE, fetch_checkpoint, resolve_checkpoint


def fake_checkpoint(root, step=18750, calibrated=True):
    """The files `save_checkpoint` and `calibrate` leave behind, contents faked."""
    out = root / "4b"
    step_dir = out / f"step-{step}"
    step_dir.mkdir(parents=True)
    (step_dir / "adapter_model.safetensors").write_bytes(b"weights")
    (step_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-4B-Base", "r": 32})
    )
    (step_dir / "mode_b_head.pt").write_bytes(b"head")
    (step_dir / "tokenizer.json").write_text("{}")
    (step_dir / "README.md").write_text("PEFT auto card")
    (step_dir / TRAINING_STATE).write_bytes(b"optimiser state, never shipped")
    if calibrated:
        (out / "calibration.json").write_text(json.dumps({"temperatures": {"choice:A": 2.0}}))
    return out


class TestBuildRelease:
    def test_release_is_flat_self_describing_and_excludes_training_state(self, tmp_path):
        out = fake_checkpoint(tmp_path)
        release = tmp_path / "release"
        manifest = build_release(out, release, preset="4b")

        files = {p.name for p in release.iterdir()}
        assert {
            "adapter_model.safetensors",
            "adapter_config.json",
            "mode_b_head.pt",
            "tokenizer.json",
            "calibration.json",
            RELEASE_MANIFEST,
            MODEL_CARD,
        } <= files
        assert TRAINING_STATE not in files, "optimiser state is for resuming, not serving"
        assert MODEL_CARD not in manifest["files"], "the card is written, not copied"
        assert manifest["base_model"] == "Qwen/Qwen3.5-4B-Base"
        assert manifest["step"] == 18750 and manifest["preset"] == "4b"
        assert manifest["calibrated"] and manifest["mode_b_head"]
        assert "train_max_label_options" not in manifest, "no cap applies to training any more"
        assert manifest["prompt_style"] == "plain"
        assert json.loads((release / RELEASE_MANIFEST).read_text()) == manifest

    def test_prompt_style_is_recorded_in_manifest_and_card(self, tmp_path):
        out = fake_checkpoint(tmp_path)
        manifest = build_release(out, tmp_path / "release", preset="4b", prompt_style="chat")
        assert manifest["prompt_style"] == "chat"
        assert "(`chat`)" in (tmp_path / "release" / MODEL_CARD).read_text()

    def test_missing_calibration_is_recorded_not_hidden(self, tmp_path):
        out = fake_checkpoint(tmp_path, calibrated=False)
        manifest = build_release(out, tmp_path / "release", preset="4b")
        assert manifest["calibrated"] is False
        assert "Uncalibrated" in (tmp_path / "release" / MODEL_CARD).read_text()

    def test_resolves_the_newest_step_from_the_parent(self, tmp_path):
        out = fake_checkpoint(tmp_path, step=2000)
        later = out / "step-4000"
        later.mkdir()
        for f in (out / "step-2000").iterdir():
            (later / f.name).write_bytes(f.read_bytes())
        manifest = build_release(out, tmp_path / "release", preset="4b")
        assert manifest["step"] == 4000

    def test_model_card_names_the_base_and_serving_command(self, tmp_path):
        out = fake_checkpoint(tmp_path)
        manifest = build_release(out, tmp_path / "release", preset="4b", name="lev-test")
        card = model_card(manifest)
        assert card.startswith("---\nlicense: apache-2.0\nbase_model: Qwen/Qwen3.5-4B-Base")
        assert "# lev-test" in card and "lev serve --checkpoint" in card


class TestResolvingAReleaseBack:
    def test_a_flat_release_directory_is_a_checkpoint(self, tmp_path):
        out = fake_checkpoint(tmp_path)
        release = tmp_path / "release"
        build_release(out, release, preset="4b")
        assert resolve_checkpoint(release) == release

    def test_local_paths_pass_through_fetch(self, tmp_path):
        assert fetch_checkpoint(tmp_path) == tmp_path

    @pytest.mark.parametrize("spec", ["/no/such/dir", "./missing", "not-a-repo", "a/b/c"])
    def test_a_missing_path_that_is_not_a_hub_id_fails_clearly(self, spec):
        with pytest.raises(FileNotFoundError, match="neither a local path nor a Hub id"):
            fetch_checkpoint(spec)


def test_model_card_describes_routing_as_it_is_served_and_trained(tmp_path):
    """The card ships with the weights, so it must describe Mode B as training on
    the full large taxonomies and Choice temperatures as banded by option count."""
    out = fake_checkpoint(tmp_path)
    build_release(out, tmp_path / "release", preset="4b")
    card = (tmp_path / "release" / MODEL_CARD).read_text()
    assert "27+" not in card
    assert "option-count band" in card
