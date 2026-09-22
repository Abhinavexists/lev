"""The lev -> levbench handoff: a file, because levbench must not import lev."""

from __future__ import annotations

import json

import pytest
from conftest import an_example
from lev.data.build import SPLIT_FILES, from_json, to_json, write_jsonl
from lev.data.export_eval import MIN_USEFUL_ITEMS, export, truth_for
from lev.data.splits import Split
from lev.types import Choice, Noul, Score


def example(question, target, index, source="src", abstain=False):
    return an_example(
        question,
        target=target,
        source=source,
        state=f"state {index}",
        abstain=abstain,
        soft_target=[0.5, 0.5] if abstain else None,
    )


CHOICE = Choice(instructions="which", criteria={"billing": None, "technical": None})
SCORE = Score(instructions="how much", criteria=["low", "mid", "high"])
NOUL = Noul(instructions="is it urgent?")


class TestRoundTrip:
    @pytest.mark.parametrize("question", [CHOICE, SCORE, NOUL])
    def test_jsonl_preserves_the_typed_question(self, tmp_path, question):
        original = example(question, 1, 0)
        path = tmp_path / "rows.jsonl"
        write_jsonl(path, [original])
        restored = from_json(json.loads(path.read_text()))
        assert restored.question == original.question
        assert (restored.state, restored.target, restored.layout) == (
            original.state,
            original.target,
            original.layout,
        )

    def test_soft_targets_survive_the_round_trip(self):
        row = example(CHOICE, 0, 0, abstain=True)
        assert from_json(to_json(row)).soft_target == [0.5, 0.5]


class TestTruthShapes:
    def test_choice_truth_is_the_option_string(self):
        assert truth_for(CHOICE, 1) == "technical"

    def test_score_truth_is_the_level_index(self):
        assert truth_for(SCORE, 2) == 2

    def test_noul_truth_collapses_the_rating_the_way_the_readout_will(self):
        """Stored as a rating 0-8, compared as a bool. Both ends must map."""
        assert truth_for(NOUL, 8) is True
        assert truth_for(NOUL, 0) is False


class TestExport:
    def write_split(self, tmp_path, rows):
        tmp_path.mkdir(parents=True, exist_ok=True)
        write_jsonl(tmp_path / SPLIT_FILES[Split.TEST], rows)
        return tmp_path

    def test_one_file_per_source_plus_an_index(self, tmp_path):
        rows = [example(CHOICE, i % 2, i, source="a") for i in range(80)]
        rows += [example(SCORE, i % 3, i, source="b") for i in range(80)]
        data = self.write_split(tmp_path / "mix", rows)
        out = tmp_path / "eval"
        index = export(data, out)

        assert set(index["sources"]) == {"a", "b"}
        assert (out / "a.json").is_file() and (out / "b.json").is_file()
        assert json.loads((out / "index.json").read_text())["total_items"] == 160

    def test_levbench_can_read_what_was_written(self, tmp_path):
        """The actual contract. levbench never imports lev; this file is the seam."""
        pytest.importorskip("levbench")
        from levbench.tasks import load_task_file

        rows = [example(CHOICE, i % 2, i, source="a") for i in range(MIN_USEFUL_ITEMS + 10)]
        out = tmp_path / "eval"
        export(self.write_split(tmp_path / "mix", rows), out)

        items, questions = load_task_file(out / "a.json")
        assert len(items) == MIN_USEFUL_ITEMS + 10
        assert questions["a"].type == "choice"
        assert set(items[0].labels) == {"a"}
        assert items[0].labels["a"] in questions["a"].criteria

    def test_abstain_rows_are_excluded(self, tmp_path):
        """Scoring them measures abstention, not accuracy."""
        rows = [example(CHOICE, i % 2, i, source="a") for i in range(MIN_USEFUL_ITEMS + 10)]
        rows += [example(CHOICE, 0, 999, source="a", abstain=True) for _ in range(20)]
        out = tmp_path / "eval"
        index = export(self.write_split(tmp_path / "mix", rows), out)
        assert index["total_items"] == MIN_USEFUL_ITEMS + 10

    def test_an_eval_too_small_to_measure_anything_raises(self, tmp_path):
        """At n=24 the interval is +/-16 points. That is not a measurement."""
        rows = [example(CHOICE, i % 2, i, source="a") for i in range(24)]
        with pytest.raises(ValueError, match="cannot tell you"):
            export(self.write_split(tmp_path / "mix", rows), tmp_path / "eval")


class TestVariableQuestions:
    def test_sources_whose_question_varies_per_row_are_skipped_and_listed(self, tmp_path):
        """A task file has one question for all its items; per-row QA options
        cannot be expressed in it, and must not be silently written under the
        first row's option set."""
        rows = [example(CHOICE, i % 2, i, source="a") for i in range(MIN_USEFUL_ITEMS + 10)]
        rows += [
            example(
                Choice(instructions="q", criteria={f"o{i}": None, "z": None}), 0, i, source="qa"
            )
            for i in range(30)
        ]
        tmp_path.mkdir(parents=True, exist_ok=True)
        write_jsonl(tmp_path / SPLIT_FILES[Split.TEST], rows)
        out = tmp_path / "eval"
        index = export(tmp_path, out)

        assert set(index["sources"]) == {"a"}
        assert index["skipped_variable_question"] == ["qa"]
        assert not (out / "qa.json").exists()
