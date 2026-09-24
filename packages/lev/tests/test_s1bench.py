"""The evaluation door: it opens only onto blocked subsets, and it has no
route into the training mixture."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from lev.data import s1bench
from lev.data.contamination import BLOCKED_SUBSETS, ContaminationError, assert_clean
from lev.data.s1bench import EVAL_SUBSETS, EvalItem, definition, get_subset, load_eval_subset

REPO_ROOT = Path(__file__).resolve().parents[3]


class TestTrainingDoorStaysShut:
    @pytest.mark.parametrize("subset", sorted(BLOCKED_SUBSETS))
    def test_every_blocked_subset_still_raises_on_the_training_path(self, subset: str) -> None:
        with pytest.raises(ContaminationError):
            assert_clean([subset])

    def test_adding_an_eval_door_did_not_unblock_the_hf_ids_it_reads(self) -> None:
        """The loader reads `tals/vitaminc`; the mixture still must not."""
        for spec in EVAL_SUBSETS.values():
            with pytest.raises(ContaminationError):
                assert_clean([spec.hf_id])


class TestEvalDoorOpensOnlyOnEvalData:
    @pytest.mark.parametrize("name", sorted(EVAL_SUBSETS))
    def test_loadable_subsets_resolve(self, name: str) -> None:
        assert get_subset(name).name == name

    @pytest.mark.parametrize(
        "alias", ["tals/vitaminc", "google/boolq", "AmazonScience/massive", "rajpurkar/squad_v2"]
    )
    def test_aliases_resolve_to_their_subset(self, alias: str) -> None:
        assert get_subset(alias).name in EVAL_SUBSETS

    @pytest.mark.parametrize("name", ["ag_news", "imdb", "banking77", "some/private-corpus"])
    def test_training_sources_are_refused(self, name: str) -> None:
        """The door must not become a general-purpose dataset loader."""
        with pytest.raises(ContaminationError):
            get_subset(name)


class TestStructuralSeparation:
    def test_s1bench_cannot_reach_the_mixture_builder(self) -> None:
        """Importing `mixture` (e.g. to reuse `Example`) would give the eval loaders
        a type the training path consumes. Checked in a fresh interpreter, since
        in-process the test suite's own imports are visible."""
        probe = (
            "import sys; import lev.data.s1bench; "
            "leaked = sorted(m for m in sys.modules "
            "if m in ('lev.data.mixture', 'lev.data.build', 'lev.data.sources')); "
            "print(','.join(leaked))"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        assert result.stdout.strip() == "", (
            f"lev.data.s1bench pulled in training modules: {result.stdout.strip()}"
        )


EXPECTED_TYPES = {
    "vitaminc-dev": "choice",
    "massive-en-US": "choice",
    "massive-de-DE": "choice",
    "multinli": "choice",
    "pubmedqa": "choice",
    "boolq": "noul",
    "squad2": "noul",
    "paws": "noul",
    "civil_comments": "noul",
    "aegis2": "noul",
    "helpsteer2": "score",
    "summeval-relevance": "score",
    "summeval-consistency": "score",
}


def snapshot() -> dict:
    return json.loads((REPO_ROOT / "data" / "s1bench-snapshot.json").read_text())


class TestPinnedDefinitions:
    def test_every_blocked_subset_is_loadable(self) -> None:
        assert set(EVAL_SUBSETS) == BLOCKED_SUBSETS

    @pytest.mark.parametrize("name", sorted(EVAL_SUBSETS))
    def test_definition_is_self_consistent_and_names_its_source(self, name: str) -> None:
        spec = definition(name)
        assert len(spec["ids"]) == len(set(spec["ids"])) == spec["count"]
        assert sum(spec["labels"].values()) == spec["count"]
        assert spec["source"].startswith("https://github.com/bespokelabsai/nimble/blob/")
        assert s1bench.question_for(spec).type == EXPECTED_TYPES[name]

    @pytest.mark.parametrize("name", sorted(EVAL_SUBSETS))
    def test_record_counts_match_the_board(self, name: str) -> None:
        """helpsteer2 is one record short of the board's 250: the manifest's
        token-length filter dropped rows."""
        board = next(t for t in snapshot()["targets"] if t["target"] == "jev")["subsets"][name]
        expected = board["total"] - (1 if name == "helpsteer2" else 0)
        assert definition(name)["count"] == expected

    def test_jev_numbers_match_the_snapshot(self) -> None:
        """Read from `data/s1bench-snapshot.json` rather than restated, so a
        regenerated snapshot fails here."""
        data = snapshot()
        board = next(t for t in data["targets"] if t["target"] == "jev")["subsets"]
        for name, subset in EVAL_SUBSETS.items():
            assert subset.jev_published == pytest.approx(data["published_jev"][name], abs=1e-6)
            measured = board.get(name, {}).get("acc")
            if measured is None:
                assert subset.jev_measured is None, f"{name} has no board measurement"
            else:
                assert subset.jev_measured == pytest.approx(measured, abs=5e-5), name


class TestSelection:
    """`load_eval_subset` keeps exactly the pinned ids and refuses drifted upstream data."""

    def fake(self, monkeypatch, rows, ids, labels):
        spec = {**definition("boolq"), "ids": ids, "labels": labels, "count": len(ids)}
        monkeypatch.setattr(s1bench, "definition", lambda name: spec)
        monkeypatch.setitem(s1bench._READERS, "boolq", lambda: iter(rows))

    def test_keeps_only_the_pinned_ids_sorted(self, monkeypatch) -> None:
        rows = [EvalItem("boolq-b", {"q": "2"}, False), EvalItem("boolq-a", {"q": "1"}, True)]
        rows.append(EvalItem("boolq-z", {"q": "unpinned"}, True))
        self.fake(monkeypatch, rows, ["boolq-b", "boolq-a"], {"True": 1, "False": 1})
        question, items = load_eval_subset("boolq")
        assert question.type == "noul"
        assert [item.id for item in items] == ["boolq-a", "boolq-b"]

    def test_a_missing_pinned_id_raises(self, monkeypatch) -> None:
        self.fake(monkeypatch, [EvalItem("boolq-a", {}, True)], ["boolq-a", "boolq-b"], {"True": 2})
        with pytest.raises(ValueError, match="pinned ids not in the upstream data"):
            load_eval_subset("boolq")

    def test_drifted_labels_raise(self, monkeypatch) -> None:
        self.fake(monkeypatch, [EvalItem("boolq-a", {}, False)], ["boolq-a"], {"True": 1})
        with pytest.raises(ValueError, match="label counts"):
            load_eval_subset("boolq")
