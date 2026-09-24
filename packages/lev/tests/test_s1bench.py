"""The evaluation door: it opens only onto blocked subsets, and it has no
route into the training mixture."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from lev.data.contamination import BLOCKED_SUBSETS, ContaminationError, assert_clean
from lev.data.s1bench import EVAL_SUBSETS, get_subset, subset_names

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

    @pytest.mark.parametrize("alias", ["tals/vitaminc", "google/boolq", "AmazonScience/massive"])
    def test_aliases_resolve_to_their_subset(self, alias: str) -> None:
        assert get_subset(alias).name in EVAL_SUBSETS

    @pytest.mark.parametrize("name", ["ag_news", "imdb", "banking77", "some/private-corpus"])
    def test_training_sources_are_refused(self, name: str) -> None:
        """The door must not become a general-purpose dataset loader."""
        with pytest.raises(ContaminationError):
            get_subset(name)

    @pytest.mark.parametrize("name", ["squad2", "pubmedqa", "multinli"])
    def test_blocked_but_unrun_subsets_have_no_loader(self, name: str) -> None:
        """Blocked, so not a ContaminationError -- but nothing to validate against."""
        with pytest.raises(KeyError, match="no.*loader|never ran"):
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


class TestValidationTargets:
    def test_only_subsets_with_a_measured_jev_score_are_listed(self) -> None:
        """A loader with no measured number cannot be validated, so it is not built."""
        assert len(EVAL_SUBSETS) == 6
        assert set(subset_names()) < BLOCKED_SUBSETS

    def test_published_and_measured_agree_except_on_aegis2(self) -> None:
        """Sets the pass criterion: ~1pp for five, looser for aegis2."""
        for name, spec in EVAL_SUBSETS.items():
            if name == "aegis2":
                assert spec.agreement_pp == pytest.approx(3.2, abs=0.05)
            else:
                assert spec.agreement_pp < 1.0, f"{name} disagrees by {spec.agreement_pp:.2f}pp"

    def test_registry_matches_the_snapshot_it_was_read_from(self) -> None:
        """Read from `data/s1bench-snapshot.json` rather than restating it, so a
        regenerated snapshot fails here instead of silently disagreeing with the
        numbers the harness validates against."""
        snapshot = json.loads((REPO_ROOT / "data" / "s1bench-snapshot.json").read_text())
        jev = next(t for t in snapshot["targets"] if t["target"] == "jev")

        for name, spec in EVAL_SUBSETS.items():
            recorded = jev["subsets"][name]
            assert spec.items == recorded["total"], f"{name} item count"
            # The registry carries the 4dp figure; the snapshot the raw ratio.
            assert spec.jev_measured == pytest.approx(recorded["acc"], abs=5e-5), f"{name} acc"
            assert spec.jev_published == pytest.approx(snapshot["published_jev"][name], abs=1e-6), (
                f"{name} published"
            )

    def test_every_subset_that_actually_ran_has_a_loader(self) -> None:
        """The six are exactly those with a measured accuracy -- not a hand-picked
        subset of them."""
        snapshot = json.loads((REPO_ROOT / "data" / "s1bench-snapshot.json").read_text())
        jev = next(t for t in snapshot["targets"] if t["target"] == "jev")
        measured = {name for name, v in jev["subsets"].items() if v.get("acc") is not None}
        assert set(EVAL_SUBSETS) == measured
