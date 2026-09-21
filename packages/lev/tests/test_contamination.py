"""The guard must catch aliases, not just exact names, and must raise not warn."""

from __future__ import annotations

import pytest
from lev.data import BLOCKED_SUBSETS, ContaminationError, assert_clean, check_mixture
from lev.data.contamination import resolve


class TestResolution:
    @pytest.mark.parametrize("name", sorted(BLOCKED_SUBSETS))
    def test_every_blocked_subset_resolves_to_itself(self, name):
        assert resolve(name) == name

    @pytest.mark.parametrize(
        ("alias", "expected"),
        [
            ("tals/vitaminc", "vitaminc-dev"),
            ("VitaminC", "vitaminc-dev"),
            ("google-research-datasets/paws", "paws"),
            ("paws-x", "paws"),
            ("nvidia/HelpSteer2", "helpsteer2"),
            ("super_glue/boolq", "boolq"),
            ("google/boolq", "boolq"),
            ("amazonscience/massive", "massive-en-US"),
            ("boolq:train", "boolq"),
            ("  BoolQ  ", "boolq"),
        ],
    )
    def test_aliases_and_variants_resolve(self, alias, expected):
        assert resolve(alias) == expected

    @pytest.mark.parametrize(
        "innocent",
        ["squad2", "multinli", "civil_comments", "pubmedqa", "my-org/internal-tickets"],
    )
    def test_innocent_datasets_pass(self, innocent):
        assert resolve(innocent) is None


class TestGuard:
    def test_clean_mixture_passes(self):
        assert_clean(["squad2", "multinli", "nanojev-events"])

    def test_contaminated_mixture_raises(self):
        with pytest.raises(ContaminationError) as exc:
            assert_clean(["squad2", "tals/vitaminc"])
        assert "vitaminc-dev" in str(exc.value)
        assert "ARCHITECTURE.md" in str(exc.value), "the error must say where to read why"

    def test_reports_every_collision_not_just_the_first(self):
        hits = check_mixture(["boolq", "paws-x", "squad2", "nvidia/HelpSteer2"])
        assert set(hits.values()) == {"boolq", "paws", "helpsteer2"}
        assert "squad2" not in hits

    def test_empty_mixture_is_clean(self):
        assert_clean([])
