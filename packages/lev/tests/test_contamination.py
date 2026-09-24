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
            ("rajpurkar/squad_v2", "squad2"),
            ("nyu-mll/multi_nli", "multinli"),
            ("google/civil_comments", "civil_comments"),
            ("qiaojin/PubMedQA", "pubmedqa"),
            ("mteb/summeval", "summeval-relevance"),
            # Re-hosted copies, caught by alias segment matching.
            ("SetFit/amazon_massive_intent_en-US", "massive-en-US"),
            ("SetFit/amazon_massive_scenario_en-US", "massive-en-US"),
            ("nvidia/Aegis-AI-Content-Safety-Dataset-1.0", "aegis2"),
            ("someone/my-squad-v2-mirror", "squad2"),
            # Same intent schema as MASSIVE, via SLURP: blocked for the taxonomy.
            ("DeepPavlov/hwu64", "massive-en-US"),
        ],
    )
    def test_aliases_and_variants_resolve(self, alias, expected):
        assert resolve(alias) == expected

    @pytest.mark.parametrize(
        "innocent",
        [
            "PolyAI/banking77",
            "fancyzhx/ag_news",
            "SetFit/sst5",
            "my-org/internal-tickets",
            # Training sources: must stay clean under alias segment matching.
            "stanfordnlp/snli",
            "facebook/anli",
            "SetFit/qqp",
            "SetFit/mrpc",
            "lmsys/toxic-chat",
            "toxigen/toxigen-data",
            "PKU-Alignment/BeaverTails",
            "openbmb/UltraFeedback",
            "benayas/snips",
            "ehovy/race",
            "tau/commonsense_qa",
            "allenai/sciq",
            "allenai/openbookqa",
            "allenai/ai2_arc",
        ],
    )
    def test_innocent_datasets_pass(self, innocent):
        assert resolve(innocent) is None

    def test_subsets_that_did_not_run_are_blocked_too(self):
        """The 7 subsets `s1-fast` skipped are still evaluation data: easy to
        forget, since the completed-run leaderboard never mentions them."""
        unrun = [
            "massive-de-DE",
            "squad2",
            "multinli",
            "civil_comments",
            "summeval-relevance",
            "summeval-consistency",
            "pubmedqa",
        ]
        assert set(unrun) <= BLOCKED_SUBSETS
        for name in unrun:
            with pytest.raises(ContaminationError):
                assert_clean([name])

    def test_the_block_list_covers_every_published_subset(self):
        """Tripwire against the snapshot: if a subset appears there, block it."""
        import json
        from pathlib import Path as _P

        snapshot = _P(__file__).resolve().parents[3] / "data" / "s1bench-snapshot.json"
        published = set(json.loads(snapshot.read_text())["published_jev"])
        assert published <= BLOCKED_SUBSETS, sorted(published - BLOCKED_SUBSETS)


class TestGuard:
    def test_clean_mixture_passes(self):
        assert_clean(["PolyAI/banking77", "fancyzhx/ag_news", "nanojev-events"])

    def test_contaminated_mixture_raises(self):
        with pytest.raises(ContaminationError) as exc:
            assert_clean(["fancyzhx/ag_news", "tals/vitaminc"])
        assert "vitaminc-dev" in str(exc.value)
        assert "ARCHITECTURE.md" in str(exc.value), "the error must say where to read why"

    def test_reports_every_collision_not_just_the_first(self):
        hits = check_mixture(["boolq", "paws-x", "dair-ai/emotion", "nvidia/HelpSteer2"])
        assert set(hits.values()) == {"boolq", "paws", "helpsteer2"}
        assert "dair-ai/emotion" not in hits

    def test_empty_mixture_is_clean(self):
        assert_clean([])
