"""The data pipeline: registry, sampling, splits, and the guards around them.

Offline: `load_source` takes an injected `load_dataset`. Nothing here checks
that the real dataset ids resolve; a dead id surfaces on the next `lev data build`.
"""

from __future__ import annotations

import pytest
from conftest import a_choice, an_example
from lev.data.mixture import Example, MixtureSpec, build_mixture
from lev.data.sources import (
    MODE_B_SOURCES,
    REGISTRY,
    build_question,
    default_weights,
    humanise,
    load_source,
)
from lev.data.splits import (
    DEFAULT_FRACTIONS,
    Split,
    check_coverage,
    split_examples,
)
from lev.labels import NOUL_RATING_TOKENS
from lev.prompt import Layout
from lev.router import candidate_count
from lev.types import Choice, Noul, Score


class FakeFeature:
    def __init__(self, names):
        self.names = names


class FakeDataset:
    """Enough of a `datasets.Dataset` for the loader: rows, features, shuffle."""

    def __init__(self, rows, features):
        self.rows, self.features = rows, features

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def shuffle(self, seed):
        import random

        shuffled = list(self.rows)
        random.Random(seed).shuffle(shuffled)
        return FakeDataset(shuffled, self.features)

    def select(self, indices):
        return FakeDataset([self.rows[i] for i in indices], self.features)


def label_sorted_loader(n_labels: int, per_label: int, text_field="text", label_field="label"):
    """A corpus grouped by label -- which is how most of the real ones ship."""
    rows = [
        {text_field: f"row {label}-{i}", label_field: label}
        for label in range(n_labels)
        for i in range(per_label)
    ]

    def loader(hf_id, config=None, split=None, cache_dir=None):
        return FakeDataset(rows, {label_field: FakeFeature([f"c{i}" for i in range(n_labels)])})

    return loader


class TestRegistry:
    def test_every_primitive_is_represented(self):
        kinds = {spec.primitive for spec in REGISTRY.values()}
        assert kinds == {"choice", "score", "noul"}, (
            "a mixture missing a primitive leaves one readout untrained"
        )

    def test_mode_b_sources_exist_and_exceed_the_code_ceiling(self):
        assert MODE_B_SOURCES, "no source exercises Mode B, the differentiator"
        for name in MODE_B_SOURCES:
            assert REGISTRY[name].is_mode_b

    def test_mode_b_gets_a_material_share_of_the_mixture(self):
        weights = default_weights()
        share = sum(weights[n] for n in MODE_B_SOURCES)
        # Size-proportional weighting would starve Mode B; this keeps the
        # over-weighting.
        assert share >= 0.2, f"Mode B is only {share:.1%} of the mixture"

    def test_weights_are_a_distribution_over_known_sources(self):
        weights = default_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-9
        assert set(weights) <= set(REGISTRY)

    def test_humanise_unpacks_snake_case_intents(self):
        assert humanise("card_arrival") == "card arrival"
        assert humanise("Sci/Tech") == "Sci/Tech"


class TestQuestions:
    def test_noul_needs_exactly_two_labels(self):
        with pytest.raises(ValueError, match="exactly 2"):
            build_question(REGISTRY["imdb"], ["a", "b", "c"])

    def test_score_keeps_level_order(self):
        spec = REGISTRY["sst5"]
        question = build_question(spec, list(spec.label_names))
        assert question.criteria == list(spec.label_names)

    def test_choice_options_are_humanised(self):
        question = build_question(REGISTRY["banking77"], ["card_arrival", "age_limit"])
        assert set(question.criteria) == {"card arrival", "age limit"}


class TestSampling:
    def test_limit_samples_rather_than_truncates(self):
        """A head slice of a label-sorted corpus yields one class (`imdb[:400]` is
        400 negative reviews)."""
        spec = REGISTRY["dbpedia_14"]
        loader = label_sorted_loader(14, 100, text_field="content")
        rows = list(load_source(spec, limit=140, load_dataset=loader))
        assert len({e.target for e in rows}) == 14, "a head slice would give 2"

    def test_sampling_is_reproducible(self):
        loader = label_sorted_loader(14, 100, text_field="content")
        a = [e.state for e in load_source(REGISTRY["dbpedia_14"], 50, load_dataset=loader)]
        b = [e.state for e in load_source(REGISTRY["dbpedia_14"], 50, load_dataset=loader)]
        assert a == b

    def test_noul_labels_land_on_the_ends_of_the_rating_scale(self):
        """The raw class would supervise "yes" as rating 1, which
        `noul_probability` reads back as P(yes) = 0.125."""
        loader = label_sorted_loader(2, 20)
        rows = list(load_source(REGISTRY["imdb"], load_dataset=loader))
        assert {e.target for e in rows} == {0, len(NOUL_RATING_TOKENS) - 1}

    def test_out_of_range_label_raises(self):
        def loader(*a, **k):
            return FakeDataset([{"text": "x", "label": 9}], {"label": FakeFeature(["a", "b"])})

        with pytest.raises(ValueError, match="out of range"):
            list(load_source(REGISTRY["ag_news"], load_dataset=loader))

    def test_blank_rows_are_skipped(self):
        def loader(*a, **k):
            rows = [{"text": "  ", "label": 0}, {"text": "real", "label": 1}]
            return FakeDataset(rows, {"label": FakeFeature(["a", "b", "c", "d"])})

        assert len(list(load_source(REGISTRY["ag_news"], load_dataset=loader))) == 1

    def test_unnamed_label_feature_raises_rather_than_guessing(self):
        def loader(*a, **k):
            return FakeDataset([{"text": "x", "label": 0}], {"label": object()})

        with pytest.raises(TypeError, match="pinned rather than guessed"):
            list(load_source(REGISTRY["ag_news"], load_dataset=loader))


def make(source, target, index, n_options=4):
    """One row whose state names its own source, so donor tests can trace it."""
    return an_example(
        a_choice(n_options), target=target, source=source, state=f"{source} text {index}"
    )


class TestSplits:
    @pytest.fixture
    def examples(self):
        return [make("s", i % 4, i) for i in range(4000)]

    def test_three_splits_roughly_match_the_requested_fractions(self, examples):
        splits = split_examples(examples)
        for split, want in DEFAULT_FRACTIONS.items():
            got = len(splits[split]) / len(examples)
            assert abs(got - want) < 0.02, f"{split}: {got:.3f} vs {want}"

    def test_splits_are_disjoint(self, examples):
        splits = split_examples(examples)
        states = {s: {e.state for e in items} for s, items in splits.items()}
        assert not states[Split.TRAIN] & states[Split.TEST]
        assert not states[Split.TRAIN] & states[Split.CALIBRATION]
        assert not states[Split.CALIBRATION] & states[Split.TEST]

    def test_assignment_is_stable_across_processes(self):
        """`hash()` is salted per process, so a split built on it reshuffles on
        every rebuild. Checked in two fresh interpreters with different hash
        seeds, against assignments pinned when the split salt was fixed."""
        import os
        import subprocess
        import sys

        probe = (
            "from lev.data.splits import assign, row_key; "
            "print(' '.join(assign(row_key('s', i, f'text {i}')).value for i in range(8)))"
        )
        pinned = "test train train test calibration train test test"
        for seed in ("1", "2"):
            env = {**os.environ, "PYTHONHASHSEED": seed}
            out = subprocess.run(
                [sys.executable, "-c", probe], capture_output=True, text=True, check=True, env=env
            )
            assert out.stdout.strip() == pinned

    def test_every_label_appears_in_train(self, examples):
        report = check_coverage(split_examples(examples))
        assert not report.missing_from_train
        assert report.total == len(examples)

    def test_label_only_outside_train_raises(self):
        splits = {
            Split.TRAIN: [make("s", 0, 1)],
            Split.CALIBRATION: [],
            Split.TEST: [make("s", 3, 2)],
        }
        with pytest.raises(ValueError, match="never in it"):
            check_coverage(splits)

    def test_option_never_observed_at_all_raises(self):
        """A sample holding 2 of 4 options passes the in-train check but not this."""
        splits = {
            Split.TRAIN: [make("s", 0, 1), make("s", 1, 2)],
            Split.CALIBRATION: [],
            Split.TEST: [],
        }
        with pytest.raises(ValueError, match="never show some of the options"):
            check_coverage(splits)

    def test_noul_is_exempt_from_the_offered_option_check(self):
        """Nine rating levels, two classes in the data. Not an error."""
        from lev.types import Noul

        rows = [
            Example("t", "n", Noul(instructions="q"), t, Layout.STATE_FIRST, "imdb")
            for t in (0, 8, 0, 8)
        ]
        report = check_coverage({Split.TRAIN: rows, Split.CALIBRATION: [], Split.TEST: []})
        assert not report.unseen_labels


class TestAbstain:
    @pytest.fixture
    def mixture(self):
        sources = ["a", "b"]
        loaders = {s: (lambda s=s: [make(s, i % 4, i) for i in range(50)]) for s in sources}
        spec = MixtureSpec(
            sources=dict.fromkeys(sources, 0.5), n_examples=2000, abstain_fraction=0.2
        )
        return list(build_mixture(spec, loaders))

    def test_abstain_examples_carry_a_foreign_state(self, mixture):
        """An abstain row must be unanswerable, so its state is replaced, not just
        its label."""
        abstained = [e for e in mixture if e.abstain]
        assert abstained
        assert all(not str(e.state).startswith(e.source) for e in abstained)

    def test_abstain_targets_are_uniform_over_the_candidate_set(self, mixture):
        for example in (e for e in mixture if e.abstain):
            k = candidate_count(example.question)
            assert example.soft_target == pytest.approx([1 / k] * k)

    def test_answerable_examples_have_no_soft_target(self, mixture):
        assert all(e.soft_target is None for e in mixture if not e.abstain)

    def test_abstain_rate_is_honoured(self, mixture):
        rate = sum(e.abstain for e in mixture) / len(mixture)
        assert 0.17 < rate < 0.23


class TestLargeTaxonomyThreshold:
    def test_the_threshold_is_the_single_letter_alphabet(self):
        from lev.labels import LABEL_OPTION_CAP

        assert LABEL_OPTION_CAP == 26


class TestAbstainDonors:
    """Adjacent sources cannot supply an abstain state: an imdb question is still
    answerable from a rotten_tomatoes review."""

    def test_adjacency_is_symmetric_and_excludes_self(self):
        from lev.data.sources import adjacency_map

        mapping = adjacency_map()
        for source, related in mapping.items():
            assert source not in related
            for other in related:
                assert source in mapping[other], f"{source}/{other} adjacency is one-way"

    def test_the_sentiment_corpora_are_all_mutually_adjacent(self):
        from lev.data.sources import adjacency_map

        mapping = adjacency_map()
        assert "rotten_tomatoes" in mapping["imdb"]
        assert "yelp_review_full" in mapping["sst5"]
        assert "clinc_oos" in mapping["banking77"]

    def test_no_abstain_state_comes_from_an_adjacent_source(self):
        sources = ["imdb", "rotten_tomatoes", "ag_news"]
        adjacent = {
            "imdb": frozenset({"rotten_tomatoes"}),
            "rotten_tomatoes": frozenset({"imdb"}),
            "ag_news": frozenset(),
        }
        loaders = {s: (lambda s=s: [make(s, i % 4, i) for i in range(30)]) for s in sources}
        spec = MixtureSpec(
            sources=dict.fromkeys(sources, 1 / 3),
            n_examples=3000,
            abstain_fraction=0.4,
            adjacent=adjacent,
        )
        rows = [e for e in build_mixture(spec, loaders) if e.abstain]
        assert rows
        for row in rows:
            donor = str(row.state).split()[0]
            assert donor != row.source
            assert donor not in adjacent[row.source], (
                f"{row.source} got an abstain state from adjacent {donor}"
            )

    def test_a_source_adjacent_to_everything_still_yields_an_example(self):
        """Degrading to "any other source" beats emitting nothing."""
        sources = ["a", "b"]
        loaders = {s: (lambda s=s: [make(s, 0, i) for i in range(10)]) for s in sources}
        spec = MixtureSpec(
            sources=dict.fromkeys(sources, 0.5),
            n_examples=200,
            abstain_fraction=1.0,
            adjacent={"a": frozenset({"b"}), "b": frozenset({"a"})},
        )
        rows = list(build_mixture(spec, loaders))
        assert len(rows) == 200 and all(r.abstain for r in rows)


class TestModeBPrompt:
    def test_mode_b_does_not_emit_an_empty_option_header(self):
        """`Options:` with nothing under it costs a header and informs nothing."""
        from lev.prompt import build as build_prompt
        from lev.types import Choice, Score

        choice = build_prompt(
            "s", "q", Choice(instructions="pick", criteria={"a": None, "b": None}), None
        ).full
        assert "Options:" not in choice
        assert "2 candidates" in choice

        score = build_prompt("s", "q", Score(instructions="rate", criteria=["lo", "hi"]), None).full
        assert "Levels:" not in score

    def test_mode_a_still_lists_every_option_with_its_code(self):
        from lev.prompt import build as build_prompt
        from lev.types import Choice

        text = build_prompt(
            "s",
            "q",
            Choice(instructions="pick", criteria={"alpha": None, "beta": None}),
            ["A", "B"],
        ).full
        assert "  A: alpha" in text and "  B: beta" in text


class TestReaders:
    """Per-row readers are pure functions; drive them with the real column shapes."""

    def test_race_maps_the_answer_letter_to_an_index(self):
        import random

        from lev.data.sources import _read_race

        row = {"article": "A.", "question": "Q?", "options": ["w", "x", "y", "z"], "answer": "C"}
        state, options, target = _read_race(row, random.Random(0))
        assert options[target] == "y"
        assert state == {"passage": "A.", "question": "Q?"}

    def test_keyed_choices_match_arc_numeric_keys(self):
        import random

        from lev.data.sources import _read_keyed_choices

        row = {
            "question": "Q?",
            "choices": {"label": ["1", "2", "3"], "text": ["a", "b", "c"]},
            "answerKey": "3",
        }
        _, options, target = _read_keyed_choices("question")(row, random.Random(0))
        assert options[target] == "c"

    def test_keyed_choices_attach_context_when_present(self):
        import random

        from lev.data.sources import _read_keyed_choices

        row = {
            "question_stem": "Q?",
            "fact1": "F.",
            "choices": {"label": ["A", "B"], "text": ["a", "b"]},
            "answerKey": "B",
        }
        state, _, _ = _read_keyed_choices("question_stem", "fact1")(row, random.Random(0))
        assert state == {"question": "Q?", "context": "F."}

    def test_sciq_shuffles_but_keeps_the_gold_index_right(self):
        import random

        from lev.data.sources import _read_sciq

        row = {
            "question": "Q?",
            "support": "",
            "correct_answer": "gold",
            "distractor1": "d1",
            "distractor2": "d2",
            "distractor3": "d3",
        }
        seen_first = set()
        for seed in range(20):
            _, options, target = _read_sciq(row, random.Random(seed))
            assert options[target] == "gold"
            seen_first.add(options[0])
        assert len(seen_first) > 1, "the gold answer must not always sit first"

    def test_sciq_skips_duplicate_options(self):
        import random

        from lev.data.sources import _read_sciq

        row = {
            "question": "Q?",
            "support": "",
            "correct_answer": "x",
            "distractor1": "x",
            "distractor2": "y",
            "distractor3": "z",
        }
        assert _read_sciq(row, random.Random(0)) is None

    def test_nli_skips_rows_without_gold(self):
        import random

        from lev.data.sources import _read_nli

        assert _read_nli({"premise": "p", "hypothesis": "h", "label": -1}, random.Random(0)) is None
        _, options, target = _read_nli(
            {"premise": "p", "hypothesis": "h", "label": 2}, random.Random(0)
        )
        assert options is None and target == 2

    def test_toxigen_uses_the_dataset_threshold(self):
        import random

        from lev.data.sources import _read_toxigen

        assert _read_toxigen({"text": "t", "toxicity_human": 2.9}, random.Random(0))[2] == 0
        assert _read_toxigen({"text": "t", "toxicity_human": 3.0}, random.Random(0))[2] == 1
        assert _read_toxigen({"text": "t", "toxicity_human": None}, random.Random(0)) is None

    def test_beavertails_yes_means_safe(self):
        import random

        from lev.data.sources import _read_beavertails

        row = {"prompt": "p", "response": "r", "is_safe": True}
        assert _read_beavertails(row, random.Random(0))[2] == 1

    def test_ultrafeedback_flattens_completions_and_drops_bad_ratings(self):
        import random

        from lev.data.sources import _read_ultrafeedback

        row = {
            "instruction": "do x",
            "completions": [
                {"response": "a", "annotations": {"helpfulness": {"Rating": "5"}}},
                {"response": "b", "annotations": {"helpfulness": {"Rating": "N/A"}}},
                {"response": "", "annotations": {"helpfulness": {"Rating": "3"}}},
                {"response": "c", "annotations": {"helpfulness": {"Rating": "1"}}},
            ],
        }
        rows = _read_ultrafeedback(row, random.Random(0))
        assert [(r[0]["response"], r[2]) for r in rows] == [("a", 4), ("c", 0)]

    def test_snips_unknown_category_is_skipped(self):
        import random

        from lev.data.sources import SNIPS_INTENTS, _read_snips

        assert _read_snips({"text": "t", "category": "Nope"}, random.Random(0)) is None
        assert _read_snips({"text": "t", "category": "GetWeather"}, random.Random(0))[2] == (
            SNIPS_INTENTS.index("GetWeather")
        )

    def test_humanise_splits_camel_case(self):
        assert humanise("SearchScreeningEvent") == "Search Screening Event"
        assert humanise("card_arrival") == "card arrival"


class TestReaderSources:
    def test_per_row_options_become_per_row_questions(self):
        rows = [
            {
                "article": "A",
                "question": f"Q{i}",
                "options": ["w", "x", "y", "z"],
                "answer": "ABCD"[i % 4],
            }
            for i in range(8)
        ]

        def loader(hf_id, config=None, split=None, cache_dir=None, **kw):
            return FakeDataset(rows, {})

        examples = list(load_source(REGISTRY["race"], load_dataset=loader))
        assert len(examples) == 8
        assert all(
            isinstance(e.question, Choice) and len(e.question.criteria) == 4 for e in examples
        )
        assert [list(e.question.criteria)[e.target] for e in examples] == ["w", "x", "y", "z"] * 2

    def test_noul_reader_targets_land_on_the_rating_ends(self):
        rows = [{"text1": "a", "text2": "b", "label": 1}, {"text1": "c", "text2": "d", "label": 0}]

        def loader(hf_id, config=None, split=None, cache_dir=None, **kw):
            return FakeDataset(rows, {})

        examples = list(load_source(REGISTRY["mrpc"], load_dataset=loader))
        assert [e.target for e in examples] == [len(NOUL_RATING_TOKENS) - 1, 0]
        assert examples[0].state == {"sentence1": "a", "sentence2": "b"}

    def test_every_noul_source_carries_a_negation(self):
        """The aegis2 failure: Noul learned yes = good because no training
        question ever made yes the bad outcome."""
        for name, spec in REGISTRY.items():
            if spec.primitive == "noul":
                assert spec.negations, f"{name} has no negated question"

    def test_large_taxonomy_sources_keep_full_sets_half_the_time_and_cut_deep(self):
        from lev.data.sources import LARGE_SET_MIN_OPTIONS, augmentation_map

        aug = augmentation_map()
        for name in MODE_B_SOURCES:
            assert aug[name].min_options == LARGE_SET_MIN_OPTIONS
            assert aug[name].keep_full_fraction == 0.5
        assert aug["ag_news"].min_options == 2 and aug["ag_news"].keep_full_fraction == 0.0


class TestAugmentation:
    def choice_pool(self, n_options=10, n=40):
        q = Choice(
            instructions="canonical",
            criteria={f"opt{i}": f"desc {i}" for i in range(n_options)},
        )
        return [an_example(q, target=i % n_options, source="c", state=f"s{i}") for i in range(n)]

    def run(self, pool, augment, **knobs):
        from lev.data.mixture import Augment

        spec = MixtureSpec(
            sources={"c": 1.0},
            n_examples=600,
            abstain_fraction=knobs.pop("abstain_fraction", 0.0),
            augment={"c": Augment(**augment)},
            **knobs,
        )
        return list(build_mixture(spec, {"c": lambda: pool}))

    def test_paraphrases_are_sampled_and_canonical_survives(self):
        out = self.run(self.choice_pool(), {"paraphrases": ("p1", "p2")}, paraphrase_fraction=0.5)
        seen = {e.question.instructions for e in out}
        assert seen == {"canonical", "p1", "p2"}

    def test_subsampling_keeps_the_gold_option_and_reindexes(self):
        out = self.run(self.choice_pool(), {"min_options": 2}, subsample_fraction=1.0)
        sizes = {len(e.question.criteria) for e in out}
        assert min(sizes) >= 2 and max(sizes) < 10, sizes
        for e in out:
            key = list(e.question.criteria)[e.target]
            original_index = int(key.removeprefix("opt"))
            assert original_index == int(e.state.removeprefix("s")) % 10

    def test_mode_b_floor_is_respected(self):
        out = self.run(self.choice_pool(n_options=30), {"min_options": 27}, subsample_fraction=1.0)
        assert all(len(e.question.criteria) >= 27 for e in out)

    def test_descriptions_are_sometimes_withheld(self):
        out = self.run(self.choice_pool(), {}, description_dropout=0.5, subsample_fraction=0.0)
        with_desc = [e for e in out if any(v for v in e.question.criteria.values())]
        without = [e for e in out if not any(v for v in e.question.criteria.values())]
        assert with_desc and without

    def test_order_is_shuffled_but_shuffle_is_not_universal(self):
        out = self.run(self.choice_pool(), {}, subsample_fraction=0.0, shuffle_fraction=0.5)
        canonical = [f"opt{i}" for i in range(10)]
        orders = [list(e.question.criteria) for e in out]
        assert any(o == canonical for o in orders) and any(o != canonical for o in orders)

    def test_negation_flips_the_noul_target(self):
        q = Noul(instructions="is it good?")
        top = len(NOUL_RATING_TOKENS) - 1
        pool = [
            an_example(q, target=top if i % 2 else 0, source="c", state=f"s{i}") for i in range(40)
        ]
        out = self.run(pool, {"negations": ("is it bad?",)}, negate_fraction=1.0)
        for e in out:
            assert e.question.instructions == "is it bad?"
            original = top if int(e.state.removeprefix("s")) % 2 else 0
            assert e.target == top - original

    def test_negation_never_touches_abstain_rows(self):
        q = Noul(instructions="is it good?")
        pool = [an_example(q, target=8, source="c", state=f"s{i}") for i in range(40)]
        pool += [an_example(a_choice(3), target=0, source="d", state=f"d{i}") for i in range(40)]
        from lev.data.mixture import Augment

        spec = MixtureSpec(
            sources={"c": 0.5, "d": 0.5},
            n_examples=400,
            abstain_fraction=0.5,
            negate_fraction=1.0,
            augment={"c": Augment(negations=("is it bad?",))},
        )
        out = list(build_mixture(spec, {"c": lambda: pool[:40], "d": lambda: pool[40:]}))
        for e in out:
            if e.source == "c" and e.abstain:
                assert e.question.instructions == "is it good?"
                assert e.soft_target is not None and len(e.soft_target) == len(NOUL_RATING_TOKENS)

    def test_abstain_soft_target_matches_the_subsampled_candidate_count(self):
        out = self.run(
            self.choice_pool(), {"min_options": 2}, subsample_fraction=1.0, abstain_fraction=0.5
        )
        for e in out:
            if e.abstain:
                assert len(e.soft_target) == len(e.question.criteria)

    def test_scores_keep_their_level_order(self):
        q = Score(instructions="how", criteria=["low", "mid", "high"])
        pool = [an_example(q, target=i % 3, source="c", state=f"s{i}") for i in range(30)]
        out = self.run(pool, {"paraphrases": ("p",)}, subsample_fraction=1.0, shuffle_fraction=1.0)
        assert all(e.question.criteria == ["low", "mid", "high"] for e in out)

    def test_no_augment_entry_means_canonical_questions(self):
        pool = self.choice_pool()
        spec = MixtureSpec(sources={"c": 1.0}, n_examples=100)
        out = list(build_mixture(spec, {"c": lambda: pool}))
        assert all(e.question is pool[0].question for e in out)


class TestPairAndEvidenceReaders:
    def test_nli_fever_uses_the_string_label_not_the_integer(self):
        import random

        from lev.data.sources import FEVER_LABELS, _read_nli_fever

        row = {
            "premise": "claim",
            "hypothesis": "evidence",
            "fever_gold_label": "REFUTES",
            "label": 2,
        }
        state, options, target = _read_nli_fever(row, random.Random(0))
        assert state == {"claim": "claim", "evidence": "evidence"}
        assert options is None and FEVER_LABELS[target] == "REFUTES"
        assert _read_nli_fever({**row, "fever_gold_label": "weird"}, random.Random(0)) is None

    def test_parade_binary_label_is_the_paraphrase_flag(self):
        import random

        from lev.data.sources import _read_parade

        row = {"Definition1": "a", "Definition2": "b", "Binary labels": 1, "Four-class labels": 3}
        assert _read_parade(row, random.Random(0))[2] == 1

    def test_strategyqa_state_carries_the_facts(self):
        import random

        from lev.data.sources import _read_strategyqa

        row = {"question": "q?", "facts": "f.", "answer": False}
        state, _, target = _read_strategyqa(row, random.Random(0))
        assert state == {"question": "q?", "facts": "f."} and target == 0

    def test_swapping_two_words_keeps_every_token_and_changes_the_order(self):
        import random

        from lev.data.sources import swap_two_words

        text = "the quick brown foxes jumped over lazy river dogs"
        swapped = swap_two_words(text, random.Random(3))
        assert swapped is not None and swapped != text
        assert sorted(swapped.split()) == sorted(text.split())
        assert swap_two_words("no swap", random.Random(0)) is None

    def test_adversarial_pairs_add_a_negative_for_positives_only(self):
        import random

        from lev.data.sources import _read_pair

        reader = _read_pair("text1", "text2", adversarial=1.0)
        positive = {
            "text1": "alpha beta gamma delta",
            "text2": "alpha beta gamma delta epsilon",
            "label": 1,
        }
        rows = reader(positive, random.Random(0))
        assert [r[2] for r in rows] == [1, 0]
        assert rows[1][0]["sentence1"] == positive["text1"]
        assert sorted(rows[1][0]["sentence2"].split()) == sorted(positive["text2"].split())
        negative = {**positive, "label": 0}
        assert [r[2] for r in reader(negative, random.Random(0))] == [0]

    def test_yes_no_wrapper_emits_one_true_and_one_false_per_row(self):
        import random

        from lev.data.sources import _read_race, yes_no_from_choices

        row = {"article": "A.", "question": "Q?", "options": ["w", "x", "y", "z"], "answer": "C"}
        rows = yes_no_from_choices(_read_race)(row, random.Random(0))
        assert [r[2] for r in rows] == [1, 0]
        assert rows[0][0]["proposed_answer"] == "y" and rows[1][0]["proposed_answer"] != "y"
        assert rows[0][0]["passage"] == "A." and rows[0][1] is None


class TestKeepFull:
    def test_large_taxonomy_rows_split_between_full_and_cut_sets(self):
        from lev.data.mixture import Augment

        q = Choice(instructions="c", criteria={f"o{i}": None for i in range(40)})
        pool = [an_example(q, target=i % 40, source="big", state=f"s{i}") for i in range(40)]
        spec = MixtureSpec(
            sources={"big": 1.0},
            n_examples=400,
            subsample_fraction=1.0,
            augment={"big": Augment(min_options=15, keep_full_fraction=0.5)},
        )
        sizes = [len(e.question.criteria) for e in build_mixture(spec, {"big": lambda: pool})]
        full = sum(s == 40 for s in sizes)
        assert 120 < full < 280, f"about half should keep the full set, got {full}/400"
        assert (
            min(sizes) >= 15
            and any(15 <= s <= 26 for s in sizes)
            and any(27 <= s < 40 for s in sizes)
        )


class TestBuildDataset:
    """`build_dataset` end to end: two fake sources in, three split files out."""

    @staticmethod
    def loader(hf_id, config=None, split=None, cache_dir=None, **_):
        if hf_id == REGISTRY["ag_news"].hf_id:
            rows = [{"text": f"news story {i}", "label": i % 4} for i in range(400)]
            return FakeDataset(
                rows, {"label": FakeFeature(["World", "Sports", "Business", "Sci/Tech"])}
            )
        if hf_id == REGISTRY["imdb"].hf_id:
            rows = [{"text": f"film review {i}", "label": i % 2} for i in range(400)]
            return FakeDataset(rows, {"label": FakeFeature(["neg", "pos"])})
        raise AssertionError(f"unexpected source {hf_id}")

    def test_every_split_draws_from_every_source(self, tmp_path):
        """The loaders are built in a loop; a late-bound closure would make every
        split draw from the last source while coverage and round-trip checks pass."""
        from lev.data.build import SPLIT_FILES, build_dataset, read_jsonl

        manifest = build_dataset(
            tmp_path,
            limit_per_source=None,
            n_examples=300,
            sources={"ag_news": 0.5, "imdb": 0.5},
            loader=self.loader,
        )
        prefix = {"ag_news": "news story", "imdb": "film review"}
        for split, filename in SPLIT_FILES.items():
            rows = read_jsonl(tmp_path / filename)
            assert {e.source for e in rows} == {"ag_news", "imdb"}, split.value
            # A row is labelled with the pool it was drawn from, so a crossed
            # loader leaves the labels right and the content wrong. Check both.
            for e in rows:
                if not e.abstain:
                    assert e.state.startswith(prefix[e.source]), (
                        f"{split.value}: {e.source} row holds {e.state!r}"
                    )
        assert manifest["split_counts"]["train"] == 300
        assert set(manifest["sources"]) == {"ag_news", "imdb"}

    def test_held_out_splits_keep_the_canonical_question(self, tmp_path):
        """The test split keeps each source's canonical wording, since an exported
        eval needs one question per source."""
        from lev.data.build import SPLIT_FILES, build_dataset, read_jsonl
        from lev.data.splits import Split

        build_dataset(
            tmp_path,
            limit_per_source=None,
            n_examples=300,
            sources={"ag_news": 0.5, "imdb": 0.5},
            loader=self.loader,
        )
        test_rows = read_jsonl(tmp_path / SPLIT_FILES[Split.TEST])
        for source in ("ag_news", "imdb"):
            wordings = {e.question.instructions for e in test_rows if e.source == source}
            assert wordings == {REGISTRY[source].instructions}
