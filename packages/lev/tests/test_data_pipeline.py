"""The data pipeline: registry, sampling, splits, and the guards around them.

Everything here runs offline: `load_source` takes an injected `load_dataset`, so
a network would only add a slower test that fails when HF is down. Nothing here
checks that the real dataset ids still resolve -- a dead or renamed id surfaces
on the next `lev data build`.
"""

from __future__ import annotations

import pytest
from conftest import a_choice, an_example
from lev.data.mixture import Example, MixtureSpec, build_mixture
from lev.data.sources import (
    MODE_B_SOURCES,
    REGISTRY,
    SINGLE_TOKEN_CODE_CEILING,
    build_question,
    default_weights,
    humanise,
    load_source,
)
from lev.data.splits import (
    DEFAULT_FRACTIONS,
    Split,
    assign,
    check_coverage,
    row_key,
    split_examples,
)
from lev.labels import NOUL_RATING_TOKENS
from lev.prompt import Layout
from lev.router import candidate_count


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
        # Size-proportional weighting would give Mode B a few percent and it
        # would not learn. This asserts the deliberate over-weighting survives.
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
        """The bug this guards: `imdb[:400]` is 400 negative reviews.

        A head slice of a label-sorted corpus yields one class, trains a prior on
        the label, and is invisible afterwards because every split drawn from it
        is skewed identically.
        """
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
        """A Noul is read out as nine rating tokens, not two options.

        Passing the raw class through would supervise "yes" as rating 1, which
        `noul_probability` reads back as P(yes)=0.125 -- the model learns to say
        no on every positive example and the loss looks fine.
        """
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
        """`hash()` is salted per process; a split built on it would not reproduce."""
        assert assign(row_key("s", 7, "some text")) == assign(row_key("s", 7, "some text"))
        assert assign("lev") is assign("lev")

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
        """The check a per-label split cannot make on its own.

        If a sample only ever contains 2 of 4 options, both are in train, the
        first check passes, and the mixture is still junk.
        """
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
        """Flagging an answerable row `abstain` teaches doubt where there is none.

        The question has to become genuinely unanswerable, which means taking the
        state away, not labelling the row differently.
        """
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


class TestModeBCeiling:
    def test_the_ceiling_matches_the_alphabet(self):
        assert SINGLE_TOKEN_CODE_CEILING == 26


class TestAbstainDonors:
    """Adjacent sources cannot supply an abstain state.

    "A different source" is not the same as "a source that cannot answer this".
    imdb and rotten_tomatoes are both movie reviews, so swapping one for the
    other leaves the question perfectly answerable -- and it gets labelled
    uniform, which is the exact mislabelling the augmentation exists to avoid.
    """

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
