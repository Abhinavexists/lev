"""Tests for everything that runs without a GPU: schema, labels, router, prompts."""

from __future__ import annotations

import pytest
from lev.labels import NOUL_RATING_TOKENS, label_codes, noul_probability, single_token_codes
from lev.prompt import ANSWER_CUE, Layout, build, render_state, schema_block
from lev.router import Mode, route, route_all
from lev.types import Choice, Noul, Score


class TestLabelCodes:
    def test_codes_widen_from_letters_to_pairs(self):
        assert label_codes(3) == ["A", "B", "C"]
        assert label_codes(26)[-1] == "Z"
        assert label_codes(27)[26] == "AA"
        assert label_codes(52)[51] == "AZ"

    def test_codes_are_unique(self):
        codes = label_codes(700)
        assert len(set(codes)) == len(codes)

    def test_rejects_zero_and_oversized_option_counts(self):
        with pytest.raises(ValueError):
            label_codes(0)
        with pytest.raises(ValueError):
            label_codes(100_000)

    def test_single_token_check_uses_the_space_prefix(self, rich_tokenizer):
        # The scored token follows "Answer:", so it is space-prefixed. Verifying
        # the bare code would pass while the real token differs.
        assert single_token_codes(rich_tokenizer, 5) == ["A", "B", "C", "D", "E"]
        assert single_token_codes(rich_tokenizer, 5, prefix="") is None

    def test_returns_none_when_impossible(self, poor_tokenizer):
        assert single_token_codes(poor_tokenizer, 4) is not None
        assert single_token_codes(poor_tokenizer, 5) is None


class TestNoul:
    def test_probability_is_the_scale_expectation(self):
        top = len(NOUL_RATING_TOKENS) - 1
        assert noul_probability({0: 1.0}) == 0.0
        assert noul_probability({top: 1.0}) == 1.0
        assert noul_probability({0: 0.5, top: 0.5}) == pytest.approx(0.5)

    def test_midpoint_is_a_half(self):
        assert noul_probability({4: 1.0}) == pytest.approx(0.5)


class TestRouter:
    def test_small_choice_uses_mode_a(self, rich_tokenizer):
        q = Choice(criteria={"a": None, "b": None, "c": None})
        r = route(q, rich_tokenizer)
        assert r.mode is Mode.LABEL_TOKEN
        assert r.codes == ["A", "B", "C"]

    def test_falls_through_to_mode_b_instead_of_rejecting(self, poor_tokenizer):
        """The differentiator: where other implementations reject, lev routes."""
        q = Choice(criteria={k: None for k in "abcdefgh"})
        r = route(q, poor_tokenizer)
        assert r.mode is Mode.CANDIDATE_PATH
        assert r.codes is None
        assert "single-token" in r.reason

    def test_policy_cap_forces_mode_b_below_the_tokenizer_limit(self, rich_tokenizer):
        q = Choice(criteria={k: None for k in "abcdefghij"})
        assert route(q, rich_tokenizer).mode is Mode.LABEL_TOKEN
        assert route(q, rich_tokenizer, max_label_options=5).mode is Mode.CANDIDATE_PATH

    def test_noul_routes_on_the_rating_scale(self, rich_tokenizer):
        r = route(Noul(instructions="urgent?"), rich_tokenizer)
        assert r.mode is Mode.LABEL_TOKEN
        assert len(r.codes) == len(NOUL_RATING_TOKENS)

    def test_score_routes_on_level_count(self, rich_tokenizer):
        r = route(Score(criteria=["low", "mid", "high"]), rich_tokenizer)
        assert r.mode is Mode.LABEL_TOKEN and len(r.codes) == 3

    def test_route_all_is_per_question(self, poor_tokenizer):
        routes = route_all(
            {
                "small": Choice(criteria={"a": None, "b": None}),
                "big": Choice(criteria={k: None for k in "abcdefgh"}),
            },
            poor_tokenizer,
        )
        assert routes["small"].mode is Mode.LABEL_TOKEN
        assert routes["big"].mode is Mode.CANDIDATE_PATH


class TestPrompt:
    def test_state_first_caches_the_state(self):
        r = build("the state", "q", Choice(criteria={"a": None, "b": None}), ["A", "B"])
        assert r.prefix.startswith("Context:")
        assert "the state" in r.prefix
        assert r.suffix.endswith(ANSWER_CUE)
        assert "the state" not in r.suffix

    def test_schema_first_prefix_is_state_independent(self):
        q = Choice(criteria={"a": None, "b": None})
        a = build("state one", "q", q, ["A", "B"], layout=Layout.SCHEMA_FIRST)
        b = build("state two", "q", q, ["A", "B"], layout=Layout.SCHEMA_FIRST)
        assert a.prefix == b.prefix, "schema-first prefix must not depend on the state"
        assert a.suffix != b.suffix

    def test_state_first_prefix_is_question_independent(self):
        s = "shared state"
        a = build(s, "q1", Choice(criteria={"a": None, "b": None}), ["A", "B"])
        b = build(s, "q2", Score(criteria=["lo", "hi"]), ["A", "B"])
        assert a.prefix == b.prefix, "state-first prefix must not depend on the question"

    def test_json_state_is_key_sorted(self):
        # An unsorted dump silently defeats prefix caching.
        assert render_state({"b": 1, "a": 2}) == render_state({"a": 2, "b": 1})

    def test_scored_position_is_always_last(self):
        for layout in (Layout.STATE_FIRST, Layout.SCHEMA_FIRST):
            r = build("s", "q", Noul(), [str(i) for i in range(9)], layout=layout)
            assert r.full.endswith(ANSWER_CUE)

    def test_schema_block_covers_every_question(self):
        qs = {"a": Choice(criteria={"x": None, "y": None}), "b": Score(criteria=["lo", "hi"])}
        block = schema_block(qs, {"a": ["A", "B"], "b": ["A", "B"]})
        assert "Options:" in block and "Levels:" in block


class TestSchemaValidation:
    def test_choice_needs_two_options(self):
        with pytest.raises(ValueError):
            Choice(criteria={"only": None})

    def test_score_level_bounds(self):
        with pytest.raises(ValueError):
            Score(criteria=["one"])
        with pytest.raises(ValueError):
            Score(criteria=[str(i) for i in range(11)])
        assert len(Score(criteria=["a", "b"]).criteria) == 2
