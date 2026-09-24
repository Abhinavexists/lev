"""Tests for everything that runs without a GPU: schema, labels, router, prompts."""

from __future__ import annotations

import pytest
from lev.labels import (
    LABEL_OPTION_CAP,
    NOUL_RATING_TOKENS,
    label_codes,
    noul_probability,
    single_token_codes,
)
from lev.model import average_orders
from lev.prompt import (
    ANSWER_CUE,
    CHAT_TAIL,
    Layout,
    Style,
    build,
    label_prefix,
    render_question,
    render_state,
    schema_block,
)
from lev.router import BINARY_NOUL, Mode, route, route_all
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

    def test_codes_cover_the_three_letter_boundary_and_final_code(self):
        codes = label_codes(18_278)
        assert codes[701:703] == ["ZZ", "AAA"]
        assert codes[-1] == "ZZZ"
        assert len(set(codes)) == 18_278

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


class TestOptionCap:
    def test_generous_tokenizer_still_routes_above_the_cap_to_mode_b(self):
        """An explicit `max_label_options` forces Mode B even when the tokenizer
        can express every code (Qwen3.5 encodes codes up to `BP` as one token)."""
        from string import ascii_uppercase

        from conftest import FakeTokenizer

        pairs = {f" {a}{b}" for a in ascii_uppercase for b in ascii_uppercase}
        generous = FakeTokenizer({f" {c}" for c in ascii_uppercase} | pairs)
        sixty = Choice(criteria={f"o{i}": None for i in range(60)})
        assert single_token_codes(generous, 60) is not None, "fixture must be generous"
        assert (
            route(sixty, generous, max_label_options=LABEL_OPTION_CAP).mode is Mode.CANDIDATE_PATH
        )
        at_cap = Choice(criteria={f"o{i}": None for i in range(LABEL_OPTION_CAP)})
        assert route(at_cap, generous, max_label_options=LABEL_OPTION_CAP).mode is Mode.LABEL_TOKEN


class TestBinaryNoul:
    def test_binary_route_uses_two_lettered_codes(self, rich_tokenizer):
        r = route(Noul(instructions="safe?"), rich_tokenizer, noul_binary=True)
        assert r.mode is Mode.LABEL_TOKEN
        assert r.codes == ["A", "B"]
        assert r.reason == BINARY_NOUL

    def test_default_route_is_still_the_rating_scale(self, rich_tokenizer):
        r = route(Noul(instructions="safe?"), rich_tokenizer)
        assert len(r.codes) == len(NOUL_RATING_TOKENS)

    def test_binary_render_lists_yes_then_no(self):
        text = render_question("q", Noul(instructions="Is it safe?"), ["A", "B"])
        assert "A: yes" in text and "B: no" in text
        assert "Rate 0-8" not in text

    def test_reversed_order_puts_no_first(self):
        text = render_question("q", Noul(instructions="Is it safe?"), ["A", "B"], order=[1, 0])
        assert "A: no" in text and "B: yes" in text

    def test_binary_render_carries_the_criteria(self):
        q = Noul(instructions="Is it safe?", criteria={"true": "harmless", "false": "harmful"})
        text = render_question("q", q, ["A", "B"])
        assert "A: yes - harmless" in text and "B: no - harmful" in text


class TestOrderAveraging:
    def test_reversed_rendering_assigns_codes_by_position(self):
        q = Choice(criteria={"refund": None, "replace": None, "info": None})
        forward = render_question("q", q, ["A", "B", "C"])
        backward = render_question("q", q, ["A", "B", "C"], order=[2, 1, 0])
        assert "A: refund" in forward and "C: info" in forward
        assert "A: info" in backward and "C: refund" in backward

    def test_average_maps_reversed_row_back_to_canonical_slots(self):
        forward = [0.7, 0.2, 0.1]
        backward = [0.1, 0.2, 0.7]  # rendered reversed: position 0 showed candidate 2
        merged = average_orders([forward, backward], [None, [2, 1, 0]])
        assert merged == pytest.approx([0.7, 0.2, 0.1])

    def test_average_cancels_a_first_position_bias(self):
        """A model that always adds mass to whatever is listed first sees that
        bonus land on different candidates in the two orders; averaging spreads
        it back out."""
        biased_forward = [0.6, 0.2, 0.2]  # candidate 0 first, gets the bonus
        biased_backward = [0.6, 0.2, 0.2]  # candidate 2 first, gets the bonus
        merged = average_orders([biased_forward, biased_backward], [None, [2, 1, 0]])
        assert merged == pytest.approx([0.4, 0.2, 0.4])
        assert sum(merged) == pytest.approx(1.0)

    def test_single_order_is_identity(self):
        assert average_orders([[0.25, 0.75]], [None]) == pytest.approx([0.25, 0.75])

    def test_build_passes_order_through_state_first(self):
        q = Choice(criteria={"a": None, "b": None})
        r = build("state", "q", q, ["A", "B"], Layout.STATE_FIRST, order=[1, 0])
        assert "A: b" in r.suffix and r.suffix.endswith(ANSWER_CUE)


class TestWhichQuestionsGetTwoOrders:
    def orders(self, question, tokenizer, **config):
        from lev.model import DecisionEngine, EngineConfig

        engine = DecisionEngine(model=None, tokenizer=tokenizer, config=EngineConfig(**config))
        return engine._orders(
            question, route(question, tokenizer, noul_binary=config.get("noul_readout") == "binary")
        )

    def test_choice_is_read_both_ways(self, rich_tokenizer):
        assert self.orders(Choice(criteria={"a": None, "b": None, "c": None}), rich_tokenizer) == [
            None,
            [2, 1, 0],
        ]

    def test_binary_noul_is_read_both_ways(self, rich_tokenizer):
        assert self.orders(Noul(instructions="?"), rich_tokenizer, noul_readout="binary") == [
            None,
            [1, 0],
        ]

    def test_score_keeps_its_level_order(self, rich_tokenizer):
        """Levels run low to high and training never reorders them; a reversed
        scale is a prompt the model has not seen (helpsteer2: -2.4 points)."""
        assert self.orders(Score(criteria=["low", "mid", "high"]), rich_tokenizer) == [None]

    def test_rating_noul_keeps_its_scale(self, rich_tokenizer):
        assert self.orders(Noul(instructions="?"), rich_tokenizer) == [None]

    def test_disabled_by_config(self, rich_tokenizer):
        q = Choice(criteria={"a": None, "b": None})
        assert self.orders(q, rich_tokenizer, order_average=False) == [None]

    def test_schema_first_never_doubles_the_prefix(self, rich_tokenizer):
        q = Choice(criteria={"a": None, "b": None})
        assert self.orders(q, rich_tokenizer, layout=Layout.SCHEMA_FIRST) == [None]


class TestShapeBuckets:
    def test_width_rounds_up_to_the_multiple(self):
        from lev.model import bucket

        assert bucket(1, 32) == 32 and bucket(32, 32) == 32 and bucket(33, 32) == 64

    def test_rows_round_up_to_a_power_of_two(self):
        from lev.model import pow2

        assert [pow2(n) for n in (1, 2, 3, 5, 8, 9)] == [1, 2, 4, 8, 8, 16]


class TestChatStyle:
    def test_state_first_wraps_evidence_in_the_user_turn_and_opens_the_assistant(self):
        q = Choice(instructions="Which team?", criteria={"billing": "money", "tech": None})
        r = build("the ticket", "q", q, ["A", "B"], Layout.STATE_FIRST, style=Style.CHAT)
        assert r.prefix.startswith("<|im_start|>system\n")
        assert "<|im_start|>user\n# Evidence\nthe ticket\n\n" in r.prefix
        assert "# Criterion\nWhich team?" in r.suffix
        assert "A. billing: money" in r.suffix and "B. tech" in r.suffix
        assert "Respond with only the letter of the best option." in r.suffix
        assert r.suffix.endswith(CHAT_TAIL) and r.full.count("<|im_end|>") == 2

    def test_schema_first_prefix_is_state_independent_in_chat_too(self):
        q = Choice(instructions="Which team?", criteria={"a": None, "b": None})
        a = build("s1", "q", q, ["A", "B"], Layout.SCHEMA_FIRST, style=Style.CHAT)
        b = build("s2", "q", q, ["A", "B"], Layout.SCHEMA_FIRST, style=Style.CHAT)
        assert a.prefix == b.prefix and a.suffix != b.suffix
        assert a.suffix.startswith("# Evidence\ns1") and a.suffix.endswith(CHAT_TAIL)

    def test_score_levels_are_numbered_and_noul_binary_is_a_lettered_pair(self):
        s = render_question(
            "s", Score(criteria=["low", "mid", "high"]), ["A", "B", "C"], style=Style.CHAT
        )
        assert "A. (level 0 of 2) low" in s and "C. (level 2 of 2) high" in s
        n = render_question(
            "n",
            Noul(instructions="Safe?", criteria={"true": "harmless"}),
            ["A", "B"],
            style=Style.CHAT,
        )
        assert "A. yes: harmless" in n and "B. no: the statement is false" in n
        rating = render_question(
            "n", Noul(instructions="Safe?"), [str(i) for i in range(9)], style=Style.CHAT
        )
        assert "# Scale" in rating and "Respond with only a digit from 0 to 8." in rating

    def test_reversed_order_applies_in_chat_style(self):
        q = Choice(criteria={"x": None, "y": None, "z": None})
        text = render_question("q", q, ["A", "B", "C"], order=[2, 1, 0], style=Style.CHAT)
        assert "A. z" in text and "C. x" in text

    def test_label_prefix_is_bare_for_chat_and_space_for_plain(
        self, rich_tokenizer, chat_tokenizer
    ):
        assert label_prefix(Style.PLAIN) == " " and label_prefix(Style.CHAT) == ""
        q = Choice(criteria={"a": None, "b": None, "c": None})
        assert route(q, chat_tokenizer, label_prefix="").codes == ["A", "B", "C"]
        # A tokenizer with only space-prefixed letters cannot express bare codes:
        # the chat style falls through to Mode B there rather than misreading.
        assert route(q, rich_tokenizer, label_prefix="").mode is Mode.CANDIDATE_PATH

    def test_plain_style_is_unchanged(self):
        q = Choice(instructions="Which team?", criteria={"a": None, "b": None})
        r = build("state", "q", q, ["A", "B"])
        assert r.prefix == "Context:\nstate\n\n" and r.suffix.endswith(ANSWER_CUE)
        assert "<|im_start|>" not in r.full


class TestSkipMultiTokenCodes:
    def tokenizer_without(self, broken: str):
        from string import ascii_uppercase

        from conftest import FakeTokenizer

        pairs = {f" {a}{b}" for a in ascii_uppercase for b in ascii_uppercase} - {f" {broken}"}
        return FakeTokenizer({f" {c}" for c in ascii_uppercase} | pairs)

    def test_default_scheme_stops_at_the_first_split_code(self):
        tok = self.tokenizer_without("BQ")  # the 69th code, as in Qwen3.5
        assert single_token_codes(tok, 68) is not None
        assert single_token_codes(tok, 69) is None

    def test_skipping_passes_over_split_codes_and_keeps_the_order(self):
        tok = self.tokenizer_without("BQ")
        codes = single_token_codes(tok, 77, skip_multi_token=True)
        assert len(codes) == 77 and "BQ" not in codes and len(set(codes)) == 77
        assert codes[:68] == single_token_codes(tok, 68), "codes below the split are unchanged"
        assert codes[68] == "BR"

    def test_route_uses_mode_a_past_the_split_only_when_asked(self):
        tok = self.tokenizer_without("BQ")
        q = Choice(criteria={f"o{i}": None for i in range(77)})
        assert route(q, tok).mode is Mode.CANDIDATE_PATH
        assert route(q, tok, skip_multi_token=True).mode is Mode.LABEL_TOKEN
        assert (
            route(q, tok, max_label_options=76, skip_multi_token=True).mode is Mode.CANDIDATE_PATH
        )


class TestRoutePreviewMatchesServing:
    def test_lev_route_previews_the_mode_the_server_uses(self, tmp_path, monkeypatch, capsys):
        """A 60-option Choice is Mode A when served, since the tokenizer can
        express 60 codes, so the preview must say Mode A too."""
        import json
        import sys
        import types
        from string import ascii_uppercase

        from conftest import FakeTokenizer
        from lev.cli import main

        pairs = {f" {a}{b}" for a in ascii_uppercase for b in ascii_uppercase}
        tokenizer = FakeTokenizer({f" {c}" for c in ascii_uppercase} | pairs)
        stub = types.ModuleType("transformers")
        stub.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *_a, **_k: tokenizer)
        monkeypatch.setitem(sys.modules, "transformers", stub)

        request = {
            "state": "turn the lights off",
            "questions": {
                "intent": {"type": "choice", "criteria": {f"o{i}": None for i in range(60)}}
            },
        }
        path = tmp_path / "request.json"
        path.write_text(json.dumps(request))

        main(["route", str(path)])
        assert "mode A" in capsys.readouterr().out

    def test_lev_route_prints_a_mode_b_route_without_codes(self, tmp_path, monkeypatch, capsys):
        """A Mode B route has no label codes to print."""
        import json
        import sys
        import types

        from conftest import FakeTokenizer
        from lev.cli import main

        tokenizer = FakeTokenizer({" A", " B"})  # only two codes are single tokens
        stub = types.ModuleType("transformers")
        stub.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *_a, **_k: tokenizer)
        monkeypatch.setitem(sys.modules, "transformers", stub)
        request = {
            "state": "s",
            "questions": {"q": {"type": "choice", "criteria": {"a": None, "b": None, "c": None}}},
        }
        path = tmp_path / "request.json"
        path.write_text(json.dumps(request))

        main(["route", str(path)])
        assert "mode B" in capsys.readouterr().out


class TestSystemOneAcceptsPlainDicts:
    """`lev.load` hands users an engine; they pass questions as JSON-shaped dicts."""

    def engine(self, tokenizer, **config):
        from lev.model import DecisionEngine, EngineConfig

        return DecisionEngine(model=None, tokenizer=tokenizer, config=EngineConfig(**config))

    def test_a_dict_question_is_parsed_and_routed(self, rich_tokenizer):
        # Capped at two label options with no head loaded, a three-option Choice
        # needs Mode B, so routing refuses before any forward pass is attempted.
        engine = self.engine(rich_tokenizer, max_label_options=2)
        questions = {"pick": {"type": "choice", "criteria": {"a": None, "b": None, "c": None}}}
        with pytest.raises(RuntimeError, match="need Mode B"):
            engine.system_one("state", questions)

    def test_a_malformed_dict_is_a_value_error(self, rich_tokenizer):
        """ValueError is what the server turns into a 422."""
        engine = self.engine(rich_tokenizer)
        with pytest.raises(ValueError, match="at least 2 options"):
            engine.system_one("state", {"pick": {"type": "choice", "criteria": {"a": None}}})
