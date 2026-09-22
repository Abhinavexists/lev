"""The decision engine: prefill once, fork per question, read logits, never generate.

    state ──> prefill ──> KV cache (8 full-attn layers) + conv state (24 linear)
                             │
                 ┌───────────┼───────────┐        fork, batched in one forward
                 ▼           ▼           ▼
              q1 suffix   q2 suffix   qN suffix
                 │           │           │
                 ▼           ▼           ▼
             logits at the "Answer:" position, restricted to candidates
                 │
                 ▼
         temperature (per type and mode) ──> softmax ──> typed answer

Why the fork is cheap here specifically: Qwen3.5 is a hybrid, so only 8 of its 32
layers hold a K/V cache. The other 24 carry small conv/recurrent state. Forking a
full-attention model's cache N ways is what makes naive implementations slow.

Verified end-to-end on `Qwen/Qwen3.5-4B-Base` and `-0.8B-Base`: both modes answer
Choice, Score and Noul in one prefill with `output_tokens == 0`. A Mode B question
needs a trained candidate-path head; without one, `system_one` refuses up front
rather than answering badly.

The cache fork was the one part that could not be reasoned about -- see `_fork`.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

from .calibrate import CalibrationProfile
from .labels import LABEL_OPTION_CAP, noul_probability
from .prompt import Layout, Rendered, build, candidate_texts, schema_block
from .router import BINARY_NOUL, Mode, Route, route_all
from .types import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
)


@dataclass
class EngineConfig:
    model_id: str = "Qwen/Qwen3.5-4B-Base"
    # Same cap as training (`TrainConfig.max_label_options`), for the same
    # reason: above it the model is in a regime it was never trained for.
    max_label_options: int | None = LABEL_OPTION_CAP
    layout: Layout = Layout.STATE_FIRST
    device: str = "auto"
    dtype: str = "bfloat16"
    # "rating" is the trained 0-8 scale; "binary" is two lettered options for a
    # checkpoint that was never trained on the scale (ADR-007).
    noul_readout: Literal["rating", "binary"] = "rating"
    # Read every Mode A question in two option orders and average. Cancels the
    # letter-position bias that reordering exposed on massive-en-US (argmax
    # agreement 0.15 between two orders of the same 60 options). Two suffix rows
    # instead of one, in the same batched forward -- no extra prefill.
    order_average: bool = True


@dataclass(frozen=True)
class Variant:
    """One rendering of one question: the candidate order shown, and its suffix."""

    name: str
    order: list[int] | None
    suffix_ids: list[int]


def average_orders(prob_rows: list[list[float]], orders: list[list[int] | None]) -> list[float]:
    """Map each rendered-order distribution back to canonical order and average.

    Rendered position `j` showed candidate `order[j]`, so `probs[j]` belongs to
    canonical slot `order[j]`. Averaging probabilities rather than logits keeps
    the result a distribution without renormalising.
    """
    if len(prob_rows) != len(orders):
        raise ValueError("one order per probability row")
    k = len(prob_rows[0])
    total = [0.0] * k
    for probs, order in zip(prob_rows, orders, strict=True):
        for j, p in enumerate(probs):
            total[j if order is None else order[j]] += p
    return [t / len(prob_rows) for t in total]


class DecisionEngine:
    """Answers a batch of typed questions about one state in a single prefill."""

    def __init__(
        self,
        model,
        tokenizer,
        config: EngineConfig | None = None,
        calibration: CalibrationProfile | None = None,
        mode_b_head=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or EngineConfig()
        self.calibration = calibration or CalibrationProfile()
        self.mode_b_head = mode_b_head
        self._candidate_cache: dict[tuple[str, ...], Any] = {}

    def system_one(self, state, questions: dict[str, Question]) -> SystemOneResponse:
        routes = route_all(
            questions,
            self.tokenizer,
            self.config.max_label_options,
            noul_binary=self.config.noul_readout == "binary",
        )

        unsupported = [
            name
            for name, r in routes.items()
            if r.mode is Mode.CANDIDATE_PATH and self.mode_b_head is None
        ]
        if unsupported:
            raise RuntimeError(
                f"questions {unsupported} need Mode B, but no candidate-path head is "
                "loaded. Either load one, or set max_label_options to keep every "
                "question in Mode A."
            )

        prefix, variants = self._render(state, questions, routes)
        want_hidden = any(r.mode is Mode.CANDIDATE_PATH for r in routes.values())
        suffixes = [v.suffix_ids for v in variants]
        prefix_ids, logits, last_positions, hidden = self._forward(prefix, suffixes, want_hidden)

        # Calibrate each rendered variant, then average per question in the
        # question's own candidate order.
        probs_by_question: dict[str, list[list[float]]] = defaultdict(list)
        orders_by_question: dict[str, list[list[int] | None]] = defaultdict(list)
        for row, variant in enumerate(variants):
            question, route = questions[variant.name], routes[variant.name]
            scores = self._scores(logits, last_positions, row, route, question, hidden)
            probs = self.calibration.apply(scores, _bucket_type(question, route), route.mode.value)
            probs_by_question[variant.name].append(probs)
            orders_by_question[variant.name].append(variant.order)

        answers = {
            name: self._to_answer(
                question,
                routes[name],
                average_orders(probs_by_question[name], orders_by_question[name]),
            )
            for name, question in questions.items()
        }

        return SystemOneResponse(
            model=self.config.model_id,
            answers=answers,
            usage=Usage(
                input_tokens=len(prefix_ids) + sum(len(s) for s in suffixes),
                output_tokens=0,
                cached_input_tokens=len(prefix_ids),
            ),
        )

    def _orders(self, question: Question, route: Route) -> list[list[int] | None]:
        """The candidate orders to render a question in. `None` is its own order.

        Two orders only where the order is arbitrary and position bias can act:
        Choice and binary Noul, under lettered codes, in the state-first layout
        (schema-first puts the options in the shared prefix, so a second order
        would mean a second prefix). Ordered readouts are never reversed: a
        Score's levels run low to high and the rating scale's digits carry
        meaning, and training presents both in that order only -- a reversed
        scale is a prompt the model has never seen. Measured: reversing Score
        cost 2.4 points on helpsteer2.
        """
        if (
            not self.config.order_average
            or route.mode is not Mode.LABEL_TOKEN
            or self.config.layout is not Layout.STATE_FIRST
            or isinstance(question, Score)
            or (isinstance(question, Noul) and route.reason != BINARY_NOUL)
        ):
            return [None]
        n = len(route.codes or [])
        return [None, list(reversed(range(n)))] if n >= 2 else [None]

    def _render(
        self, state, questions: dict[str, Question], routes: dict[str, Route]
    ) -> tuple[str, list[Variant]]:
        codes = {name: route.codes for name, route in routes.items()}
        cached_schema = (
            schema_block(questions, codes) if self.config.layout is Layout.SCHEMA_FIRST else None
        )
        rendered: list[tuple[str, list[int] | None, Rendered]] = [
            (
                name,
                order,
                build(state, name, question, codes[name], self.config.layout, cached_schema, order),
            )
            for name, question in questions.items()
            for order in self._orders(question, routes[name])
        ]
        # Every variant shares one prefix by construction; assert it, because a
        # silent mismatch would make the cache wrong rather than merely slow.
        prefixes = {r.prefix for _, _, r in rendered}
        if len(prefixes) != 1:
            raise AssertionError(f"layout produced {len(prefixes)} prefixes, expected 1")
        return rendered[0][2].prefix, [
            Variant(name, order, self.tokenizer.encode(r.suffix, add_special_tokens=False))
            for name, order, r in rendered
        ]

    def _forward(self, prefix: str, suffixes: list[list[int]], want_hidden: bool = False):
        """Prefill the prefix once, then run all suffixes against forked caches."""
        import torch

        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=True)
        device = self.model.device

        with torch.no_grad():
            prefilled = self.model(
                input_ids=torch.tensor([prefix_ids], device=device), use_cache=True
            )
            cache = prefilled.past_key_values

            width = max(len(s) for s in suffixes)
            pad = self.tokenizer.pad_token_id or 0
            batch = torch.tensor([s + [pad] * (width - len(s)) for s in suffixes], device=device)
            attention = torch.tensor(
                [[1] * len(s) + [0] * (width - len(s)) for s in suffixes], device=device
            )
            # The forked cache must cover the prefix for every row.
            full_attention = torch.cat(
                [torch.ones(len(suffixes), len(prefix_ids), device=device), attention],
                dim=1,
            )

            out = self.model(
                input_ids=batch,
                attention_mask=full_attention,
                past_key_values=_fork(cache, len(suffixes), device),
                use_cache=False,
                # Only when a Mode B question is present: hidden states for a
                # full batch are large, and Mode A never looks at them.
                output_hidden_states=want_hidden,
            )

        last_positions = torch.tensor([len(s) - 1 for s in suffixes], device=device)
        hidden = out.hidden_states[-1] if want_hidden else None
        return prefix_ids, out.logits, last_positions, hidden

    def _scores(
        self,
        logits,
        last_positions,
        row: int,
        route: Route,
        question: Question,
        hidden,
    ) -> list[float]:
        """Raw candidate scores for one question, by whichever mode it routed to."""
        if route.mode is Mode.LABEL_TOKEN:
            return self._label_token_scores(logits, last_positions, row, route)
        return self._candidate_path_scores(hidden, last_positions, row, question)

    def _label_token_scores(self, logits, last_positions, row: int, route: Route) -> list[float]:
        from .readout.mode_a import LabelTokenReadout

        candidate_ids = LabelTokenReadout(self.tokenizer).candidate_ids(route.codes)
        return logits[row, int(last_positions[row]), candidate_ids].float().tolist()

    def _candidate_path_scores(
        self, hidden, last_positions, row: int, question: Question
    ) -> list[float]:
        if self.mode_b_head is None:
            raise RuntimeError("Mode B question reached the readout with no head loaded")
        if hidden is None:
            raise RuntimeError("Mode B needs hidden states; _forward was not asked for them")

        import torch

        question_repr = hidden[row, int(last_positions[row])].unsqueeze(0)  # (1, H)
        candidate_repr = self._candidate_reprs(candidate_texts(question)).unsqueeze(0)  # (1, K, H)
        head_dtype = next(self.mode_b_head.parameters()).dtype
        with torch.no_grad():
            scores = self.mode_b_head(
                question_repr.to(head_dtype), candidate_repr.to(head_dtype), None
            )
        return scores[0].float().tolist()

    def _candidate_reprs(self, texts: list[str]):
        """One hidden vector per candidate string, from a single batched forward.

        Cached per candidate set: the option list belongs to the question, not
        to the request, so a served schema re-encodes its candidates once rather
        than on every call.

        The cache is unbounded. That is fine for a server with a fixed set of
        schemas -- 151 candidates at hidden 2560 is ~1.5 MB -- and wrong for one
        accepting arbitrary caller-supplied option sets. Bound it before doing
        the latter.
        """
        import torch

        key = tuple(texts)
        if key in self._candidate_cache:
            return self._candidate_cache[key]

        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
            add_special_tokens=False,
        )
        device = self.model.device
        ids = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            states = self.model(
                input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False
            ).hidden_states[-1]
        last_positions = mask.sum(dim=1).long() - 1
        reprs = states[torch.arange(states.size(0), device=device), last_positions]
        self._candidate_cache[key] = reprs
        return reprs

    def _to_answer(self, question: Question, route: Route, probs: list[float]):
        """Shape a calibrated distribution as the question's answer type."""
        confidence = _gini(probs)

        if isinstance(question, Choice):
            distribution = dict(zip(question.criteria, probs, strict=True))
            return ChoiceAnswer(
                choice=max(distribution, key=distribution.get),
                probabilities=distribution,
                confidence=confidence,
            )

        if isinstance(question, Score):
            by_level = dict(enumerate(probs))
            return ScoreAnswer(
                score=sum(level * p for level, p in by_level.items()),
                probabilities=by_level,
                legend=dict(enumerate(question.criteria)),
                confidence=confidence,
            )

        if isinstance(question, Noul):
            if route.reason == BINARY_NOUL:
                # Two lettered options, yes first. No rating distribution exists
                # to report, so `probabilities` stays absent rather than faked.
                return NoulAnswer(noul=probs[0], probabilities=None, confidence=confidence)
            by_rating = dict(enumerate(probs))
            return NoulAnswer(
                noul=noul_probability(by_rating),
                probabilities=by_rating,
                confidence=confidence,
            )

        raise TypeError(f"unknown question type {question.type!r}")


def _bucket_type(question: Question, route: Route) -> str:
    """Calibration bucket. A binary Noul must not borrow the rating scale's
    temperature: the two readouts have nothing in common but the answer type."""
    return "noul_binary" if route.reason == BINARY_NOUL else question.type


def _fork(cache, n: int, device=None):
    """Expand a batch-1 prefix cache to `n` rows, one per question.

    `batch_repeat_interleave` is the obvious API and it is wrong here: it only
    exists on full-attention layers. Qwen3.5 is a hybrid -- 24 of its 32 layers
    are `LinearAttentionLayer`, which holds `conv_states`/`recurrent_states`
    rather than keys/values and raises `AttributeError` on that call. The hybrid
    split is the reason we chose this backbone, so the fork has to handle it.

    `reorder_cache` is defined on `CacheLayerMixin`, so *every* layer type
    implements it, and each one indexes its own state correctly. Selecting index
    0 `n` times turns one row into `n` -- `index_select` expands, it does not
    merely permute.

    The cache is deep-copied first because `reorder_cache` mutates in place, and
    the prefix cache must stay reusable for the next request.
    """
    import torch

    forked = copy.deepcopy(cache)
    rows = torch.zeros(n, dtype=torch.long, device=device)
    forked.reorder_cache(rows)
    return forked


def _gini(probs: list[float]) -> float:
    """Normalised Gini concentration: (K*sum(p^2) - 1) / (K - 1).

    LitJev's choice, and the leading hypothesis for Jev's own `confidence`.
    `levbench confidence` tests that hypothesis against the live API. Uniform -> 0,
    point mass -> 1.
    """
    k = len(probs)
    if k <= 1:
        return 1.0
    return (k * sum(p * p for p in probs) - 1.0) / (k - 1)
