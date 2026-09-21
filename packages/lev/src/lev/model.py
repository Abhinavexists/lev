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
from dataclasses import dataclass
from typing import Any

from .calibrate import CalibrationProfile
from .labels import noul_probability
from .prompt import Layout, build, candidate_texts, schema_block
from .router import Mode, Route, route_all
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
    max_label_options: int | None = None
    layout: Layout = Layout.STATE_FIRST
    device: str = "auto"
    dtype: str = "bfloat16"


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
        routes = route_all(questions, self.tokenizer, self.config.max_label_options)

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

        prefix, suffixes = self._render(state, questions, routes)
        want_hidden = any(r.mode is Mode.CANDIDATE_PATH for r in routes.values())
        prefix_ids, logits, last_positions, hidden = self._forward(prefix, suffixes, want_hidden)

        answers = {}
        for i, (name, question) in enumerate(questions.items()):
            scores = self._scores(logits, last_positions, i, routes[name], question, hidden)
            answers[name] = self._to_answer(question, routes[name], scores)

        return SystemOneResponse(
            model=self.config.model_id,
            answers=answers,
            usage=Usage(
                input_tokens=len(prefix_ids) + sum(len(s) for s in suffixes),
                output_tokens=0,
                cached_input_tokens=len(prefix_ids),
            ),
        )

    def _render(self, state, questions: dict[str, Question], routes: dict[str, Route]):
        codes = {name: route.codes for name, route in routes.items()}
        cached_schema = (
            schema_block(questions, codes) if self.config.layout is Layout.SCHEMA_FIRST else None
        )
        rendered = [
            build(state, name, question, routes[name].codes, self.config.layout, cached_schema)
            for name, question in questions.items()
        ]
        # Every question shares one prefix by construction; assert it, because a
        # silent mismatch would make the cache wrong rather than merely slow.
        prefixes = {r.prefix for r in rendered}
        if len(prefixes) != 1:
            raise AssertionError(f"layout produced {len(prefixes)} prefixes, expected 1")
        return rendered[0].prefix, [
            self.tokenizer.encode(r.suffix, add_special_tokens=False) for r in rendered
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

    def _to_answer(self, question: Question, route: Route, scores: list[float]):
        """Calibrate the raw scores, then shape them as the question's answer type."""
        probs = self.calibration.apply(scores, question.type, route.mode.value)
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
            by_rating = dict(enumerate(probs))
            return NoulAnswer(
                noul=noul_probability(by_rating),
                probabilities=by_rating,
                confidence=confidence,
            )

        raise TypeError(f"unknown question type {question.type!r}")


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
