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

Verified end-to-end on `Qwen/Qwen3.5-4B-Base` (CPU, transformers 5.17): Mode A
answers Choice, Score and Noul in one prefill with `output_tokens == 0`. Mode B is
written but untrained, so a question routed to it still raises.

The cache fork was the one part that could not be reasoned about -- see `_fork`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from .calibrate import CalibrationProfile
from .labels import noul_probability
from .prompt import Layout, build, schema_block
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

    # -- public API ---------------------------------------------------------

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
        prefix_ids, logits, last_positions = self._forward(prefix, suffixes)

        answers = {}
        for i, (name, question) in enumerate(questions.items()):
            scores = self._scores(logits, last_positions, i, routes[name])
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

    # -- internals ----------------------------------------------------------

    def _render(self, state, questions: dict[str, Question], routes: dict[str, Route]):
        codes = {n: r.codes for n, r in routes.items()}
        block = (
            schema_block(questions, codes) if self.config.layout is Layout.SCHEMA_FIRST else None
        )
        rendered = [
            build(state, n, q, routes[n].codes, self.config.layout, block)
            for n, q in questions.items()
        ]
        # Every question shares one prefix by construction; assert it, because a
        # silent mismatch would make the cache wrong rather than merely slow.
        prefixes = {r.prefix for r in rendered}
        if len(prefixes) != 1:
            raise AssertionError(f"layout produced {len(prefixes)} prefixes, expected 1")
        return rendered[0].prefix, [
            self.tokenizer.encode(r.suffix, add_special_tokens=False) for r in rendered
        ]

    def _forward(self, prefix: str, suffixes: list[list[int]]):
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
            )

        last_positions = torch.tensor([len(s) - 1 for s in suffixes], device=device)
        return prefix_ids, out.logits, last_positions

    def _scores(self, logits, last_positions, row: int, route: Route):
        if route.mode is Mode.LABEL_TOKEN:
            from .readout.mode_a import LabelTokenReadout

            readout = LabelTokenReadout(self.tokenizer)
            ids = readout.candidate_ids(route.codes)
            return logits[row, int(last_positions[row]), ids].float().tolist()
        raise NotImplementedError(
            "Mode B scoring runs through the trained head; see readout/mode_b.py. "
            "Wire it here once the head is trained."
        )

    def _to_answer(self, question: Question, route: Route, scores: list[float]):
        mode = route.mode.value
        kind = question.type
        probs = self.calibration.apply(scores, kind, mode)
        confidence = _gini(probs)

        if isinstance(question, Choice):
            keys = list(question.criteria)
            dist = dict(zip(keys, probs, strict=True))
            return ChoiceAnswer(
                choice=max(dist, key=dist.get), probabilities=dist, confidence=confidence
            )

        if isinstance(question, Score):
            dist = {i: p for i, p in enumerate(probs)}
            return ScoreAnswer(
                score=sum(i * p for i, p in dist.items()),
                probabilities=dist,
                legend=dict(enumerate(question.criteria)),
                confidence=confidence,
            )

        if isinstance(question, Noul):
            dist = {i: p for i, p in enumerate(probs)}
            return NoulAnswer(
                noul=noul_probability(dist), probabilities=dist, confidence=confidence
            )

        raise TypeError(f"unknown question type {kind!r}")


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
