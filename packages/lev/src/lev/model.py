"""Batched inference over typed questions, without token generation.

The default repeats the shared prefix in one batch. The optional fork mode
prefills it once and copies its cache for each question (ADR-023). Mode B
requires a trained head and separately encodes uncached candidate texts.
"""

from __future__ import annotations

import copy
import json
import threading
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import TypeAdapter

from .calibrate import CalibrationProfile
from .labels import noul_probability
from .prompt import Layout, Rendered, Style, build, candidate_texts, label_prefix, schema_block
from .router import BINARY_NOUL, Mode, Route, route_all
from .train.checkpoints import MODE_B_HEAD
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

_QUESTIONS = TypeAdapter(dict[str, Question])


@dataclass
class EngineConfig:
    model_id: str = "Qwen/Qwen3.5-4B-Base"
    # None uses Mode A up to the tokenizer's limit (ADR-025).
    max_label_options: int | None = None
    layout: Layout = Layout.STATE_FIRST
    # Must match training; recorded in the release manifest.
    prompt_style: Literal["plain", "chat"] = "plain"
    # Frozen checkpoints use binary Noul; trained ones use the 0-8 scale (ADR-007).
    noul_readout: Literal["rating", "binary"] = "rating"
    # A second option order reduces letter-position bias; costs one extra batch row.
    order_average: bool = True
    # Serving can skip split codes; training keeps large sets for Mode B (ADR-028).
    skip_multi_token_codes: bool = True
    # Forking avoids repeated prefix work but adds a forward and cache copies (ADR-023).
    prefix_mode: Literal["fork", "single"] = "single"
    # Off because it measured slower on this backbone (ADR-023). When enabled,
    # inputs are padded to shape buckets and warmup captures graphs at startup.
    compile: bool = False
    pad_to: int = 32


def bucket(n: int, multiple: int) -> int:
    """`n` rounded up to a multiple; a stable set of shapes for graph capture."""
    return max(multiple, ((n + multiple - 1) // multiple) * multiple)


def pow2(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


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


def serving_routes(
    questions: dict[str, Question], tokenizer, config: EngineConfig
) -> dict[str, Route]:
    """Resolve the serving routes, also exposed by `lev route`."""
    return route_all(
        questions,
        tokenizer,
        config.max_label_options,
        noul_binary=config.noul_readout == "binary",
        label_prefix=label_prefix(Style(config.prompt_style)),
        skip_multi_token=config.skip_multi_token_codes,
    )


class DecisionEngine:
    """Answers a batch of typed questions about one state in one forward pass."""

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
        self.checkpoint: Path | None = None
        self._candidate_cache: dict[tuple[str, ...], Any] = {}
        # Read once: a compiled module proxies attributes, but not reliably.
        self._device = getattr(model, "device", None)
        # One forward at a time: CUDA-graph replay is not thread-safe. Concurrent
        # requests still overlap parsing, tokenising and the network.
        self._lock = threading.Lock()
        self._eager_model = model
        if self.config.compile:
            import torch

            self.model = torch.compile(model, mode="reduce-overhead", dynamic=True)

    def warmup(
        self,
        shapes: tuple[tuple[int, int], ...] = (
            (2, 64),
            (2, 128),
            (4, 128),
            (8, 128),
            (2, 256),
            (2, 512),
        ),
    ) -> float:
        """Run one forward per shape bucket so compile and graph capture happen
        before the first request. Returns seconds spent. Falls back to eager,
        with a warning, if the compiled path fails.
        """
        import torch

        started = time.perf_counter()
        pad = self.tokenizer.pad_token_id or 0
        try:
            for rows, width in shapes:
                ids = torch.full((rows, width), pad, dtype=torch.long, device=self._device)
                mask = torch.ones_like(ids)
                with torch.no_grad(), self._lock:
                    self.model(input_ids=ids, attention_mask=mask, use_cache=False)
                    if self.mode_b_head is not None:
                        self.model(
                            input_ids=ids,
                            attention_mask=mask,
                            use_cache=False,
                            output_hidden_states=True,
                        )
        except Exception as error:  # noqa: BLE001 -- any compile failure means eager
            print(
                f"WARNING: compiled forward failed ({type(error).__name__}: {error}); serving eager"
            )
            self.model = self._eager_model
            self.config.compile = False
        return time.perf_counter() - started

    def system_one(self, state, questions: Mapping[str, Question | dict]) -> SystemOneResponse:
        """Answer every question about `state` in one forward pass. Questions may
        be `Noul`/`Choice`/`Score` objects or the same shapes as plain dicts."""
        questions = _QUESTIONS.validate_python(questions)
        routes = serving_routes(questions, self.tokenizer, self.config)

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
            probs = self.calibration.apply(
                scores, _bucket_type(question, route), route.mode.value, n_options=len(scores)
            )
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
        """Return candidate orders; None means the original order.

        Only lettered Choice and binary Noul in state-first layout are reversed.
        Schema-first shares options in the prefix; ordered scales retain the
        low-to-high ordering used in training.
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
        style = Style(self.config.prompt_style)
        codes = {name: route.codes for name, route in routes.items()}
        cached_schema = (
            schema_block(questions, codes, style)
            if self.config.layout is Layout.SCHEMA_FIRST
            else None
        )
        rendered: list[tuple[str, list[int] | None, Rendered]] = [
            (
                name,
                order,
                build(
                    state,
                    name,
                    question,
                    codes[name],
                    self.config.layout,
                    cached_schema,
                    order,
                    style,
                ),
            )
            for name, question in questions.items()
            for order in self._orders(question, routes[name])
        ]
        # Every variant shares one prefix by construction; a mismatch would make
        # the cache wrong, not just slow.
        prefixes = {r.prefix for _, _, r in rendered}
        if len(prefixes) != 1:
            raise AssertionError(f"layout produced {len(prefixes)} prefixes, expected 1")
        return rendered[0][2].prefix, [
            Variant(name, order, self.tokenizer.encode(r.suffix, add_special_tokens=False))
            for name, order, r in rendered
        ]

    def _forward(self, prefix: str, suffixes: list[list[int]], want_hidden: bool = False):
        """Logits (and hidden states) at each suffix's last token, one of two ways."""
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=True)
        if self.config.prefix_mode == "single":
            return self._forward_single(prefix_ids, suffixes, want_hidden)
        return self._forward_forked(prefix_ids, suffixes, want_hidden)

    def _forward_single(self, prefix_ids: list[int], suffixes: list[list[int]], want_hidden: bool):
        """One right-padded batch of prefix+suffix rows; no cache, no fork.

        Under `compile`, the batch is padded to a shape bucket -- rows to a
        power of two (duplicating the first row), width to `pad_to` -- so the
        set of recorded graphs stays small. Padding rows and tokens change no
        real row's logits: right padding is masked, and rows are independent.
        """
        import torch

        device = self._device
        rows = [prefix_ids + s for s in suffixes]
        n_real = len(rows)
        width = max(len(r) for r in rows)
        if self.config.compile:
            width = bucket(width, self.config.pad_to)
            rows = rows + [rows[0]] * (pow2(n_real) - n_real)
        pad = self.tokenizer.pad_token_id or 0
        batch = torch.tensor([r + [pad] * (width - len(r)) for r in rows], device=device)
        attention = torch.tensor(
            [[1] * len(r) + [0] * (width - len(r)) for r in rows], device=device
        )
        with torch.no_grad(), self._lock:
            out = self.model(
                input_ids=batch,
                attention_mask=attention,
                use_cache=False,
                output_hidden_states=want_hidden,
            )
        last_positions = torch.tensor([len(r) - 1 for r in rows[:n_real]], device=device)
        logits = out.logits[:n_real]
        hidden = out.hidden_states[-1][:n_real] if want_hidden else None
        return prefix_ids, logits, last_positions, hidden

    def _forward_forked(self, prefix_ids: list[int], suffixes: list[list[int]], want_hidden: bool):
        """Prefill the prefix once, then run all suffixes against forked caches."""
        import torch

        device = self._device

        with torch.no_grad(), self._lock:
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
                # Only for Mode B: full-batch hidden states are large.
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
        if route.mode is Mode.LABEL_TOKEN:
            return self._label_token_scores(logits, last_positions, row, route)
        return self._candidate_path_scores(hidden, last_positions, row, question)

    def _label_token_scores(self, logits, last_positions, row: int, route: Route) -> list[float]:
        from .readout.mode_a import LabelTokenReadout

        readout = LabelTokenReadout(
            self.tokenizer, prefix=label_prefix(Style(self.config.prompt_style))
        )
        candidate_ids = readout.candidate_ids(route.codes)
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

        Cached per candidate set, so a served schema encodes its candidates
        once. The cache is unbounded: fine for a fixed set of schemas (151
        candidates at hidden 2560 is ~1.5 MB), not for arbitrary caller-supplied
        option sets.
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
        device = self._device
        ids = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device)
        with torch.no_grad(), self._lock:
            states = self.model(
                input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False
            ).hidden_states[-1]
        last_positions = mask.sum(dim=1).long() - 1
        reprs = states[torch.arange(states.size(0), device=device), last_positions]
        self._candidate_cache[key] = reprs
        return reprs

    def _to_answer(self, question: Question, route: Route, probs: list[float]):
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
                # Two lettered options, yes first; no rating distribution exists,
                # so `probabilities` stays absent.
                return NoulAnswer(noul=probs[0], probabilities=None, confidence=confidence)
            by_rating = dict(enumerate(probs))
            return NoulAnswer(
                noul=noul_probability(by_rating),
                probabilities=by_rating,
                confidence=confidence,
            )

        raise TypeError(f"unknown question type {question.type!r}")


def load(
    checkpoint: str | Path | None = None,
    *,
    model_id: str = "Qwen/Qwen3.5-4B-Base",
    cache_dir: str | None = None,
    calibration: str | Path | None = None,
    noul_readout: Literal["rating", "binary"] | None = None,
    prompt_style: Literal["plain", "chat"] | None = None,
    compile: bool = False,
    max_label_options: int | None = None,
    skip_multi_token_codes: bool = True,
) -> DecisionEngine:
    """Load a checkpoint -- a release directory, a training output or a Hub id --
    into a ready `DecisionEngine`. With no checkpoint, serves `model_id` frozen.

        engine = lev.load("interfaze-ai/lev")
        engine.system_one(state, {"urgent": {"type": "noul", "instructions": "..."}})

    A release's `lev_release.json` names the base model and prompt style its
    weights were trained under and overrides `model_id` and `prompt_style`. The
    Mode B head and `calibration.json` beside the adapter are picked up; a
    missing calibration is warned about, not hidden.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .train.checkpoints import CALIBRATION, resolve_checkpoint

    # A checkpoint directory holds a LoRA *adapter*, not a full model: it has
    # `adapter_config.json` and no model config. So the base is loaded from
    # `model_id` and the adapter applied on top.
    resolved = resolve_checkpoint(checkpoint, cache_dir) if checkpoint else None
    manifest_file = resolved / "lev_release.json" if resolved else None
    if manifest_file and manifest_file.is_file():
        manifest = json.loads(manifest_file.read_text())
        prompt_style = manifest.get("prompt_style", prompt_style)
        if manifest["base_model"] != model_id:
            print(
                f"release manifest names base {manifest['base_model']!r}; "
                f"using it instead of {model_id!r}"
            )
            model_id = manifest["base_model"]

    # Tokenizer from the checkpoint when it saved one: the label-token readout
    # depends on which ids a code encodes to, so a mismatch produces plausible
    # wrong answers rather than an error.
    tok_source = resolved if resolved and (resolved / "tokenizer.json").is_file() else model_id
    tokenizer = AutoTokenizer.from_pretrained(str(tok_source), cache_dir=cache_dir)

    # The adapter wrapper and backbone expose different static types.
    model: Any = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.device_count() > 1 else None,
        cache_dir=cache_dir,
    )
    if torch.cuda.is_available() and torch.cuda.device_count() == 1:
        model = model.cuda()

    head = None
    if resolved:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(resolved))
        head = _load_head(resolved / MODE_B_HEAD, model)
    model = model.eval()

    # Default to the profile sitting beside the weights it was fitted for:
    # inside a flat release directory, or one level up in a training tree.
    if calibration is None and resolved:
        candidates = [resolved / CALIBRATION, resolved.parent / CALIBRATION]
        calibration = next((c for c in candidates if c.is_file()), candidates[-1])
    profile = CalibrationProfile()
    if calibration:
        try:
            profile = CalibrationProfile.load(calibration)
        except FileNotFoundError:
            print(f"WARNING: no calibration at {calibration}; serving raw softmax")

    config = EngineConfig(
        model_id=model_id,
        # A stock checkpoint pins the 0-8 rating scale at one end regardless of
        # content (ADR-007), so untrained serving reads Noul as two options.
        noul_readout=noul_readout or ("rating" if resolved else "binary"),
        compile=compile,
        prompt_style=prompt_style or "plain",
        max_label_options=max_label_options,
        skip_multi_token_codes=skip_multi_token_codes,
    )
    engine = DecisionEngine(model, tokenizer, config, profile, mode_b_head=head)
    engine.checkpoint = resolved
    if compile:
        print(f"warmup: compiled forward in {engine.warmup():.0f}s", flush=True)
    return engine


def _load_head(path: Path, model):
    """Load the saved Mode B head, or return None for Mode A-only serving."""
    import torch

    from .readout.mode_b import CandidatePathReadout

    if not path.is_file():
        return None

    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    # Shapes come from the saved tensors, not a config that may have drifted.
    proj_dim, hidden_size = state_dict["question_proj.weight"].shape
    head = CandidatePathReadout(hidden_size=hidden_size, proj_dim=proj_dim)
    head.load_state_dict(state_dict)
    return head.to(device=next(model.parameters()).device, dtype=torch.float32).eval()


def _bucket_type(question: Question, route: Route) -> str:
    """Keep binary Noul calibration separate from the trained rating scale."""
    return "noul_binary" if route.reason == BINARY_NOUL else question.type


def _fork(cache, n: int, device=None):
    """Copy a batch-1 prefix cache and expand it to `n` rows.

    `reorder_cache` supports both full- and linear-attention layers;
    `batch_repeat_interleave` does not. Copy first because reordering mutates
    the cache and the original must stay reusable.
    """
    import torch

    forked = copy.deepcopy(cache)
    rows = torch.zeros(n, dtype=torch.long, device=device)
    forked.reorder_cache(rows)
    return forked


def _gini(probs: list[float]) -> float:
    """Normalised Gini concentration: uniform -> 0, point mass -> 1."""
    k = len(probs)
    if k <= 1:
        return 1.0
    return (k * sum(p * p for p in probs) - 1.0) / (k - 1)
