"""Turn Examples into tensors the readouts can be trained through.

Batches are **homogeneous in mode** (`ModeBatcher` groups them), so the two
readouts never need masking against each other.

The scored position is the final token of the rendered prompt, whose next-token
distribution is the answer. Rows are right-padded, so `last_positions` carries the
real index; reading `seq - 1` would train on a pad token's logits without failing.

Mode B also needs one representation per candidate. Candidates belong to the
question, so a batch's unique candidate strings are encoded once and shared (77
short rows, not 8 x 77), making a 77-option question cost about what a 4-option
one does.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from ..data.mixture import Example
from ..prompt import Style, candidate_texts, label_prefix
from ..prompt import build as build_prompt
from ..router import Mode, candidate_count, route

IGNORE_INDEX = -100


class RouteCache:
    """Resolves a question's readout mode once per distinct question.

    Re-tokenising an option set for each of 200k rows would be the slowest step
    in the pipeline; the batcher and collator share one cache.
    """

    def __init__(self, tokenizer, max_label_options: int | None = None, style: Style = Style.PLAIN):
        self.tokenizer = tokenizer
        self.max_label_options = max_label_options
        self.style = style
        self._routes: dict[tuple[str, str, int], object] = {}

    def route_for(self, example: Example):
        # Keyed on the option count: a source has many questions (subsampled
        # option sets, per-row QA choices), and the route depends only on the count.
        key = (example.source, example.name, candidate_count(example.question))
        if key not in self._routes:
            self._routes[key] = route(
                example.question,
                self.tokenizer,
                self.max_label_options,
                label_prefix=label_prefix(self.style),
            )
        return self._routes[key]


@dataclass
class Batch:
    """One homogeneous training batch. Tensors are torch, typed loosely to keep
    this module importable without torch for the shape tests."""

    mode: Mode
    input_ids: object  # (B, T)
    attention_mask: object  # (B, T)
    last_positions: object  # (B,)
    candidate_mask: object  # (B, Kmax) True where padded
    targets: object  # (B,) gold index, IGNORE_INDEX on abstain rows
    soft_targets: object | None  # (B, Kmax) or None if no abstain rows
    ordinal: object  # (B,) True for the ordered types: Score and Noul
    # Mode A only: the label token id per candidate slot.
    candidate_token_ids: object | None = None  # (B, Kmax)
    # Mode B only: a shared pool of encoded candidate strings, and per-row
    # indices into it.
    candidate_input_ids: object | None = None  # (C, Tc)
    candidate_attention_mask: object | None = None  # (C, Tc)
    candidate_last_positions: object | None = None  # (C,)
    candidate_index: object | None = None  # (B, Kmax) -> row in C, 0 where padded

    @property
    def size(self) -> int:
        return int(self.input_ids.shape[0])


def render(example: Example, codes: list[str] | None, style: Style = Style.PLAIN) -> str:
    rendered = build_prompt(
        example.state,
        example.name,
        example.question,
        codes,
        layout=example.layout,
        style=style,
    )
    return rendered.full


class ModeBatcher:
    """Groups examples by readout mode and length, then emits fixed-size batches.

    An example whose options do not fit single tokens goes to Mode B; nothing is
    dropped or capped.

    **Batches are length-bucketed.** A batch pads to its longest row: on the real
    mixture at batch 32, random batching computes 4.43x the real tokens (one
    1,300-token imdb review among 31 short tickets); bucketing brings that to
    1.43x, against a 1.03x floor (ADR-017).

    Sorting happens within a window of `bucket_window` batches, not globally, so
    length does not correlate with training order or source; batch order is then
    shuffled.
    """

    def __init__(
        self,
        tokenizer,
        batch_size: int,
        max_label_options: int | None = None,
        bucket_window: int = 64,
        seed: int = 17,
        routes: RouteCache | None = None,
    ):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.bucket_window = bucket_window
        self.seed = seed
        self.routes = routes or RouteCache(tokenizer, max_label_options)
        self.style = self.routes.style

    def route_for(self, example: Example):
        return self.routes.route_for(example)

    @staticmethod
    def length_of(example: Example) -> int:
        """Character count as a proxy for token count.

        Tokenising twice would cost more than the padding it saves, and characters
        track tokens closely enough (1.43x against a 1.03x floor).
        """
        return len(str(example.state))

    def __call__(self, examples: Iterable[Example], epoch: int = 0) -> Iterator[list[Example]]:
        pending: dict[Mode, list[Example]] = {Mode.LABEL_TOKEN: [], Mode.CANDIDATE_PATH: []}
        for example in examples:
            pending[self.route_for(example).mode].append(example)

        if self.bucket_window <= 1:
            batches = [
                rows[i : i + self.batch_size]
                for rows in pending.values()
                for i in range(0, len(rows), self.batch_size)
            ]
        else:
            batches = []
            window = self.batch_size * self.bucket_window
            for rows in pending.values():
                for start in range(0, len(rows), window):
                    chunk = sorted(rows[start : start + window], key=self.length_of)
                    batches.extend(
                        chunk[i : i + self.batch_size]
                        for i in range(0, len(chunk), self.batch_size)
                    )

        # Seeded on the epoch: differs between epochs, reproduces on a re-run.
        random.Random(self.seed + epoch).shuffle(batches)
        yield from (b for b in batches if b)


class DecisionCollator:
    """Examples -> Batch. All rows must share a mode; `ModeBatcher` guarantees it."""

    def __init__(
        self,
        tokenizer,
        max_seq_len: int = 4096,
        max_label_options: int | None = None,
        routes: RouteCache | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.routes = routes or RouteCache(tokenizer, max_label_options)
        self.style = self.routes.style

    def __call__(self, examples: list[Example]) -> Batch:
        import torch

        if not examples:
            raise ValueError("empty batch")
        routes = [self.routes.route_for(example) for example in examples]
        modes = {r.mode for r in routes}
        if len(modes) != 1:
            raise ValueError(f"batch mixes readout modes: {sorted(m.value for m in modes)}")
        mode = modes.pop()

        prompts = [render(e, r.codes, self.style) for e, r in zip(examples, routes, strict=True)]
        encoded = self._encode(prompts)

        n_candidates = [len(candidate_texts(e.question)) for e in examples]
        k_max = max(n_candidates)
        candidate_mask = torch.ones(len(examples), k_max, dtype=torch.bool)
        for i, n in enumerate(n_candidates):
            candidate_mask[i, :n] = False

        targets = torch.tensor(
            [IGNORE_INDEX if e.abstain else e.target for e in examples], dtype=torch.long
        )
        soft_targets = None
        if any(e.abstain for e in examples):
            soft_targets = torch.zeros(len(examples), k_max)
            for i, e in enumerate(examples):
                if e.soft_target is None:
                    continue
                if len(e.soft_target) != n_candidates[i]:
                    raise ValueError(
                        f"soft_target has {len(e.soft_target)} entries but the "
                        f"question has {n_candidates[i]} candidates"
                    )
                soft_targets[i, : n_candidates[i]] = torch.tensor(e.soft_target)

        # Noul is ordinal too ("0 = certainly no, 8 = certainly yes"): without
        # this, rating 4 against a truth of 8 costs as much as rating 0.
        ordinal = torch.tensor(
            [e.question.type in ("score", "noul") for e in examples], dtype=torch.bool
        )

        batch = Batch(
            mode=mode,
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            last_positions=encoded["last_positions"],
            candidate_mask=candidate_mask,
            targets=targets,
            soft_targets=soft_targets,
            ordinal=ordinal,
        )

        if mode is Mode.LABEL_TOKEN:
            batch.candidate_token_ids = self._label_token_ids(routes, k_max)
        else:
            self._attach_candidate_pool(batch, examples, k_max)
        return batch

    def _encode(self, prompts: list[str]) -> dict:
        out = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            add_special_tokens=False,
        )
        # The last real token, from the mask rather than len(text): truncation
        # may have shortened the row.
        out["last_positions"] = out["attention_mask"].sum(dim=1).long() - 1
        if (out["last_positions"] < 0).any():
            raise ValueError("a prompt encoded to zero tokens")
        return dict(out)

    def _label_token_ids(self, routes, k_max: int):
        import torch

        from ..readout.mode_a import LabelTokenReadout

        readout = LabelTokenReadout(self.tokenizer, prefix=label_prefix(self.style))
        ids = torch.zeros(len(routes), k_max, dtype=torch.long)
        for i, route_ in enumerate(routes):
            row = readout.candidate_ids(route_.codes)
            ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        return ids

    def _attach_candidate_pool(self, batch: Batch, examples: list[Example], k_max: int) -> None:
        """Encode each distinct candidate string once and index into the pool."""
        import torch

        pool: dict[str, int] = {}
        index = torch.zeros(len(examples), k_max, dtype=torch.long)
        for i, example in enumerate(examples):
            for j, text in enumerate(candidate_texts(example.question)):
                index[i, j] = pool.setdefault(text, len(pool))

        encoded = self.tokenizer(
            list(pool),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
            add_special_tokens=False,
        )
        batch.candidate_input_ids = encoded["input_ids"]
        batch.candidate_attention_mask = encoded["attention_mask"]
        batch.candidate_last_positions = encoded["attention_mask"].sum(dim=1).long() - 1
        batch.candidate_index = index
