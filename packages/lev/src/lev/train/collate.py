"""Turn Examples into tensors the readouts can be trained through.

Batches are **homogeneous in mode**. A mixed batch is possible -- you mask the
two readouts against each other -- but the masking is where the bugs live, and a
batch that is all Mode A or all Mode B needs no masking at all. `ModeBatcher`
groups examples before they reach the collator.

The scored position is the final token of the rendered prompt, i.e. the token
whose *next-token* distribution is the answer. Rows are right-padded, so that is
not `seq - 1` for every row, and `last_positions` carries the real index. Getting
this wrong reads the logits of a pad token and trains on noise without failing.

Mode B needs one extra thing: a representation per candidate. Candidates are a
property of the *question*, not the row, so the unique candidate strings for a
batch are encoded once and shared -- 77 short rows rather than 8 x 77. This is
NanoJev's "one backbone forward" arrangement, and it is what makes training a
77-option question cost roughly what a 4-option one costs.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from ..data.mixture import Example
from ..prompt import build as build_prompt
from ..prompt import candidate_texts
from ..router import Mode, candidate_count, route

IGNORE_INDEX = -100


class RouteCache:
    """Resolves a question's readout mode once per distinct question.

    Re-tokenising an option set for each of 200k rows is the slowest thing in
    the pipeline and returns the same answer every time. Both the batcher and
    the collator need the route, so they share one of these.
    """

    def __init__(self, tokenizer, max_label_options: int | None = None):
        self.tokenizer = tokenizer
        self.max_label_options = max_label_options
        self._routes: dict[tuple[str, str, int], object] = {}

    def route_for(self, example: Example):
        # The option count is part of the key because one source now carries
        # many questions: subsampled option sets and per-row QA choices. The
        # route depends only on the count, so this stays one entry per distinct
        # size rather than one per row.
        key = (example.source, example.name, candidate_count(example.question))
        if key not in self._routes:
            self._routes[key] = route(example.question, self.tokenizer, self.max_label_options)
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


def render(example: Example, codes: list[str] | None) -> str:
    rendered = build_prompt(
        example.state,
        example.name,
        example.question,
        codes,
        layout=example.layout,
    )
    return rendered.full


class ModeBatcher:
    """Groups examples by readout mode and length, then emits fixed-size batches.

    Routing happens here rather than in the loop because the route depends on the
    tokenizer, and the whole point of the router is that the boundary is measured
    rather than assumed. An example whose options do not fit single tokens goes to
    Mode B; nothing is dropped and nothing is capped.

    **Batches are length-bucketed**, and that is worth as much as everything else
    in this file put together. A batch is padded to its longest row, so the model
    computes on the rectangle, not on the real tokens. Measured through this
    collator on the real mixture at batch 32, random batching spends **4.43x** of
    every forward pass on padding -- one 1,300-token imdb review lands among 31
    short banking tickets and drags the rectangle up to it. Bucketing brings that
    to 1.43x, against a 1.03x floor. See ADR-017.

    Sorting happens inside a shuffled *window*, not globally: a globally sorted
    epoch would feed every short example before every long one, which correlates
    batch composition with training order and with source. A window of
    `bucket_window` batches is long enough to make batches homogeneous and short
    enough that the order stays effectively random. Batch order is shuffled again
    afterwards so length does not track step number.
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

    def route_for(self, example: Example):
        return self.routes.route_for(example)

    @staticmethod
    def length_of(example: Example) -> int:
        """Character count as a proxy for token count.

        Tokenising twice -- once to bucket and once to collate -- would cost more
        than the padding it saves. Characters and tokens correlate closely enough
        that the buckets land at 1.43x against a 1.03x theoretical floor.
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

        # Seeded on the epoch so the order differs between epochs and still
        # reproduces on a re-run.
        random.Random(self.seed + epoch).shuffle(batches)
        yield from (b for b in batches if b)


class DecisionCollator:
    """Examples -> Batch. All rows must share a mode; `ModeBatcher` guarantees it."""

    def __init__(
        self,
        tokenizer,
        max_seq_len: int = 4096,
        max_label_options: int | None = None,
        label_prefix: str = " ",
        routes: RouteCache | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.label_prefix = label_prefix
        self.routes = routes or RouteCache(tokenizer, max_label_options)

    def __call__(self, examples: list[Example]) -> Batch:
        import torch

        if not examples:
            raise ValueError("empty batch")
        routes = [self.routes.route_for(example) for example in examples]
        modes = {r.mode for r in routes}
        if len(modes) != 1:
            raise ValueError(f"batch mixes readout modes: {sorted(m.value for m in modes)}")
        mode = modes.pop()

        prompts = [render(e, r.codes) for e, r in zip(examples, routes, strict=True)]
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

        # Noul is ordinal too, and more explicitly so than Score: its prompt
        # says "0 = certainly no, 8 = certainly yes" and its target is one of
        # the two ends. Without this, predicting rating 4 when the truth is 8 is
        # penalised exactly as hard as predicting 0 -- which is the failure the
        # ordinal term exists to prevent, on the readout where it matters most.
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
        import torch

        out = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            add_special_tokens=False,
        )
        # The scored position is the last *real* token, which right-padding moves
        # away from `seq - 1`. Derive it from the mask rather than from len(text):
        # truncation can have shortened the row.
        out["last_positions"] = out["attention_mask"].sum(dim=1).long() - 1
        if (out["last_positions"] < 0).any():
            raise ValueError("a prompt encoded to zero tokens")
        return {k: v if isinstance(v, torch.Tensor) else v for k, v in out.items()}

    def _label_token_ids(self, routes, k_max: int):
        import torch

        from ..readout.mode_a import LabelTokenReadout

        readout = LabelTokenReadout(self.tokenizer, prefix=self.label_prefix)
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
