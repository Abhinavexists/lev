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

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from ..data.mixture import Example
from ..prompt import build as build_prompt
from ..router import Mode, candidate_count, route
from ..router import candidate_texts as question_candidates

IGNORE_INDEX = -100


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
    ordinal: object  # (B,) True for Score questions
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


def candidate_texts(example: Example) -> list[str]:
    """The strings Mode B scores, for one example. See `router.candidate_texts`."""
    return question_candidates(example.question)


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
    """Groups examples by readout mode, then emits fixed-size batches.

    Routing happens here rather than in the loop because the route depends on the
    tokenizer, and the whole point of the router is that the boundary is measured
    rather than assumed. An example whose options do not fit single tokens goes to
    Mode B; nothing is dropped and nothing is capped.
    """

    def __init__(self, tokenizer, batch_size: int, max_label_options: int | None = None):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_label_options = max_label_options
        self._routes: dict[tuple[str, str, int], object] = {}

    def route_for(self, example: Example):
        # Routes are cached per question, not per row: re-tokenising an option
        # set for each of 200k rows is the slowest thing in the pipeline and it
        # returns the same answer every time.
        #
        # The option count is part of the key. Source and name alone would be
        # enough for the real mixture, where one source carries exactly one
        # question -- but if that ever stops being true the cache would hand a
        # 77-option question the route computed for a 4-option one, and the
        # batch would be collated for the wrong readout entirely.
        key = (example.source, example.name, candidate_count(example.question))
        if key not in self._routes:
            self._routes[key] = route(example.question, self.tokenizer, self.max_label_options)
        return self._routes[key]

    def __call__(self, examples: Iterable[Example]) -> Iterator[list[Example]]:
        pending: dict[Mode, list[Example]] = {Mode.LABEL_TOKEN: [], Mode.CANDIDATE_PATH: []}
        for example in examples:
            mode = self.route_for(example).mode
            pending[mode].append(example)
            if len(pending[mode]) == self.batch_size:
                yield pending[mode]
                pending[mode] = []
        for leftover in pending.values():
            if leftover:
                yield leftover


class DecisionCollator:
    """Examples -> Batch. All rows must share a mode; `ModeBatcher` guarantees it."""

    def __init__(
        self,
        tokenizer,
        max_seq_len: int = 4096,
        max_label_options: int | None = None,
        label_prefix: str = " ",
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.label_prefix = label_prefix
        self.batcher = ModeBatcher(tokenizer, batch_size=0, max_label_options=max_label_options)

    def __call__(self, examples: list[Example]) -> Batch:
        import torch

        if not examples:
            raise ValueError("empty batch")
        routes = [self.batcher.route_for(e) for e in examples]
        modes = {r.mode for r in routes}
        if len(modes) != 1:
            raise ValueError(f"batch mixes readout modes: {sorted(m.value for m in modes)}")
        mode = modes.pop()

        prompts = [render(e, r.codes) for e, r in zip(examples, routes, strict=True)]
        encoded = self._encode(prompts)

        n_candidates = [len(candidate_texts(e)) for e in examples]
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
            for j, text in enumerate(candidate_texts(example)):
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
