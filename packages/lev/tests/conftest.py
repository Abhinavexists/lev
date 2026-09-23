"""Shared test fixtures: fake tokenizers, and builders for Examples/questions.

Nothing here imports torch, so the torch-free half of the suite still collects
on a bare `uv sync`.
"""

from __future__ import annotations

import pytest
from lev.data.mixture import Example
from lev.prompt import Layout
from lev.train.config import PRESETS
from lev.types import Choice


class FakeTokenizer:
    """Treats a fixed vocabulary of strings as single tokens; splits everything else.

    `single_tokens` is the set of strings (including any leading space) that encode
    to exactly one id. That is precisely the property `labels.single_token_codes`
    checks, so a fake is enough to test the routing decision faithfully.
    """

    def __init__(self, single_tokens: set[str]):
        self.single_tokens = single_tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if text in self.single_tokens:
            return [1]
        # Anything outside the vocabulary must encode to *more* than one token,
        # including single characters -- otherwise a one-char string looks
        # single-token by accident and the router test proves nothing.
        return [1] * (len(text) + 1)


@pytest.fixture
def rich_tokenizer() -> FakeTokenizer:
    """Space-prefixed A-Z and 0-8 are single tokens. Two-letter codes are not."""
    from string import ascii_uppercase

    return FakeTokenizer({f" {c}" for c in ascii_uppercase} | {f" {i}" for i in range(9)})


@pytest.fixture
def chat_tokenizer() -> FakeTokenizer:
    """Bare A-Z and 0-8 are single tokens too -- what follows an assistant turn."""
    from string import ascii_uppercase

    letters = set(ascii_uppercase) | {f" {c}" for c in ascii_uppercase}
    digits = {str(i) for i in range(9)} | {f" {i}" for i in range(9)}
    return FakeTokenizer(letters | digits)


@pytest.fixture
def poor_tokenizer() -> FakeTokenizer:
    """Only the first four letters are single tokens — forces Mode B quickly."""
    return FakeTokenizer({" A", " B", " C", " D"} | {f" {i}" for i in range(9)})


class BatchingTokenizer(FakeTokenizer):
    """Adds the `__call__` surface the collator needs: padding and tensors.

    Encoding is one id per character so that different prompts get genuinely
    different lengths -- a fake that returns equal lengths would hide the exact
    bug `last_positions` exists to prevent.
    """

    pad_token_id = 0
    padding_side = "right"

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if text in self.single_tokens:
            # A stable, distinct id per label so the collator's gather is checkable.
            return [1000 + sorted(self.single_tokens).index(text)]
        return [1] * (len(text) + 1)

    def __call__(
        self,
        texts,
        return_tensors=None,
        padding=True,
        truncation=True,
        max_length=None,
        add_special_tokens=False,
    ):
        import torch

        rows = [[(ord(c) % 97) + 2 for c in t] for t in texts]
        if truncation and max_length:
            rows = [r[:max_length] for r in rows]
        width = max(len(r) for r in rows)
        input_ids = torch.zeros(len(rows), width, dtype=torch.long)
        mask = torch.zeros(len(rows), width, dtype=torch.long)
        for i, r in enumerate(rows):
            input_ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
            mask[i, : len(r)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}


@pytest.fixture
def batching_tokenizer() -> BatchingTokenizer:
    from string import ascii_uppercase

    return BatchingTokenizer({f" {c}" for c in ascii_uppercase} | {f" {i}" for i in range(9)})


# Builders shared by the training-side tests.


@pytest.fixture
def train_config():
    """A throwaway copy of the smoke preset, safe for a test to mutate."""
    from dataclasses import replace

    return replace(PRESETS["smoke"])


def a_choice(n_options: int = 4) -> Choice:
    return Choice(instructions="pick", criteria={f"option {i}": None for i in range(n_options)})


def an_example(
    question,
    target: int = 0,
    source: str = "s",
    state: str = "a state",
    abstain: bool = False,
    soft_target: list[float] | None = None,
) -> Example:
    return Example(
        state=state,
        name=source,
        question=question,
        target=target,
        layout=Layout.STATE_FIRST,
        source=source,
        abstain=abstain,
        soft_target=soft_target,
    )
