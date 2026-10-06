"""Shared fixtures; import Torch only when tensor batching is needed."""

from __future__ import annotations

import pytest
from lev.data.mixture import Example
from lev.prompt import Layout
from lev.train.config import PRESETS
from lev.types import Choice


class FakeTokenizer:
    """Control which strings encode as a single token."""

    def __init__(self, single_tokens: set[str]):
        self.single_tokens = single_tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if text in self.single_tokens:
            return [1]
        # Encode unlisted characters as multiple tokens to avoid accidental routing matches.
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
    """Encode one id per character to expose padding and last-token errors."""

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
