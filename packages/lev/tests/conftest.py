"""A fake tokenizer, so the router and label logic are testable without torch."""

from __future__ import annotations

import pytest


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
def poor_tokenizer() -> FakeTokenizer:
    """Only the first four letters are single tokens — forces Mode B quickly."""
    return FakeTokenizer({" A", " B", " C", " D"} | {f" {i}" for i in range(9)})
