"""Label codes, tokenizer checks and conversion of Noul ratings to probability."""

from __future__ import annotations

from itertools import islice, product
from string import ascii_uppercase
from typing import Protocol


class TokenEncoder(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]: ...


# Training taxonomy cap; serving routes by tokenizer support (ADR-025/026).
LABEL_OPTION_CAP = len(ascii_uppercase)


def label_codes(n: int) -> list[str]:
    if n < 1:
        raise ValueError("need at least one code")
    codes = list(islice(_all_codes(), n))
    if n > len(codes):
        raise ValueError(f"{n} options exceeds the {len(codes)}-code scheme")
    return codes


def single_token_codes(
    tokenizer, n: int, prefix: str = " ", skip_multi_token: bool = False
) -> list[str] | None:
    """Find n single-token codes using the prompt boundary prefix; optionally skip split codes."""
    if not skip_multi_token:
        try:
            codes = label_codes(n)
        except ValueError:
            return None
        return codes if all(_is_single_token(tokenizer, prefix + c) for c in codes) else None

    picked: list[str] = []
    for code in _all_codes():
        if _is_single_token(tokenizer, prefix + code):
            picked.append(code)
            if len(picked) == n:
                return picked
    return None


def _all_codes():
    yield from ascii_uppercase
    for width in (2, 3):
        for t in product(ascii_uppercase, repeat=width):
            yield "".join(t)


def _is_single_token(tokenizer, text: str) -> bool:
    return len(tokenizer.encode(text, add_special_tokens=False)) == 1


# Rating probabilities allow Noul calibration (ARCHITECTURE §3.4).
NOUL_RATING_TOKENS = [str(i) for i in range(9)]


def noul_probability(level_probs: dict[int, float]) -> float:
    """Collapse a 9-level rating distribution to p(yes) = sum(i/8 * p_i)."""
    top = len(NOUL_RATING_TOKENS) - 1
    return sum((level / top) * p for level, p in level_probs.items())
