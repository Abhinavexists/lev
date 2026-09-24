"""Label codes, tokenizer checks and conversion of Noul ratings to probability."""

from __future__ import annotations

from itertools import islice, product
from string import ascii_uppercase

# The single-letter boundary: above it the source registry treats a label set as
# a large taxonomy, whose full set trains the Mode B head. Routing itself follows
# the tokenizer, not this cap (ADR-025/026).
LABEL_OPTION_CAP = len(ascii_uppercase)


def label_codes(n: int) -> list[str]:
    """First `n` codes: A..Z, then AA..ZZ, then AAA..ZZZ.

    Raises for n > 18,278 (past ZZZ).
    """
    if n < 1:
        raise ValueError("need at least one code")
    codes = list(islice(_all_codes(), n))
    if n > len(codes):
        raise ValueError(f"{n} options exceeds the {len(codes)}-code scheme")
    return codes


def single_token_codes(
    tokenizer, n: int, prefix: str = " ", skip_multi_token: bool = False
) -> list[str] | None:
    """Return `n` single-token codes, or None to request Mode B.

    The prefix must match the answer boundary: a space in plain prompts,
    no space in chat prompts. By default, all first `n` codes must pass.
    Serving may skip split codes; training retains contiguous codes (ADR-028).
    """
    if not skip_multi_token:
        try:
            codes = label_codes(n)
        except ValueError:
            return None
        verified = [c for c in codes if _is_single_token(tokenizer, prefix + c)]
        return verified[:n] if len(verified) >= n else None

    picked: list[str] = []
    for code in _all_codes():
        if _is_single_token(tokenizer, prefix + code):
            picked.append(code)
            if len(picked) == n:
                return picked
    return None


def _all_codes():
    """A..Z, AA..ZZ, AAA..ZZZ, lazily -- the same order as `label_codes`."""
    yield from ascii_uppercase
    for width in (2, 3):
        for t in product(ascii_uppercase, repeat=width):
            yield "".join(t)


def _is_single_token(tokenizer, text: str) -> bool:
    return len(tokenizer.encode(text, add_special_tokens=False)) == 1


# Noul is read from a rating scale rather than a two-way yes/no, so the answer
# carries a real distribution and is calibratable like Choice and Score.
# See docs/ARCHITECTURE.md §3.4.
NOUL_RATING_TOKENS = [str(i) for i in range(9)]


def noul_probability(level_probs: dict[int, float]) -> float:
    """Collapse a 9-level rating distribution to p(yes) = sum(i/8 * p_i)."""
    top = len(NOUL_RATING_TOKENS) - 1
    return sum((level / top) * p for level, p in level_probs.items())
