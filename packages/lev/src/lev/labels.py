"""Internal label codes, and the single-token check that decides the readout mode.

Mode A scores a bare letter code at the answer boundary, never the option's own text.
So option keys are arbitrary strings and the model only ever sees `A`, `B`, ... This
is why `Choice.criteria` is a map rather than a list.

The hard constraint: each code must be exactly one token for the loaded tokenizer.
LitJev rejects a tokenizer that cannot express the codes. We do not reject — we fall
through to Mode B, which has no such limit. That fallthrough is the design's point.
"""

from __future__ import annotations

from itertools import product
from string import ascii_uppercase

# The single-letter boundary. Once the routing cap for Mode A (ADR-020); since
# ADR-025/026 routing follows the tokenizer on both sides, and this is the
# threshold the registry uses to call a source "large-taxonomy" (its full set
# is the Mode B head's data) and the edge of the mid/large calibration bands.
LABEL_OPTION_CAP = len(ascii_uppercase)


def label_codes(n: int) -> list[str]:
    """First `n` codes: A..Z, then AA..ZZ, then AAA..ZZZ.

    Raises for n > 18,278, which is the point at which the scheme stops being a
    sensible way to pose a question rather than a technical limit.
    """
    if n < 1:
        raise ValueError("need at least one code")
    codes: list[str] = list(ascii_uppercase)
    for width in (2, 3):
        if len(codes) >= n:
            break
        codes.extend("".join(t) for t in product(ascii_uppercase, repeat=width))
    if n > len(codes):
        raise ValueError(f"{n} options exceeds the {len(codes)}-code scheme")
    return codes[:n]


def single_token_codes(
    tokenizer, n: int, prefix: str = " ", skip_multi_token: bool = False
) -> list[str] | None:
    """Return `n` codes that are each one token, or None if that is impossible.

    `prefix` matters: the model predicts the token *after* `Answer:`, which for most
    tokenizers is space-prefixed. Verifying the bare code would pass while the actual
    scored token differs — a silent correctness bug, so we verify what is really read.

    By default the codes are exactly the first `n` of the scheme, so one question
    always gets the same codes. For Qwen3.5 the 69th, `BQ`, is two tokens, which
    makes 68 options the Mode A limit. `skip_multi_token` passes over codes that
    split and takes the next single-token ones instead, lifting the limit to
    several hundred; a server opts in and pairs it with an explicit option cap
    (ADR-028), because training never shows Mode A a set above 68.

    Returning None is not a failure. It is the signal to use Mode B.
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
