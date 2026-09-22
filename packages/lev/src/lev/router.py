"""Chooses the readout mode per question.

This is the piece no other implementation has. Every Family-A project (simple-jev,
litjev, decider, reflex) hits a hard ceiling when the option set will not fit into
single tokens, and either caps the option count or rejects the request. We route
those questions to Mode B instead, which scores candidate *text* and has no ceiling.

The boundary is not an option count. It is whether every option maps to a verified
single token for the tokenizer actually loaded -- a count would be wrong the moment
you swap tokenizers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .labels import NOUL_RATING_TOKENS, single_token_codes
from .types import Choice, Noul, Question, Score


class Mode(StrEnum):
    LABEL_TOKEN = "A"
    CANDIDATE_PATH = "B"


@dataclass(frozen=True)
class Route:
    mode: Mode
    codes: list[str] | None
    reason: str

    @property
    def n_candidates(self) -> int | None:
        return len(self.codes) if self.codes else None


def candidate_count(question: Question) -> int:
    if isinstance(question, Choice):
        return len(question.criteria)
    if isinstance(question, Score):
        return len(question.criteria)
    if isinstance(question, Noul):
        return len(NOUL_RATING_TOKENS)
    raise TypeError(f"unknown question type {type(question).__name__}")


BINARY_NOUL = "binary yes/no"


def route(
    question: Question,
    tokenizer,
    max_label_options: int | None = None,
    noul_binary: bool = False,
) -> Route:
    """Pick a mode for one question.

    `max_label_options` is an optional policy cap *below* the tokenizer limit. Mode A
    accuracy decays as the label set grows -- decider measured -5 to -24 points on
    50-219 options -- so a deployment may prefer Mode B well before Mode A becomes
    impossible. None means "use Mode A whenever it is expressible".

    `noul_binary` reads a Noul as two lettered options instead of the 0-8 rating
    scale. The scale is the trained target and the only calibratable form, but a
    stock checkpoint pins it at one end regardless of content (ADR-007), so an
    untrained deployment must use this.
    """
    n = candidate_count(question)

    if max_label_options is not None and n > max_label_options:
        return Route(Mode.CANDIDATE_PATH, None, f"{n} options over policy cap {max_label_options}")

    if isinstance(question, Noul) and noul_binary:
        codes = single_token_codes(tokenizer, 2)
        if codes is not None:
            return Route(Mode.LABEL_TOKEN, codes, BINARY_NOUL)

    if isinstance(question, Noul):
        # Noul's candidates are the fixed rating scale, not user-supplied.
        codes = single_token_codes(tokenizer, n, prefix=" ")
        if codes is None:
            return Route(Mode.CANDIDATE_PATH, None, "rating tokens are not single tokens")
        return Route(Mode.LABEL_TOKEN, codes, "rating scale")

    codes = single_token_codes(tokenizer, n)
    if codes is None:
        return Route(Mode.CANDIDATE_PATH, None, f"{n} options exceed single-token codes")
    return Route(Mode.LABEL_TOKEN, codes, f"{n} options fit single-token codes")


def route_all(
    questions: dict[str, Question],
    tokenizer,
    max_label_options: int | None = None,
    noul_binary: bool = False,
) -> dict[str, Route]:
    return {
        name: route(q, tokenizer, max_label_options, noul_binary) for name, q in questions.items()
    }
