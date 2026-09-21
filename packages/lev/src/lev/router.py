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

import json
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


def candidate_texts(question: Question) -> list[str]:
    """The strings Mode B scores: the option's own text, not a label code.

    Lives here rather than in `lev.train` because the serving engine needs the
    same list, and the engine must not import the training package.
    """
    if question.type == "noul":
        return list(NOUL_RATING_TOKENS)
    if question.type == "score":
        return [
            c if isinstance(c, str) else json.dumps(c, sort_keys=True) for c in question.criteria
        ]
    return list(question.criteria)


def route(question: Question, tokenizer, max_label_options: int | None = None) -> Route:
    """Pick a mode for one question.

    `max_label_options` is an optional policy cap *below* the tokenizer limit. Mode A
    accuracy decays as the label set grows -- decider measured -5 to -24 points on
    50-219 options -- so a deployment may prefer Mode B well before Mode A becomes
    impossible. None means "use Mode A whenever it is expressible".
    """
    n = candidate_count(question)

    if max_label_options is not None and n > max_label_options:
        return Route(Mode.CANDIDATE_PATH, None, f"{n} options over policy cap {max_label_options}")

    if isinstance(question, Noul):
        # Noul's candidates are the fixed rating scale, not user-supplied.
        codes = single_token_codes(tokenizer, n, prefix=" ") or None
        if codes is None:
            return Route(Mode.CANDIDATE_PATH, None, "rating tokens are not single tokens")
        return Route(Mode.LABEL_TOKEN, codes, "rating scale")

    codes = single_token_codes(tokenizer, n)
    if codes is None:
        return Route(Mode.CANDIDATE_PATH, None, f"{n} options exceed single-token codes")
    return Route(Mode.LABEL_TOKEN, codes, f"{n} options fit single-token codes")


def route_all(
    questions: dict[str, Question], tokenizer, max_label_options: int | None = None
) -> dict[str, Route]:
    return {n: route(q, tokenizer, max_label_options) for n, q in questions.items()}
