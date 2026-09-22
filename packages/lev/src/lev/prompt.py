"""The two prompt layouts, and where each one is cut for caching.

A layout exists to make one part of the input a stable prefix, so the forward pass
over that prefix can be computed once and reused. Which part you want stable depends
on the workload, so we build both and train on both.

    STATE-FIRST   [state] | [question + options + "Answer:"]
                  ^cache            ^forked per question
                  Default. One state, many questions -- the shared-state win.

    SCHEMA-FIRST  [all questions + options] | [state + "Answer:"]
                  ^cache across requests      ^varies per request
                  One schema, many states -- high-volume batch classification.

decider measured schema-first's accuracy cost precisely, and we inherit it as a
routing rule rather than a preference (docs/ARCHITECTURE.md §5.5):
  fixed label set        -1.5 pts
  options vary per item  -5 pts
  50-219 options         -5 to -24 pts
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum

from .labels import NOUL_RATING_TOKENS
from .types import Choice, JSONContent, Noul, Question, Score

ANSWER_CUE = "Answer:"


class Layout(StrEnum):
    STATE_FIRST = "state_first"
    SCHEMA_FIRST = "schema_first"


@dataclass(frozen=True)
class Rendered:
    """A prompt split at the point the cache is taken.

    `prefix` is computed once and reused; `suffix` is the varying part. The scored
    position is always the final token of `suffix`.
    """

    prefix: str
    suffix: str

    @property
    def full(self) -> str:
        return self.prefix + self.suffix


def render_state(state) -> str:
    if isinstance(state, str):
        return state
    # sort_keys so an identical dict always produces an identical prefix -- an
    # unsorted dump silently defeats prefix caching.
    return json.dumps(state, sort_keys=True, ensure_ascii=False, indent=None)


BINARY_NOUL_OPTIONS = ("yes", "no")


def _ordered(items: list, order: list[int] | None) -> list:
    return items if order is None else [items[i] for i in order]


def render_question(
    name: str, question: Question, codes: list[str] | None, order: list[int] | None = None
) -> str:
    """Render one question. `codes` are Mode A label codes; None means Mode B.

    `order` lists the candidates in the sequence they should appear, as indices
    into the question's own candidate order. Codes are assigned by position, so
    rendering the same question under two orders and averaging the readouts
    cancels the model's preference for whichever letter comes first -- the
    position bias reflex measured and corrects the same way.

    A Noul with exactly two codes is the binary yes/no readout, used when no
    trained adapter exists: a stock checkpoint cannot rate 0-8 (ADR-007) but can
    pick between two lettered options.
    """
    lines = [f"Question: {question.instructions or name}"]

    if isinstance(question, Choice):
        # No codes means Mode B, which does not list the options: the head
        # embeds each candidate's own text, so listing 151 intents would cost
        # ~700 tokens per prompt and buy nothing.
        if codes:
            lines.append("Options:")
            items = _ordered(list(question.criteria.items()), order)
            for code, (key, desc) in zip(codes, items, strict=False):
                lines.append(f"  {code}: {key}" + (f" - {render_content(desc)}" if desc else ""))
        else:
            lines.append(f"Choose the best of the {len(question.criteria)} candidates given.")
    elif isinstance(question, Score):
        if codes:
            lines.append("Levels:")
            for code, desc in zip(codes, _ordered(list(question.criteria), order), strict=False):
                lines.append(f"  {code}: {render_content(desc)}")
        else:
            lines.append(f"Choose the best of the {len(question.criteria)} levels given.")
    elif isinstance(question, Noul):
        if codes and len(codes) == len(BINARY_NOUL_OPTIONS):
            lines.append("Options:")
            described = {"yes": "true", "no": "false"}
            for code, option in zip(codes, _ordered(list(BINARY_NOUL_OPTIONS), order), strict=True):
                desc = (question.criteria or {}).get(described[option])
                lines.append(f"  {code}: {option}" + (f" - {render_content(desc)}" if desc else ""))
        else:
            lines.append(
                "Rate 0-8 how strongly this is true (0 = certainly no, 8 = certainly yes)."
            )
            if question.criteria:
                for key in ("true", "false"):
                    if (desc := question.criteria.get(key)) is not None:
                        lines.append(f"  {key}: {render_content(desc)}")

    return "\n".join(lines)


def render_content(value: JSONContent) -> str:
    """A `JSONContent` field as prompt text. Objects are dumped with sorted keys
    so an identical value always renders identically, as `render_state` does."""
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def candidate_texts(question: Question) -> list[str]:
    """The strings Mode B scores: each option's own text, not a label code.

    A rendering concern, so it lives beside the other renderers -- and both the
    serving engine and the training collator need it, while neither may import
    the other.
    """
    if isinstance(question, Noul):
        return list(NOUL_RATING_TOKENS)
    if isinstance(question, Score):
        return [render_content(level) for level in question.criteria]
    return list(question.criteria)


def build(
    state,
    name: str,
    question: Question,
    codes: list[str] | None,
    layout: Layout = Layout.STATE_FIRST,
    cached_schema: str | None = None,
    order: list[int] | None = None,
) -> Rendered:
    """Render one question against one state, cut for the chosen layout."""
    state_block = f"Context:\n{render_state(state)}\n\n"
    question_block = render_question(name, question, codes, order)

    if layout is Layout.STATE_FIRST:
        return Rendered(prefix=state_block, suffix=f"{question_block}\n{ANSWER_CUE}")

    # Schema-first: the question catalogue is the reusable prefix, so it must not
    # depend on the state. `cached_schema` lets a caller pass the whole catalogue
    # (every question, not just this one) to be cached across requests.
    prefix = (cached_schema if cached_schema is not None else question_block) + "\n\n"
    return Rendered(prefix=prefix, suffix=f"{state_block}{ANSWER_CUE}")


def schema_block(questions: dict[str, Question], codes: dict[str, list[str] | None]) -> str:
    """The full question catalogue, for schema-first caching across states."""
    return "\n\n".join(render_question(n, q, codes.get(n)) for n, q in questions.items())
