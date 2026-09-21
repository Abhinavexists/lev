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

from .types import Choice, Noul, Question, Score

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


def render_question(name: str, question: Question, codes: list[str] | None) -> str:
    """Render one question. `codes` are Mode A label codes; None means Mode B."""
    lines = [f"Question: {question.instructions or name}"]

    if isinstance(question, Choice):
        # Under Mode B `codes` is None and the options are *not* listed: the
        # head embeds each candidate's own text, so listing 151 intents would
        # buy nothing and cost ~700 tokens per prompt. Emitting the bare
        # "Options:" header with nothing under it -- which an earlier version
        # did -- is the worst of both: the cost of a header and the information
        # of none.
        if codes:
            lines.append("Options:")
            for code, (key, desc) in zip(codes, question.criteria.items(), strict=False):
                lines.append(f"  {code}: {key}" + (f" - {_text(desc)}" if desc else ""))
        else:
            lines.append(f"Choose the best of the {len(question.criteria)} candidates given.")
    elif isinstance(question, Score):
        if codes:
            lines.append("Levels:")
            for code, desc in zip(codes, question.criteria, strict=False):
                lines.append(f"  {code}: {_text(desc)}")
        else:
            lines.append(f"Choose the best of the {len(question.criteria)} levels given.")
    elif isinstance(question, Noul):
        lines.append("Rate 0-8 how strongly this is true (0 = certainly no, 8 = certainly yes).")
        if question.criteria:
            for key in ("true", "false"):
                if (desc := question.criteria.get(key)) is not None:
                    lines.append(f"  {key}: {_text(desc)}")

    return "\n".join(lines)


def _text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def build(
    state,
    name: str,
    question: Question,
    codes: list[str] | None,
    layout: Layout = Layout.STATE_FIRST,
    schema_block: str | None = None,
) -> Rendered:
    """Render one question against one state, cut for the chosen layout."""
    state_block = f"Context:\n{render_state(state)}\n\n"
    question_block = render_question(name, question, codes)

    if layout is Layout.STATE_FIRST:
        return Rendered(prefix=state_block, suffix=f"{question_block}\n{ANSWER_CUE}")

    # Schema-first: the question catalogue is the reusable prefix, so it must not
    # depend on the state. `schema_block` lets a caller pass the whole catalogue
    # (every question, not just this one) to be cached across requests.
    prefix = (schema_block if schema_block is not None else question_block) + "\n\n"
    return Rendered(prefix=prefix, suffix=f"{state_block}{ANSWER_CUE}")


def schema_block(questions: dict[str, Question], codes: dict[str, list[str] | None]) -> str:
    """The full question catalogue, for schema-first caching across states."""
    return "\n\n".join(render_question(n, q, codes.get(n)) for n, q in questions.items())
