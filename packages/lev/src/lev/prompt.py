"""Prompt text and prefix/suffix boundaries shared by training and serving."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from .labels import NOUL_RATING_TOKENS
from .types import Choice, JSONContent, Noul, Question, Score

ANSWER_CUE = "Answer:"


class Layout(StrEnum):
    STATE_FIRST = "state_first"
    SCHEMA_FIRST = "schema_first"


# The prompt format must match training (ADR-027).
class Style(StrEnum):
    PLAIN = "plain"
    CHAT = "chat"


def label_prefix(style: Style) -> str:
    return " " if style is Style.PLAIN else ""


CHAT_SYSTEM_PROMPT = (
    "You are a System One decision model. You read the Evidence and answer each "
    "Criterion by choosing exactly one of the listed options. You never explain. "
    "You answer with the single option label only."
)
CHAT_TAIL = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
CHAT_ASK = {
    "choice": "Respond with only the letter of the best option.",
    "score": "Respond with only the letter of the level that best matches.",
    "noul": "Respond with only a digit from 0 to 8.",
}


# The prefix is reusable and the suffix varies; the suffix's final token is the one scored.
@dataclass(frozen=True)
class Rendered:
    prefix: str
    suffix: str

    @property
    def full(self) -> str:
        return self.prefix + self.suffix


def render_state(state) -> str:
    if isinstance(state, str):
        return state
    # sort_keys: an identical dict must produce an identical prefix.
    return json.dumps(state, sort_keys=True, ensure_ascii=False, indent=None)


BINARY_NOUL_OPTIONS = ("yes", "no")


def _ordered(items: list, order: list[int] | None) -> list:
    return items if order is None else [items[i] for i in order]


# Two Noul codes select the binary readout.
def render_question(
    name: str,
    question: Question,
    codes: list[str] | None,
    order: list[int] | None = None,
    style: Style = Style.PLAIN,
) -> str:
    if style is Style.CHAT:
        return _render_question_chat(name, question, codes, order)
    lines = [f"Question: {question.instructions or name}"]

    if isinstance(question, Choice):
        # Mode B embeds candidates separately, avoiding long option lists in prompts.
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


def _render_question_chat(
    name: str, question: Question, codes: list[str] | None, order: list[int] | None
) -> str:
    lines = [f"# Criterion\n{question.instructions or name}\n"]

    if isinstance(question, Choice):
        if codes:
            lines.append("# Options")
            for code, (key, desc) in zip(
                codes, _ordered(list(question.criteria.items()), order), strict=False
            ):
                lines.append(f"{code}. {key}" + (f": {render_content(desc)}" if desc else ""))
            lines.append(f"\n{CHAT_ASK['choice']}")
        else:
            lines.append(f"Choose the best of the {len(question.criteria)} candidates given.")
    elif isinstance(question, Score):
        top = len(question.criteria) - 1
        if codes:
            lines.append("# Options")
            for code, (level, desc) in zip(
                codes, _ordered(list(enumerate(question.criteria)), order), strict=False
            ):
                lines.append(f"{code}. (level {level} of {top}) {render_content(desc)}")
            lines.append(f"\n{CHAT_ASK['score']}")
        else:
            lines.append(f"Choose the best of the {len(question.criteria)} levels given.")
    elif isinstance(question, Noul):
        criteria = question.criteria or {}
        if codes and len(codes) == len(BINARY_NOUL_OPTIONS):
            yes = render_content(criteria.get("true") or "the statement is true")
            no = render_content(criteria.get("false") or "the statement is false")
            pair = [("yes", yes), ("no", no)]
            lines.append("# Options")
            for code, (option, desc) in zip(codes, _ordered(pair, order), strict=True):
                lines.append(f"{code}. {option}: {desc}")
            lines.append(f"\n{CHAT_ASK['choice']}")
        else:
            lines.append("# Scale")
            lines.append("0 = certainly no ... 8 = certainly yes")
            for key, label in (("true", "yes"), ("false", "no")):
                if (desc := criteria.get(key)) is not None:
                    lines.append(f"{label}: {render_content(desc)}")
            lines.append(f"\n{CHAT_ASK['noul']}")

    return "\n".join(lines)


# Sorted object keys keep the rendering deterministic.
def render_content(value: JSONContent) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


# Shared by Mode B serving and training, so both see the same candidate text.
def candidate_texts(question: Question) -> list[str]:
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
    style: Style = Style.PLAIN,
) -> Rendered:
    if style is Style.CHAT:
        return _build_chat(state, name, question, codes, layout, cached_schema, order)
    state_block = f"Context:\n{render_state(state)}\n\n"
    question_block = render_question(name, question, codes, order)

    if layout is Layout.STATE_FIRST:
        return Rendered(prefix=state_block, suffix=f"{question_block}\n{ANSWER_CUE}")

    # The reusable schema prefix must contain the full catalogue and exclude state.
    prefix = (cached_schema if cached_schema is not None else question_block) + "\n\n"
    return Rendered(prefix=prefix, suffix=f"{state_block}{ANSWER_CUE}")


# Score the boundary after the empty think block.
def _build_chat(state, name, question, codes, layout, cached_schema, order) -> Rendered:
    head = f"<|im_start|>system\n{CHAT_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
    evidence = f"# Evidence\n{render_state(state)}\n\n"
    criterion = render_question(name, question, codes, order, Style.CHAT)
    if layout is Layout.STATE_FIRST:
        return Rendered(prefix=head + evidence, suffix=f"{criterion}\n{CHAT_TAIL}")
    catalogue = cached_schema if cached_schema is not None else criterion
    return Rendered(prefix=f"{head}{catalogue}\n\n", suffix=f"{evidence.rstrip()}\n{CHAT_TAIL}")


# The full question catalogue, so schema-first caching can share it across states.
def schema_block(
    questions: Mapping[str, Question],
    codes: Mapping[str, list[str] | None],
    style: Style = Style.PLAIN,
) -> str:
    return "\n\n".join(
        render_question(n, q, codes.get(n), style=style) for n, q in questions.items()
    )
