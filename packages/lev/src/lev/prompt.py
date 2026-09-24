"""Prompt text and prefix/suffix boundaries shared by training and serving.

State-first shares the state prefix across questions. Schema-first shares
the question catalogue across states. Both score the final suffix token.
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


class Style(StrEnum):
    """Prompt format recorded with the weights and shared by training and serving.

    Plain uses Context/Question/Options/Answer sections. Chat uses ChatML with
    a system prompt and an empty think block (ADR-027).
    """

    PLAIN = "plain"
    CHAT = "chat"


def label_prefix(style: Style) -> str:
    """The character before a label token as the model emits it: a space after
    `Answer:`, nothing at the start of an assistant turn. Verifying the wrong
    variant passes while the scored token differs."""
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
    # sort_keys: an identical dict must produce an identical prefix.
    return json.dumps(state, sort_keys=True, ensure_ascii=False, indent=None)


BINARY_NOUL_OPTIONS = ("yes", "no")


def _ordered(items: list, order: list[int] | None) -> list:
    return items if order is None else [items[i] for i in order]


def render_question(
    name: str,
    question: Question,
    codes: list[str] | None,
    order: list[int] | None = None,
    style: Style = Style.PLAIN,
) -> str:
    """Render one question. `codes` are Mode A label codes; None means Mode B.

    `order` gives the display sequence as indices into the question's own
    candidate order. Codes are assigned by position, so averaging two orders
    cancels the model's preference for the first letter (as reflex does).

    A Noul with exactly two codes is the binary yes/no readout for an untrained
    checkpoint, which cannot rate 0-8 (ADR-007) but can pick a lettered option.
    """
    if style is Style.CHAT:
        return _render_question_chat(name, question, codes, order)
    lines = [f"Question: {question.instructions or name}"]

    if isinstance(question, Choice):
        # Mode B does not list options: the head embeds each candidate's text,
        # and listing 151 intents would cost ~700 tokens per prompt.
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
    """reflex's headed layout: `# Criterion`, `# Options` with `A. text` lines,
    and an explicit ask. Codes are positional, as in the plain style."""
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


def render_content(value: JSONContent) -> str:
    """A `JSONContent` field as prompt text. Objects are dumped with sorted keys
    so an identical value always renders identically, as `render_state` does."""
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def candidate_texts(question: Question) -> list[str]:
    """Candidate text shared by the Mode B serving and training paths."""
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
    """Render one question against one state, cut for the chosen layout."""
    if style is Style.CHAT:
        return _build_chat(state, name, question, codes, layout, cached_schema, order)
    state_block = f"Context:\n{render_state(state)}\n\n"
    question_block = render_question(name, question, codes, order)

    if layout is Layout.STATE_FIRST:
        return Rendered(prefix=state_block, suffix=f"{question_block}\n{ANSWER_CUE}")

    # Schema-first: the question catalogue is the reusable prefix, so it must not
    # depend on the state. `cached_schema` lets a caller pass the whole catalogue
    # (every question, not just this one) to be cached across requests.
    prefix = (cached_schema if cached_schema is not None else question_block) + "\n\n"
    return Rendered(prefix=prefix, suffix=f"{state_block}{ANSWER_CUE}")


def _build_chat(state, name, question, codes, layout, cached_schema, order) -> Rendered:
    """ChatML: system turn, then a user turn holding evidence and criterion in
    either order, then the opened assistant turn the label token is read from.
    The scored position is the final `\\n\\n` after the empty think block."""
    head = f"<|im_start|>system\n{CHAT_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
    evidence = f"# Evidence\n{render_state(state)}\n\n"
    criterion = render_question(name, question, codes, order, Style.CHAT)
    if layout is Layout.STATE_FIRST:
        return Rendered(prefix=head + evidence, suffix=f"{criterion}\n{CHAT_TAIL}")
    catalogue = cached_schema if cached_schema is not None else criterion
    return Rendered(prefix=f"{head}{catalogue}\n\n", suffix=f"{evidence.rstrip()}\n{CHAT_TAIL}")


def schema_block(
    questions: dict[str, Question],
    codes: dict[str, list[str] | None],
    style: Style = Style.PLAIN,
) -> str:
    """The full question catalogue, for schema-first caching across states."""
    return "\n\n".join(
        render_question(n, q, codes.get(n), style=style) for n, q in questions.items()
    )
