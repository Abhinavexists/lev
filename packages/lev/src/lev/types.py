"""TypeSafe-compatible request and response models for `/v1/systemone`.

lev also exposes Noul rating probabilities and confidence. Clients that only
read the shared `noul` probability can ignore these extra fields.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

type JSONContent = str | dict | list

MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10


class Noul(BaseModel):
    """A yes/no question. The answer is the probability that the answer is yes."""

    type: Literal["noul"] = "noul"
    instructions: JSONContent | None = None
    criteria: dict[Literal["true", "false"], JSONContent | None] | None = None


class Choice(BaseModel):
    """Pick one option. `criteria` maps your option keys to their descriptions."""

    type: Literal["choice"] = "choice"
    instructions: JSONContent | None = None
    criteria: dict[str, JSONContent | None]

    @field_validator("criteria")
    @classmethod
    def _at_least_two(cls, v: dict) -> dict:
        if len(v) < 2:
            raise ValueError("a choice needs at least 2 options")
        return v


class Score(BaseModel):
    """Rate against ordered levels. `criteria[i]` describes level `i`."""

    type: Literal["score"] = "score"
    instructions: JSONContent | None = None
    criteria: list[JSONContent]

    @field_validator("criteria")
    @classmethod
    def _level_count(cls, v: list) -> list:
        if not MIN_SCORE_LEVELS <= len(v) <= MAX_SCORE_LEVELS:
            raise ValueError(
                f"a score needs {MIN_SCORE_LEVELS}-{MAX_SCORE_LEVELS} levels, got {len(v)}"
            )
        return v


type Question = Annotated[Noul | Choice | Score, Field(discriminator="type")]


def question_payload(question: Question) -> dict:
    """The question as a levbench task file carries it. `criteria` is required on
    a Choice and a Score and optional on a Noul, so an absent Noul criteria map
    stays absent rather than round-tripping as an empty one."""
    payload: dict = {"type": question.type, "instructions": question.instructions}
    if isinstance(question, Choice | Score) or question.criteria:
        payload["criteria"] = question.criteria
    return payload


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float
    # lev only; see the module docstring.
    probabilities: dict[int, float] | None = None
    confidence: float | None = None


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    probabilities: dict[int, float]
    legend: dict[int, JSONContent]
    confidence: float


type Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(BaseModel):
    input_tokens: int
    # Always 0: answers are read from logits, nothing is generated.
    output_tokens: int = 0
    cached_input_tokens: int = 0


class SystemOneRequest(BaseModel):
    state: JSONContent
    questions: dict[str, Question]
    model: str | None = None


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage
