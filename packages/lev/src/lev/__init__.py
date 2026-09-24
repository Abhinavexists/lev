"""lev — a typed, calibrated decision model.

State in, typed probabilistic decisions out. One prefill, many questions, no
generated tokens. Wire-compatible with TypeSafe's `/v1/systemone`.

The pure-Python core (schema, prompt layouts, router, calibration, contamination
guard) imports without torch, so it is testable on any machine. Anything needing a
model -- `lev.load`, `lev.server` -- imports torch only when called and needs
the `[train]` extra.
"""

from .calibrate import CalibrationProfile
from .model import DecisionEngine, load
from .prompt import Layout
from .router import Mode, route, route_all
from .types import (
    Answer,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
)

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "CalibrationProfile",
    "Choice",
    "ChoiceAnswer",
    "DecisionEngine",
    "Layout",
    "Mode",
    "Noul",
    "NoulAnswer",
    "Question",
    "Score",
    "ScoreAnswer",
    "SystemOneRequest",
    "SystemOneResponse",
    "Usage",
    "load",
    "route",
    "route_all",
]
