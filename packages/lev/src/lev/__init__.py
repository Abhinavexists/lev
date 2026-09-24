"""Typed, calibrated decisions without token generation.

The core imports without Torch; loading a model requires the train extra.
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

__version__ = "0.1.1"

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
