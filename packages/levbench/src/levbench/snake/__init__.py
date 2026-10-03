"""A decision model plays Snake, one `/v1/systemone` call per move.

Ported from laya-mlx (Apache-2.0, github.com/mizorewww/laya-mlx) with its rules,
planner and prompts verbatim, so runs are comparable (ADR-022). The planner scores
both Noul answers live; the compact prompt states them (docs/FINDINGS.md §12).
"""

from .game import DIRECTIONS, SnakeGame
from .policy import Decision, ModelPolicy, PlannerClient
from .run import RunSummary, load_record, play, replay

__all__ = [
    "DIRECTIONS",
    "Decision",
    "ModelPolicy",
    "PlannerClient",
    "RunSummary",
    "SnakeGame",
    "load_record",
    "play",
    "replay",
]
