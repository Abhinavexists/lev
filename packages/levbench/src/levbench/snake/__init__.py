"""A decision model plays Snake, one `/v1/systemone` call per move.

Ported from laya-mlx's demo (Apache-2.0, github.com/mizorewww/laya-mlx) with
its rules, planner and prompt wording verbatim, so runs are comparable. laya
calls its model in-process; this uses the same client as `levbench eval`, so it
runs against lev, Jev or a frozen baseline, and its latency includes the network.

Added over laya: the planner knows the true answer to both Noul questions ("is a
safe route available?", "is food reachable?"), so the model's yes/no estimates
are scored live. The compact prompt's state text *states* those answers, making
this the cheapest test of whether a model reads its question -- the failure
behind the first run's aegis2 score, 52 points below Jev.
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
