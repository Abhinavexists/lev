"""A decision model plays Snake, one `/v1/systemone` call per move.

Ported from laya-mlx's demo (Apache-2.0, github.com/mizorewww/laya-mlx) with
its rules, planner and prompt wording kept verbatim, so a run here is
comparable to theirs. The difference is the wire: laya calls its model
in-process; this talks to any System One server through the same client
`levbench eval` uses, so the same game runs against lev, against Jev, or
against a frozen baseline, and the latency it shows includes the network.

One thing laya does not report is added: the planner knows the true answer
to both Noul questions ("is a safe route available?", "is food reachable?"),
so the model's yes/no estimates are scored live. In the compact prompt the
state text *states* those answers, which makes this the cheapest possible
test of whether a model reads its question -- the failure that cost lev 52
points on aegis2.
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
