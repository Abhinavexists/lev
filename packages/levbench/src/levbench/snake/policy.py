"""Turn a board into one System One call, and the answer into a move.

The prompts are laya-mlx's, word for word, in both its variants. `compact`
puts the two Noul answers into the state text itself -- "Safe route: yes.
Food reachable through empty cells: yes." -- so a model that reads its input
scores 100% on those questions and one that does not, cannot. `detailed`
describes the same facts in prose and gives the move options fuller
descriptions.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

from typesafe_sdk import Choice, Noul

from ..runner import call_once
from .game import DIRECTIONS, SnakeGame

PROMPTS = ("compact", "detailed")


@dataclass
class Decision:
    probabilities: dict[str, float]
    proposed: str
    executed: str
    safe_directions: list[str]
    intervened: bool
    # The model's two Noul estimates, and what the planner knows to be true.
    dead_end_risk: float
    food_reachable: float
    route_truth: bool
    food_truth: bool
    inference_ms: float
    decision_ms: float
    input_tokens: int
    output_tokens: int
    safe_count: int
    planner_best: str
    served_by: str

    def to_dict(self) -> dict:
        return asdict(self)


def describe(game: SnakeGame, prompt: str) -> tuple[str, dict[str, Any], dict]:
    """The state text and questions for one board, plus the planner's view.

    Returned separately from `decide` so the wording can be tested without a
    model and logged without a call.
    """
    if prompt not in PROMPTS:
        raise ValueError(f"prompt must be one of {PROMPTS}, got {prompt!r}")
    moves = game.moves()
    safe = [m for m in moves if m.safe]
    preferred = max(safe, key=lambda m: m.advance).direction if safe else "NONE"
    reachable, space = game.food_reachability()

    if prompt == "detailed":
        descriptions = {}
        for move in moves:
            if not move.legal:
                descriptions[move.direction] = f"Collision: {move.reason}. Unsafe."
            elif not move.safe:
                descriptions[move.direction] = "Unsafe route. Risk of trapping the snake."
            elif move.eats:
                descriptions[move.direction] = "Safe. Eat the food immediately. Best move."
            elif move.direction == preferred:
                descriptions[move.direction] = "Safe. Best progress toward food."
            else:
                descriptions[move.direction] = "Safe but less progress toward food."
        state = (
            f"Snake game. {len(safe)} safe directions available. "
            f"Food reachable through empty cells: {'yes' if reachable else 'no'}. "
            f"Open cells: {space}. Snake length: {len(game.body)}. "
            f"{'There is a safe route forward.' if safe else 'The snake is trapped.'}"
        )
        questions = {
            "move": Choice(
                instructions=(
                    "Select the safest move with best progress toward food. Avoid collisions."
                ),
                criteria=descriptions,
            ),
            "risk": Noul(instructions="Is there a safe route forward for the snake?"),
            "food": Noul(instructions="Is food reachable through the currently empty cells?"),
        }
    else:
        descriptions = {}
        for move in moves:
            if not move.legal:
                descriptions[move.direction] = "Blocked. Collision."
            elif not move.safe:
                descriptions[move.direction] = "Unsafe. Traps the snake."
            elif move.eats:
                descriptions[move.direction] = "Safe. Eat food now. Best."
            elif move.direction == preferred:
                descriptions[move.direction] = "Safe. Best route to food."
            else:
                descriptions[move.direction] = "Safe. Slower route."
        state = (
            f"Safe route: {'yes' if safe else 'no'}. "
            f"Food reachable through empty cells: {'yes' if reachable else 'no'}."
        )
        questions = {
            "move": Choice(
                instructions="Choose the best safe move toward food.",
                criteria=descriptions,
            ),
            "risk": Noul(instructions="Is a safe route available?"),
            "food": Noul(instructions="Is food reachable through empty cells?"),
        }

    planner = {
        "safe": [m.direction for m in safe],
        "preferred": preferred,
        "reachable": reachable,
        "space": space,
    }
    return state, questions, planner


class PlannerClient:
    """A stand-in server: the planner's own view, returned in the SDK's shapes.

    For watching the display with no server up, and as the reference row every
    model is measured against -- it never needs the shield and answers both
    Noul questions from the same facts the state text states.
    """

    served = "planner"

    def system_one(self, state, questions):
        from types import SimpleNamespace

        criteria = questions["move"].criteria
        best = next((o for o in criteria if "Best" in criteria[o]), None)
        safe = [o for o in criteria if criteria[o].startswith("Safe")]
        pick = best or (safe[0] if safe else next(iter(criteria)))
        probabilities = {o: 0.02 for o in criteria}
        probabilities[pick] = 1.0 - 0.02 * (len(criteria) - 1)
        route = 0.98 if safe else 0.02
        food = 0.98 if "reachable through empty cells: yes" in state else 0.02
        return SimpleNamespace(
            answers={
                "move": SimpleNamespace(type="choice", probabilities=probabilities, choice=pick),
                "risk": SimpleNamespace(type="noul", noul=route),
                "food": SimpleNamespace(type="noul", noul=food),
            },
            usage=SimpleNamespace(input_tokens=0, output_tokens=0),
            model=self.served,
        )


class ModelPolicy:
    """Asks the server for a move; with `guarded`, refuses to execute an unsafe one."""

    def __init__(self, client, *, guarded: bool = True, prompt: str = "compact"):
        if prompt not in PROMPTS:
            raise ValueError(f"prompt must be one of {PROMPTS}, got {prompt!r}")
        self.client = client
        self.guarded = guarded
        self.prompt = prompt

    def decide(self, game: SnakeGame) -> Decision:
        started = time.perf_counter()
        state, questions, planner = describe(game, self.prompt)
        allowed = planner["safe"]
        if not allowed and self.guarded:
            raise RuntimeError("cycle safety invariant violated: no safe move exists")

        result = call_once(self.client, state, questions)
        answers = result.answers
        probabilities = {d: float(answers["move"].probabilities[d]) for d in DIRECTIONS}
        route = float(answers["risk"].noul)
        food = float(answers["food"].noul)
        values = [*probabilities.values(), route, food]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError(f"model returned an invalid probability; no move executed: {values}")

        proposed = max(DIRECTIONS, key=probabilities.__getitem__)
        executed = (
            max(allowed, key=probabilities.__getitem__)
            if self.guarded and proposed not in allowed
            else proposed
        )
        return Decision(
            probabilities=probabilities,
            proposed=proposed,
            executed=executed,
            safe_directions=allowed,
            intervened=proposed != executed,
            dead_end_risk=1.0 - route,
            food_reachable=food,
            route_truth=bool(allowed),
            food_truth=bool(planner["reachable"]),
            inference_ms=result.seconds * 1000,
            decision_ms=(time.perf_counter() - started) * 1000,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            safe_count=len(allowed),
            planner_best=planner["preferred"],
            served_by=result.served_by,
        )
