"""The game loop: one decision per move, rounds, pacing, a replayable record.

Round semantics follow laya: with the shield on, a finished board starts the
next round on the next seed; unassisted, the first death ends the run so the
survival number means something. The JSONL record is laya's format
(`metadata`, `frame`, `round_end`, `end`) with `at` timestamps, so
`levbench replay` plays it back at original speed.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .. import pricing
from .game import SnakeGame
from .policy import Decision, ModelPolicy

RECORD_FORMAT = "levbench-snake-v1"


@dataclass
class RunSummary:
    backend: str
    model: str
    prompt: str
    guarded: bool
    board: str
    seed: int
    steps: int
    rounds: int
    deaths: int
    score: int
    best_score: int
    best_length: int
    alive: bool
    won: bool
    death_reason: str | None
    seconds: float
    decisions_per_second: float
    inference_ms_p50: float
    inference_ms_p95: float
    mean_input_tokens: float
    cost_usd: float
    interventions: int
    intervention_rate: float
    # How often the model's *raw* first choice was a safe move.
    raw_safe_rate: float
    # The two Noul questions, scored against what the planner knows.
    route_accuracy: float
    route_brier: float
    food_accuracy: float
    food_brier: float

    def to_dict(self) -> dict:
        return asdict(self)

    def lines(self) -> list[str]:
        shield = "on" if self.guarded else "off"
        end = "won" if self.won else ("alive" if self.alive else f"died: {self.death_reason}")
        cost = f"   cost ${self.cost_usd:.4f}" if not pricing.is_self_hosted(self.model) else ""
        return [
            f"=== snake / {self.backend} / {self.model}  prompt={self.prompt} shield={shield} ===",
            f"board {self.board} seed {self.seed}   steps {self.steps}   rounds {self.rounds}   "
            f"deaths {self.deaths}   best score {self.best_score}   final: {end}",
            f"decisions/s {self.decisions_per_second:.2f}   "
            f"inference p50 {self.inference_ms_p50:.0f} ms   p95 {self.inference_ms_p95:.0f} ms"
            f"   tokens/move {self.mean_input_tokens:.0f}{cost}",
            f"raw first choice safe {self.raw_safe_rate:.1%}   shield interventions "
            f"{self.interventions} ({self.intervention_rate:.1%})",
            f"noul vs planner truth   route: acc {self.route_accuracy:.1%} "
            f"brier {self.route_brier:.3f}   food: acc {self.food_accuracy:.1%} "
            f"brier {self.food_brier:.3f}",
        ]


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


class Recorder:
    def __init__(self, path: str | Path | None):
        self.handle = None
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            # Held open across the run and closed by `close()` in the loop's `finally`.
            self.handle = Path(path).open("a", encoding="utf-8")  # noqa: SIM115

    def write(self, event: dict) -> None:
        if self.handle:
            self.handle.write(json.dumps(event, separators=(",", ":")) + "\n")

    def close(self) -> None:
        if self.handle:
            self.handle.close()


def play(
    client,
    backend: str,
    model: str,
    *,
    width: int = 24,
    height: int = 16,
    seed: int = 7,
    initial_length: int = 6,
    steps: int | None = None,
    duration: float | None = None,
    fps: float | None = None,
    guarded: bool = True,
    prompt: str = "compact",
    record: str | Path | None = None,
    network: str = "",
    display=None,
    keys: Callable[[], str] | None = None,
    on_step: Callable[[dict, dict, dict], None] | None = None,
) -> RunSummary:
    """Play until `steps` moves, `duration` seconds, Q, or -- unassisted -- a death.

    `fps` paces moves to a budget; None waits only on inference. `display` is a
    `LiveDisplay`; `keys` returns pressed keys (SPACE pause, +/- speed, R reset,
    Q quit); `on_step` gets `(game, decision, stats)` as dicts for headless
    status lines.
    """
    game = SnakeGame(width, height, seed, initial_length)
    policy = ModelPolicy(client, guarded=guarded, prompt=prompt)
    recorder = Recorder(record)
    recorder.write(
        {
            "type": "metadata",
            "format": RECORD_FORMAT,
            "created_utc": datetime.now(UTC).isoformat(),
            "backend": backend,
            "model": model,
            "settings": {
                "width": width,
                "height": height,
                "seed": seed,
                "initial_length": initial_length,
                "prompt": prompt,
                "guarded": guarded,
                "fps": fps,
            },
            "note": "Board shown before the announced action. Risk is 1 - P(safe route).",
        }
    )

    decisions: list[Decision] = []
    stats = {
        "backend": backend,
        "model": model,
        "network": network,
        "prompt": prompt,
        "guarded": guarded,
        "interventions": 0,
        "best": 0,
        "round": 1,
        "paused": False,
        "elapsed": 0.0,
        "steps_per_second": 0.0,
        "steps": 0,
        "route_acc": 0.0,
        "food_acc": 0.0,
    }
    deaths = 0
    best_length = len(game.body)
    served = model
    timestamps: deque[float] = deque(maxlen=60)
    shown_game, shown_decision = game.snapshot(), {}
    started = time.perf_counter()

    try:
        while True:
            now = time.perf_counter()
            if (duration and now - started >= duration) or (steps and len(decisions) >= steps):
                break
            pressed = keys().lower() if keys else ""
            if "q" in pressed or "\x03" in pressed:
                break
            if " " in pressed:
                stats["paused"] = not stats["paused"]
            if "+" in pressed or "\x1b[a" in pressed:
                fps = min(240.0, (fps or 12.0) + 2)
            if "-" in pressed or "\x1b[b" in pressed:
                fps = max(1.0, (fps or 12.0) - 2)
            if "r" in pressed:
                stats["round"] += 1
                game = SnakeGame(width, height, seed + stats["round"] - 1, initial_length)
                shown_game, shown_decision = game.snapshot(), {}
            if stats["paused"]:
                stats["elapsed"] = now - started
                if display:
                    from .ui import compose

                    display.show(compose(shown_game, shown_decision, stats))
                time.sleep(0.03)
                continue
            if display:
                fits, need = display.fits(game.width, game.height)
                if not fits:
                    display.message(
                        f"Resize the terminal to at least {need[0]} columns x {need[1]} rows. "
                        "The game is waiting. Q quits."
                    )
                    time.sleep(0.1)
                    continue

            decision = policy.decide(game)
            decisions.append(decision)
            served = decision.served_by or served
            stats["model"] = served
            stats["interventions"] += decision.intervened
            shown = time.perf_counter()
            timestamps.append(shown)
            stats["elapsed"] = shown - started
            stats["steps_per_second"] = (
                (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
                if len(timestamps) > 1 and timestamps[-1] > timestamps[0]
                else 0.0
            )
            stats["steps"] = len(decisions)
            stats["best"] = max(stats["best"], game.score)
            n = len(decisions)
            stats["route_acc"] = (
                sum(((1 - d.dead_end_risk) >= 0.5) == d.route_truth for d in decisions) / n
            )
            stats["food_acc"] = (
                sum((d.food_reachable >= 0.5) == d.food_truth for d in decisions) / n
            )

            shown_game, shown_decision = game.snapshot(), decision.to_dict()
            if display:
                from .ui import compose

                display.show(compose(shown_game, shown_decision, stats))
            if on_step:
                on_step(shown_game, shown_decision, dict(stats))
            recorder.write(
                {
                    "type": "frame",
                    "at": shown - started,
                    "game": shown_game,
                    "decision": shown_decision,
                    "stats": dict(stats),
                }
            )
            if fps:
                time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - now)))

            game.step(decision.executed)
            best_length = max(best_length, len(game.body))
            stats["best"] = max(stats["best"], game.score)
            if not game.alive or game.won:
                deaths += not game.alive
                recorder.write(
                    {
                        "type": "round_end",
                        "at": time.perf_counter() - started,
                        "game": game.snapshot(),
                    }
                )
                if not guarded:
                    break
                if display:
                    from .ui import compose

                    display.show(compose(game.snapshot(), {}, stats))
                    time.sleep(1.0)
                stats["round"] += 1
                game = SnakeGame(width, height, seed + stats["round"] - 1, initial_length)
    except KeyboardInterrupt:
        pass
    finally:
        seconds = time.perf_counter() - started
        summary = _summarise(
            decisions,
            game,
            backend,
            served,
            prompt,
            guarded,
            seed,
            best_length,
            seconds,
            rounds=stats["round"],
            deaths=deaths,
            best_score=max(stats["best"], game.score),
        )
        recorder.write({"type": "end", "summary": summary.to_dict(), "game": game.snapshot()})
        recorder.close()
    return summary


def _summarise(
    decisions,
    game,
    backend,
    model,
    prompt,
    guarded,
    seed,
    best_length,
    seconds,
    *,
    rounds,
    deaths,
    best_score,
):
    n = len(decisions)
    inference = [d.inference_ms for d in decisions]
    route_p = [1.0 - d.dead_end_risk for d in decisions]
    food_p = [d.food_reachable for d in decisions]
    route_t = [d.route_truth for d in decisions]
    food_t = [d.food_truth for d in decisions]

    def acc(ps, ts):
        return sum((p >= 0.5) == t for p, t in zip(ps, ts, strict=True)) / n if n else 0.0

    def brier(ps, ts):
        return sum((p - float(t)) ** 2 for p, t in zip(ps, ts, strict=True)) / n if n else 0.0

    return RunSummary(
        backend=backend,
        model=model,
        prompt=prompt,
        guarded=guarded,
        board=f"{game.width}x{game.height}",
        seed=seed,
        steps=n,
        rounds=rounds,
        deaths=deaths,
        score=game.score,
        best_score=best_score,
        best_length=best_length,
        alive=game.alive,
        won=game.won,
        death_reason=game.death_reason,
        seconds=seconds,
        decisions_per_second=n / seconds if seconds else 0.0,
        inference_ms_p50=statistics.median(inference) if inference else 0.0,
        inference_ms_p95=_pct(inference, 0.95),
        mean_input_tokens=statistics.mean(d.input_tokens for d in decisions) if n else 0.0,
        cost_usd=sum(pricing.cost_usd(model, d.input_tokens, d.output_tokens) for d in decisions),
        interventions=sum(d.intervened for d in decisions),
        intervention_rate=sum(d.intervened for d in decisions) / n if n else 0.0,
        raw_safe_rate=sum(d.proposed in d.safe_directions for d in decisions) / n if n else 0.0,
        route_accuracy=acc(route_p, route_t),
        route_brier=brier(route_p, route_t),
        food_accuracy=acc(food_p, food_t),
        food_brier=brier(food_p, food_t),
    )


def load_record(path: str | Path) -> tuple[dict, list[dict]]:
    """Read the latest appended run, requiring strictly increasing frame timestamps."""
    metadata, frames = None, []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("type") == "metadata":
            metadata = event
            frames = []
        elif event.get("type") == "frame":
            frames.append(event)
    if metadata is None or not frames:
        raise ValueError(f"{path}: a recording needs a metadata line and at least one frame")
    times = [f["at"] for f in frames]
    if times != sorted(times) or len(set(times)) != len(times) or times[0] < 0:
        raise ValueError(f"{path}: frame timestamps must strictly increase")
    return metadata, frames


def replay(path: str | Path, display, *, speed: float = 1.0, keys=None) -> int:
    """Play a recording back on `display` at `speed` times original pace. Returns frames shown."""
    from .ui import compose

    _, frames = load_record(path)
    started = time.perf_counter()
    shown = 0
    for frame in frames:
        target = frame["at"] / speed
        while (elapsed := time.perf_counter() - started) < target:
            if keys and "q" in keys().lower():
                return shown
            time.sleep(min(0.02, target - elapsed))
        display.show(compose(frame["game"], frame["decision"], {**frame["stats"], "replay": True}))
        shown += 1
    return shown
