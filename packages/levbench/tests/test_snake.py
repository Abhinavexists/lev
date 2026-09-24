"""Snake: rules, prompts, shield and scoring, all offline through a stub server."""

from __future__ import annotations

import json
from collections import deque
from types import SimpleNamespace

import pytest
from levbench.snake import (
    DIRECTIONS,
    ModelPolicy,
    PlannerClient,
    SnakeGame,
    load_record,
    play,
    replay,
)
from levbench.snake.game import hamiltonian_cycle
from levbench.snake.policy import describe
from levbench.snake.ui import compose, layout_size, status_line


class StubServer:
    """Answers from a chosen direction and two Noul probabilities, SDK-shaped."""

    def __init__(self, pick=None, route=0.9, food=0.9, follow_planner=False, invert=False):
        self.pick, self.route, self.food = pick, route, food
        self.follow_planner, self.invert = follow_planner, invert
        self.calls: list[tuple[str, dict]] = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        options = list(questions["move"].criteria)
        if self.follow_planner:
            pick = next((o for o in options if "Best" in questions["move"].criteria[o]), options[0])
        else:
            pick = self.pick or options[0]
        probs = {o: 0.04 for o in options}
        probs[pick] = 1.0 - 0.04 * (len(options) - 1)
        route, food = self.route, self.food
        if self.invert:
            # Answer both Nouls from the state text, wrongly.
            route = 0.1 if "Safe route: yes" in state else 0.9
            food = 0.1 if "reachable through empty cells: yes" in state else 0.9
        return SimpleNamespace(
            answers={
                "move": SimpleNamespace(type="choice", probabilities=probs, choice=pick),
                "risk": SimpleNamespace(type="noul", noul=route),
                "food": SimpleNamespace(type="noul", noul=food),
            },
            usage=SimpleNamespace(input_tokens=42, output_tokens=0),
            model="stub-1",
        )


class TestRules:
    def test_cycle_visits_every_cell_once_with_adjacent_steps(self):
        cycle = hamiltonian_cycle(8, 6)
        assert len(cycle) == 48 and len(set(cycle)) == 48
        for a, b in zip(cycle, cycle[1:] + cycle[:1], strict=True):
            assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1

    def test_odd_by_odd_and_tiny_boards_are_rejected(self):
        with pytest.raises(ValueError):
            hamiltonian_cycle(5, 5)
        with pytest.raises(ValueError):
            hamiltonian_cycle(3, 8)

    def test_same_seed_same_game(self):
        a, b = SnakeGame(seed=3), SnakeGame(seed=3)
        for _ in range(40):
            move = max((m for m in a.moves() if m.safe), key=lambda m: m.advance).direction
            a.step(move)
            b.step(move)
        assert a.snapshot() == b.snapshot()

    def test_following_the_planner_never_dies(self):
        game = SnakeGame(width=8, height=6, seed=1, initial_length=3)
        for _ in range(500):
            if game.won:
                break
            safe = [m for m in game.moves() if m.safe]
            assert safe, "the cycle planner must always leave a safe move"
            game.step(max(safe, key=lambda m: m.advance).direction)
        assert game.alive

    def test_eating_grows_and_scores(self):
        game = SnakeGame(width=8, height=6, seed=1, initial_length=3)
        length = len(game.body)
        while True:
            eating = [m for m in game.moves() if m.safe and m.eats]
            if eating:
                assert game.step(eating[0].direction) is True
                break
            game.step(max((m for m in game.moves() if m.safe), key=lambda m: m.advance).direction)
        assert len(game.body) == length + 1 and game.score == 1

    def test_wall_and_reverse_are_illegal(self):
        game = SnakeGame(width=8, height=6, initial_length=3)
        game.body = deque([(0, 0), (1, 0), (2, 0)])  # head in the top-left corner
        assert game.legal_reason("LEFT") == "wall"
        assert game.legal_reason("UP") == "wall"
        assert game.legal_reason("RIGHT") == "reverse"
        assert game.legal_reason("DOWN") == "legal"


class TestPrompts:
    def test_compact_state_states_the_noul_answers(self):
        state, questions, planner = describe(SnakeGame(), "compact")
        assert state == "Safe route: yes. Food reachable through empty cells: yes."
        assert questions["risk"].instructions == "Is a safe route available?"
        assert questions["food"].instructions == "Is food reachable through empty cells?"
        assert set(questions["move"].criteria) == set(DIRECTIONS)
        assert planner["preferred"] in planner["safe"]

    def test_detailed_wording_matches_laya(self):
        state, questions, _ = describe(SnakeGame(), "detailed")
        assert state.startswith("Snake game. ") and "There is a safe route forward." in state
        assert questions["move"].instructions == (
            "Select the safest move with best progress toward food. Avoid collisions."
        )
        assert any(v.startswith("Collision:") for v in questions["move"].criteria.values()), (
            "the reverse move must be described as a collision"
        )

    def test_unknown_prompt_is_refused(self):
        with pytest.raises(ValueError):
            describe(SnakeGame(), "verbose")


class TestShield:
    def unsafe_direction(self, game):
        return next(m.direction for m in game.moves() if not m.safe)

    def test_shield_executes_best_safe_move_and_keeps_raw_probabilities(self):
        game = SnakeGame()
        bad = self.unsafe_direction(game)
        decision = ModelPolicy(StubServer(pick=bad)).decide(game)
        assert decision.proposed == bad
        assert decision.executed in decision.safe_directions
        assert decision.intervened
        assert max(decision.probabilities, key=decision.probabilities.get) == bad

    def test_unassisted_executes_the_raw_choice(self):
        game = SnakeGame()
        bad = self.unsafe_direction(game)
        decision = ModelPolicy(StubServer(pick=bad), guarded=False).decide(game)
        assert decision.executed == bad and not decision.intervened

    def test_truth_fields_come_from_the_planner(self):
        decision = ModelPolicy(StubServer(route=0.2, food=0.3)).decide(SnakeGame())
        assert decision.route_truth is True and decision.food_truth is True
        assert decision.dead_end_risk == pytest.approx(0.8)
        assert decision.food_reachable == pytest.approx(0.3)
        assert decision.served_by == "stub-1" and decision.input_tokens == 42

    def test_invalid_probability_is_refused(self):
        class Broken(StubServer):
            def system_one(self, state, questions):
                out = super().system_one(state, questions)
                out.answers["risk"].noul = float("nan")
                return out

        with pytest.raises(ValueError, match="invalid probability"):
            ModelPolicy(Broken()).decide(SnakeGame())


class TestPlay:
    def test_runs_the_requested_steps_and_summarises(self, tmp_path):
        record = tmp_path / "run.jsonl"
        summary = play(
            StubServer(follow_planner=True),
            "lev",
            "local",
            width=8,
            height=6,
            initial_length=3,
            steps=25,
            record=record,
        )
        assert summary.steps == 25 and summary.alive and summary.interventions == 0
        assert summary.model == "stub-1", "the summary reports what the server said it served"
        assert summary.route_accuracy == 1.0 and summary.food_accuracy == 1.0
        assert summary.mean_input_tokens == 42
        assert summary.rounds == 1 and summary.deaths == 0
        events = [json.loads(line) for line in record.read_text().splitlines()]
        kinds = [e["type"] for e in events]
        assert kinds[0] == "metadata" and kinds[-1] == "end" and kinds.count("frame") == 25
        frame = events[1]
        assert frame["game"]["ticks"] == 0, "the board is shown before the announced action"
        assert "probabilities" in frame["decision"] and frame["at"] >= 0
        assert events[-1]["summary"]["steps"] == 25

    def test_noul_scoring_catches_a_model_that_contradicts_its_state(self):
        summary = play(
            StubServer(follow_planner=True, invert=True),
            "lev",
            "local",
            width=8,
            height=6,
            initial_length=3,
            steps=20,
        )
        assert summary.route_accuracy == 0.0
        assert summary.route_brier > 0.5

    def test_unassisted_run_ends_on_death(self):
        summary = play(
            StubServer(pick="UP"),
            "lev",
            "local",
            width=8,
            height=6,
            initial_length=3,
            steps=200,
            guarded=False,
        )
        assert not summary.alive and summary.death_reason in {"wall", "body", "reverse"}
        assert summary.steps < 200 and summary.deaths == 1

    def test_summary_lines_render(self):
        summary = play(
            StubServer(follow_planner=True),
            "lev",
            "local",
            width=8,
            height=6,
            initial_length=3,
            steps=5,
        )
        text = "\n".join(summary.lines())
        assert "decisions/s" in text and "noul vs planner truth" in text


class TestPlannerBackend:
    def test_planner_never_needs_the_shield_and_reads_its_state(self):
        summary = play(
            PlannerClient(), "planner", "planner", width=8, height=6, initial_length=3, steps=60
        )
        assert summary.interventions == 0 and summary.raw_safe_rate == 1.0
        assert summary.route_accuracy == 1.0 and summary.food_accuracy == 1.0
        assert summary.alive


class TestRounds:
    def test_guarded_run_continues_past_a_cleared_board(self):
        """A 4x4 board with a planner-perfect player is cleared quickly; the run
        must roll into round 2 on the next seed rather than stop."""
        summary = play(
            PlannerClient(), "planner", "planner", width=4, height=4, initial_length=2, steps=60
        )
        assert summary.rounds >= 2 and summary.steps == 60
        assert summary.best_score >= 1


class TestDisplay:
    def test_layout_fits_the_default_board(self):
        assert layout_size(24, 16) == (104, 36)

    def test_compose_shows_shield_truth_and_backend(self):
        game = SnakeGame(width=8, height=6, initial_length=3)
        bad = next(m.direction for m in game.moves() if not m.safe)
        decision = ModelPolicy(StubServer(pick=bad, route=0.3, food=0.9)).decide(game).to_dict()
        stats = {
            "backend": "lev",
            "model": "stub-1",
            "network": "host",
            "round": 2,
            "best": 3,
            "interventions": 1,
            "steps_per_second": 1.5,
            "elapsed": 61,
            "guarded": True,
            "route_acc": 0.5,
            "food_acc": 1.0,
        }
        text = compose(game.snapshot(), decision, stats).text()
        assert "SHIELD" in text and "ROUND 02" in text
        assert "truth: route exists" in text and "truth: yes" in text
        assert "lev · stub-1" in text and "01:01" in text
        assert "unsafe" in text

    def test_status_line_is_one_line(self):
        game = SnakeGame(width=8, height=6, initial_length=3)
        decision = ModelPolicy(StubServer(follow_planner=True)).decide(game).to_dict()
        line = status_line(
            game.snapshot(), decision, {"steps": 1, "round": 1, "steps_per_second": 2.0}
        )
        assert "\n" not in line and "risk" in line


class TestReplay:
    def test_recording_replays_every_frame(self, tmp_path):
        record = tmp_path / "run.jsonl"
        play(
            PlannerClient(),
            "planner",
            "planner",
            width=8,
            height=6,
            initial_length=3,
            steps=8,
            record=record,
        )
        metadata, frames = load_record(record)
        assert metadata["format"] == "levbench-snake-v1" and len(frames) == 8

        class Screen:
            def __init__(self):
                self.frames = []

            def show(self, canvas):
                self.frames.append(canvas.text())

        screen = Screen()
        assert replay(record, screen, speed=1000.0) == 8
        assert "RECORDED RUN" in screen.frames[0]

    def test_broken_timestamps_are_rejected(self, tmp_path):
        bad = tmp_path / "bad.jsonl"
        bad.write_text(
            json.dumps({"type": "metadata", "format": "x"})
            + "\n"
            + json.dumps({"type": "frame", "at": 2.0, "game": {}, "decision": {}, "stats": {}})
            + "\n"
            + json.dumps({"type": "frame", "at": 1.0, "game": {}, "decision": {}, "stats": {}})
            + "\n"
        )
        with pytest.raises(ValueError, match="strictly increase"):
            load_record(bad)

    def test_reusing_a_recording_path_replays_the_latest_run(self, tmp_path):
        record = tmp_path / "run.jsonl"
        for seed, steps in ((7, 8), (8, 3)):
            play(
                PlannerClient(),
                "planner",
                "planner",
                width=8,
                height=6,
                seed=seed,
                initial_length=3,
                steps=steps,
                record=record,
            )
        metadata, frames = load_record(record)
        assert metadata["settings"]["seed"] == 8
        assert len(frames) == 3
        assert all(frame["game"]["seed"] == 8 for frame in frames)
        events = [json.loads(line) for line in record.read_text().splitlines()]
        assert sum(event["type"] == "frame" for event in events) == 11

    def test_an_empty_latest_run_does_not_borrow_previous_frames(self, tmp_path):
        record = tmp_path / "run.jsonl"
        events = [
            {"type": "metadata", "model": "previous"},
            {"type": "frame", "at": 1.0},
            {"type": "metadata", "model": "latest"},
        ]
        record.write_text("\n".join(json.dumps(event) for event in events) + "\n")
        with pytest.raises(ValueError, match="at least one frame"):
            load_record(record)
