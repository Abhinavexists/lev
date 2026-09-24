"""Deterministic Snake rules and a Hamiltonian-cycle safety planner.

A safe move preserves cycle order without crossing the tail or skipping food.
The shield rejects moves outside that policy, including legal moves that do
not cause an immediate collision. Rules follow laya-mlx so runs are comparable.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
VECTORS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}


def hamiltonian_cycle(width: int, height: int) -> list[tuple[int, int]]:
    """Every cell once, adjacent steps, closing back on the start."""
    if min(width, height) < 4 or (width % 2 and height % 2):
        raise ValueError("board must be >= 4 on each side with at least one even dimension")
    if height % 2:
        return [(y, x) for x, y in hamiltonian_cycle(height, width)]
    path = [(0, 0)]
    for y in range(height):
        xs = range(1, width) if y % 2 == 0 else range(width - 1, 0, -1)
        path.extend((x, y) for x in xs)
    path.extend((0, y) for y in range(height - 1, 0, -1))
    return path


@dataclass(frozen=True)
class MoveInfo:
    direction: str
    legal: bool
    safe: bool
    advance: int
    reason: str
    eats: bool


class SnakeGame:
    def __init__(self, width: int = 24, height: int = 16, seed: int = 7, initial_length: int = 6):
        self.width, self.height, self.seed = width, height, seed
        self.cycle = hamiltonian_cycle(width, height)
        self.indices = {cell: i for i, cell in enumerate(self.cycle)}
        self.capacity = width * height
        if not 2 <= initial_length < self.capacity:
            raise ValueError("initial length must be >= 2 and smaller than the board")
        self.initial_length = initial_length
        self.rng = random.Random(seed)
        start = self.indices[(width // 2, height // 2)]
        self.body: deque[tuple[int, int]] = deque(
            self.cycle[(start - i) % self.capacity] for i in range(initial_length)
        )
        self.score = self.ticks = 0
        self.alive, self.won = True, False
        self.death_reason: str | None = None
        self.food = self._spawn_food()

    @property
    def head(self) -> tuple[int, int]:
        return self.body[0]

    def _spawn_food(self):
        occupied = set(self.body)
        empty = [cell for cell in self.cycle if cell not in occupied]
        return self.rng.choice(empty) if empty else None

    def target(self, direction: str) -> tuple[int, int]:
        dx, dy = VECTORS[direction]
        return self.head[0] + dx, self.head[1] + dy

    def legal_reason(self, direction: str) -> str:
        x, y = cell = self.target(direction)
        if not (0 <= x < self.width and 0 <= y < self.height):
            return "wall"
        if cell == self.body[1]:
            return "reverse"
        occupied = set(self.body)
        if cell != self.food:
            occupied.remove(self.body[-1])  # the tail vacates on a non-growing step
        return "body" if cell in occupied else "legal"

    def moves(self) -> list[MoveInfo]:
        if not self.alive or self.won:
            return []
        head_index = self.indices[self.head]
        tail_distance = (self.indices[self.body[-1]] - head_index) % self.capacity
        food_distance = (self.indices[self.food] - head_index) % self.capacity
        out = []
        for direction in DIRECTIONS:
            reason = self.legal_reason(direction)
            legal = reason == "legal"
            target = self.target(direction)
            advance = (self.indices.get(target, head_index) - head_index) % self.capacity
            eats = target == self.food
            safe = legal
            if safe and (advance > tail_distance or (advance == tail_distance and eats)):
                safe, reason = False, "would cross the tail"
            if safe and (advance == 0 or advance > food_distance):
                safe, reason = False, "would skip the food on the safe route"
            out.append(MoveInfo(direction, legal, safe, advance, reason, eats))
        return out

    def food_reachability(self) -> tuple[bool, int]:
        """Whether the food is connected to the head through empty cells, and how many."""
        blocked = set(self.body) - {self.head}
        visited = {self.head}
        queue = deque([self.head])
        while queue:
            x, y = queue.popleft()
            for dx, dy in VECTORS.values():
                cell = (x + dx, y + dy)
                if (
                    0 <= cell[0] < self.width
                    and 0 <= cell[1] < self.height
                    and cell not in blocked
                    and cell not in visited
                ):
                    visited.add(cell)
                    queue.append(cell)
        return self.food in visited, len(visited)

    def step(self, direction: str) -> bool:
        """Advance one tick. Returns True when the food was eaten."""
        if not self.alive or self.won:
            raise RuntimeError("cannot step a finished game")
        if direction not in DIRECTIONS:
            raise ValueError(f"unknown direction {direction!r}")
        self.ticks += 1
        reason = self.legal_reason(direction)
        if reason != "legal":
            self.alive, self.death_reason = False, reason
            return False
        target = self.target(direction)
        self.body.appendleft(target)
        if target == self.food:
            self.score += 1
            if len(self.body) == self.capacity:
                self.won, self.food = True, None
            else:
                self.food = self._spawn_food()
            return True
        self.body.pop()
        return False

    def snapshot(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "body": [list(cell) for cell in self.body],
            "food": list(self.food) if self.food else None,
            "score": self.score,
            "length": len(self.body),
            "ticks": self.ticks,
            "alive": self.alive,
            "won": self.won,
            "death_reason": self.death_reason,
        }
