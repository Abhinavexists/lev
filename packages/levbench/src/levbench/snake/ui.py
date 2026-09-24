"""A fixed-cell terminal composition, shared by the live display and replay.

laya-mlx's layout (Apache-2.0), with the right-hand panel adapted to a client
of a remote server: the backend and served model replace the local engine
lines, and the planner's ground truth sits beside the model's two estimates.
`rich` is optional: without it the loop prints one status line per step.
"""

from __future__ import annotations

import select
import sys
from urllib.parse import urlparse

from .game import DIRECTIONS

BG = "#090f13"
FG = "#e3f3ef"
MUTED = "#68868c"
DIM = "#20353c"
GREEN = "#62f5b5"
AMBER = "#ffce73"
RED = "#ff7c8c"
CYAN = "#8ad8e9"

DIGITS = {
    "0": ("█▀█", "█ █", "▀▀▀"),
    "1": ("▄█ ", " █ ", "▀▀▀"),
    "2": ("▀▀█", "█▀▀", "▀▀▀"),
    "3": ("▀▀█", "▀▀█", "▀▀▀"),
    "4": ("█ █", "▀▀█", "  ▀"),
    "5": ("█▀▀", "▀▀█", "▀▀▀"),
    "6": ("█▀▀", "█▀█", "▀▀▀"),
    "7": ("▀▀█", "  █", "  ▀"),
    "8": ("█▀█", "█▀█", "▀▀▀"),
    "9": ("█▀█", "▀▀█", "▀▀▀"),
}


class Canvas:
    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.chars = [[" "] * width for _ in range(height)]
        self.styles = [[FG] * width for _ in range(height)]

    def put(self, row: int, column: int, text, color: str = FG) -> None:
        if not 0 <= row < self.height:
            return
        for offset, character in enumerate(str(text)):
            x = column + offset
            if 0 <= x < self.width:
                self.chars[row][x], self.styles[row][x] = character, color

    def bar(self, row: int, column: int, value: float, length: int = 20, color: str = GREEN):
        count = round(max(0.0, min(1.0, value)) * length)
        self.put(row, column, "━" * length, DIM)
        self.put(row, column, "━" * count, color)

    def number(self, row: int, column: int, value: int, color: str = GREEN) -> None:
        for index, digit in enumerate(f"{min(value, 999):03d}"):
            for line, glyphs in enumerate(DIGITS[digit]):
                self.put(row + line, column + index * 4, glyphs, color)

    def text(self) -> str:
        return "\n".join("".join(row).rstrip() for row in self.chars)

    def rich_text(self):
        from rich.text import Text

        output = Text(no_wrap=True, overflow="crop", style=f"{FG} on {BG}")
        for row, (chars, styles) in enumerate(zip(self.chars, self.styles, strict=True)):
            begin = 0
            for end in range(1, self.width + 1):
                if end == self.width or styles[end] != styles[begin]:
                    output.append("".join(chars[begin:end]), style=styles[begin])
                    begin = end
            if row + 1 != self.height:
                output.append("\n")
        return output


def layout_size(width: int, height: int) -> tuple[int, int]:
    return max(104, width * 2 + 50), max(36, height + 20)


def network_label(backend: str, base_url: str | None) -> str:
    if backend == "planner":
        return "NONE · PLANNER"
    if backend == "jev" and not base_url:
        return "api.typesafe.ai"
    if base_url:
        return urlparse(base_url).netloc or base_url
    return "localhost"


def compose(game: dict, decision: dict, stats: dict) -> Canvas:
    """Probabilities describe the displayed board, before its announced next step."""
    width, height = layout_size(game["width"], game["height"])
    c = Canvas(width, height)
    left, right, top = 3, max(58, game["width"] * 2 + 10), 6
    side = width - right - 4
    bottom = top + game["height"] + 1

    if stats.get("paused"):
        state = "PAUSED"
    elif game["won"]:
        state = "BOARD CLEAR"
    elif not game["alive"]:
        state = "GAME OVER"
    else:
        state = "LIVE"
    if stats.get("replay") and state == "LIVE":
        state = "RECORDED RUN · 1×"
    c.put(1, left, "LEV  /  SYSTEM ONE", MUTED)
    c.put(1, width - len(state) - 3, state, GREEN if game["alive"] else RED)
    c.put(2, left, "─" * (width - 6), DIM)
    c.put(4, left, "S N A K E", FG)
    c.put(4, left + 31, f"ROUND {stats.get('round', 1):02d}", MUTED)

    c.put(top, left, "┌" + "─" * (game["width"] * 2) + "┐", DIM)
    c.put(bottom, left, "└" + "─" * (game["width"] * 2) + "┘", DIM)
    for y in range(game["height"]):
        c.put(top + 1 + y, left, "│", DIM)
        c.put(top + 1 + y, left + game["width"] * 2 + 1, "│", DIM)
        c.put(top + 1 + y, left + 1, "· " * game["width"], "#13272e")
    body = game["body"]
    for index, (x, y) in reversed(list(enumerate(body))):
        fraction = 1 - index / max(1, len(body))
        color = (
            "#dcfff0"
            if index == 0
            else f"#{int(18 + 64 * fraction):02x}{int(73 + 150 * fraction):02x}"
            f"{int(57 + 102 * fraction):02x}"
        )
        c.put(top + y + 1, left + 1 + 2 * x, "██", color)
    if game["food"] is not None:
        x, y = game["food"]
        c.put(top + y + 1, left + 1 + 2 * x, "● ", AMBER)

    for offset, label, value, color in (
        (0, "SCORE", game["score"], GREEN),
        (18, "LENGTH", game["length"], FG),
        (36, "BEST", stats.get("best", game["score"]), MUTED),
    ):
        c.put(bottom + 2, left + offset, label, MUTED)
        c.number(bottom + 3, left + offset, value, color)
    fill = game["length"] / (game["width"] * game["height"])
    c.bar(bottom + 7, left, fill, 41)
    c.put(bottom + 7, left + 43, f"{100 * fill:4.1f}%", MUTED)

    c.put(4, right, f"{stats.get('backend', 'lev')} · {stats.get('model', '')}"[:side], GREEN)
    c.put(5, right, f"{stats.get('network', '')} · prompt {stats.get('prompt', 'compact')}", MUTED)
    c.put(7, right, "NEXT MOVE", FG)
    c.put(7, right + 15, "MODEL PROBABILITIES", MUTED)
    probabilities = decision.get("probabilities", {})
    safe = set(decision.get("safe_directions", DIRECTIONS))
    for index, direction in enumerate(DIRECTIONS):
        row = 9 + index
        probability = probabilities.get(direction, 0.0)
        selected = direction == decision.get("proposed")
        color = GREEN if selected else MUTED
        c.put(row, right, f"{'›' if selected else ' '} {direction:<5}", color)
        c.put(row, right + 9, "░" * 18, DIM)
        c.put(row, right + 9, "█" * round(probability * 18), color)
        c.put(row, right + 29, f"{probability:.2f}", color)
        if decision and direction not in safe:
            c.put(row, right + 35, "unsafe", RED if selected else DIM)
    c.put(14, right, "EXECUTING", MUTED)
    c.put(14, right + 12, decision.get("executed", "—"), GREEN)
    if decision.get("intervened"):
        c.put(14, right + 20, "SHIELD", AMBER)

    risk = decision.get("dead_end_risk", 0.0)
    risk_color = AMBER if risk < 0.5 else RED
    c.put(16, right, "DEAD-END RISK", MUTED)
    c.put(16, right + 15, "model", MUTED)
    if decision:
        truth = "truth: route exists" if decision.get("route_truth") else "truth: TRAPPED"
        c.put(16, right + 22, truth, GREEN if decision.get("route_truth") else RED)
    c.bar(17, right, risk, min(24, side - 9), risk_color)
    c.put(17, right + 29, f"{risk:.2f}", risk_color)

    food = decision.get("food_reachable", 0.0)
    c.put(19, right, "FOOD REACHABLE", MUTED)
    c.put(19, right + 15, "model", MUTED)
    if decision:
        truth = "truth: yes" if decision.get("food_truth") else "truth: no"
        c.put(19, right + 22, truth, GREEN if decision.get("food_truth") else RED)
    c.bar(20, right, food, min(24, side - 9), CYAN)
    c.put(20, right + 29, f"{food:.2f}", CYAN)

    c.put(22, right, "INFERENCE", MUTED)
    c.put(22, right + 18, f"{decision.get('inference_ms', 0.0):6.0f} ms", FG)
    c.put(23, right, "DECISIONS", MUTED)
    c.put(23, right + 18, f"{stats.get('steps_per_second', 0.0):6.2f} /s", FG)
    c.put(24, right, "INPUT TOKENS", MUTED)
    c.put(24, right + 18, str(decision.get("input_tokens", 0)), FG)
    c.put(25, right, "NETWORK", MUTED)
    c.put(25, right + 18, stats.get("network", "")[: side - 18], GREEN)
    c.put(26, right, "NOUL vs TRUTH", MUTED)
    c.put(
        26,
        right + 18,
        f"route {stats.get('route_acc', 0.0):4.0%}   food {stats.get('food_acc', 0.0):4.0%}",
        FG,
    )
    guarded = stats.get("guarded", True)
    c.put(28, right, "model + cycle safety" if guarded else "model · shield OFF", MUTED)
    c.put(29, right, f"Shield interventions  {stats.get('interventions', 0):04d}", AMBER)
    c.put(height - 3, left, "─" * (width - 6), DIM)
    c.put(height - 2, left, "SPACE pause   +/- speed   R reset   Q quit", MUTED)
    elapsed = stats.get("elapsed", 0.0)
    clock = f"{int(elapsed) // 60:02d}:{int(elapsed) % 60:02d}"
    c.put(height - 2, right, f"ESTIMATES BY THE MODEL       {clock}", MUTED)
    return c


def status_line(game: dict, decision: dict, stats: dict) -> str:
    move = f"{decision.get('executed', '-')}{' SHIELD' if decision.get('intervened') else ''}"
    route = "ok" if decision.get("route_truth") else "TRAPPED"
    food = "yes" if decision.get("food_truth") else "no"
    return (
        f"step {stats.get('steps', 0):>5}  round {stats.get('round', 1):>2}  "
        f"score {game['score']:>3}  len {game['length']:>3}  {move:<12}"
        f"{decision.get('inference_ms', 0.0):>6.0f} ms  "
        f"{stats.get('steps_per_second', 0.0):>5.2f}/s  "
        f"risk {decision.get('dead_end_risk', 0.0):.2f}({route})  "
        f"food {decision.get('food_reachable', 0.0):.2f}({food})"
    )


class Keyboard:
    """Non-blocking single-key reads from a TTY; a no-op anywhere else."""

    def __enter__(self):
        self.saved = None
        if sys.stdin.isatty():
            import termios
            import tty

            self.fd = sys.stdin.fileno()
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def read(self) -> str:
        import os

        if self.saved and select.select([sys.stdin], [], [], 0)[0]:
            return os.read(self.fd, 128).decode(errors="ignore")
        return ""

    def __exit__(self, *_):
        if self.saved:
            import termios

            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)


class LiveDisplay:
    """The rich full-screen view. `available()` says whether it can run here."""

    def __init__(self, alt_screen: bool = True):
        from rich.console import Console
        from rich.live import Live

        self.console = Console(style=f"on {BG}", highlight=False)
        self.live = Live(
            console=self.console,
            screen=alt_screen,
            auto_refresh=False,
            vertical_overflow="crop",
        )

    @staticmethod
    def available() -> bool:
        try:
            from rich.console import Console
        except ImportError:
            return False
        return Console().is_terminal

    def __enter__(self):
        self.live.__enter__()
        return self

    def __exit__(self, *exc):
        return self.live.__exit__(*exc)

    def fits(self, game_width: int, game_height: int) -> tuple[bool, tuple[int, int]]:
        need = layout_size(game_width, game_height)
        return (self.console.width >= need[0] and self.console.height >= need[1]), need

    def show(self, canvas: Canvas) -> None:
        self.live.update(canvas.rich_text(), refresh=True)

    def message(self, text: str) -> None:
        self.live.update(text, refresh=True)
