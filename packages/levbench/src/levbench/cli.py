"""levbench: benchmark any `/v1/systemone` server.

Subcommands: eval, sweep, compare, confidence, snake, replay.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from . import batching, confidence_id, runner
from .tasks import dataset

load_dotenv()


def repo_data_dir() -> Path:
    """Find the repo's `data/` by walking up from this file, so it survives the
    package moving within the repo."""
    for parent in Path(__file__).resolve().parents:
        if (candidate := parent / "data").is_dir():
            return candidate
    raise FileNotFoundError("could not locate the repo's data/ directory")


def _require_key(backend: str, base_url: str | None = None) -> None:
    if backend == "lev":
        return  # A local /v1/systemone server needs no credential.
    if backend == "jev" and base_url:
        return  # An explicitly overridden endpoint is assumed local too.
    needed = "TYPESAFE_API_KEY" if backend == "jev" else "ANTHROPIC_API_KEY"
    if not os.environ.get(needed):
        sys.exit(
            f"{needed} is not set.\n"
            + (
                "Get a key at https://console.typesafe.ai/settings/keys"
                if backend == "jev"
                else "Export your Anthropic API key."
            )
        )


def _task_files(path: str | None) -> list[Path | None]:
    """`None` for the built-in fixture, else one entry per task file.

    A generated eval is a directory with one file per source, because each
    source has its own questions.
    """
    if path is None:
        return [None]
    target = Path(path)
    if target.is_dir():
        files = sorted(f for f in target.glob("*.json") if f.name != "index.json")
        if not files:
            sys.exit(f"no task files in {path}")
        return files
    return [target]


def cmd_eval(args: argparse.Namespace) -> None:
    _require_key(args.backend, args.base_url)
    client, model = runner.build_client(args.backend, args.model, args.base_url, args.timeout)
    for task_file in _task_files(args.tasks):
        items, questions = dataset(task_file)
        if args.limit:
            items = items[: args.limit]
        if task_file is not None:
            print(f"\n=== {task_file.stem}  ({len(items)} items) ===")
        report = runner.run_eval(
            client, args.backend, model, items, questions, concurrency=args.concurrency
        )
        print(runner.format_report(report))


def cmd_sweep(args: argparse.Namespace) -> None:
    _require_key(args.backend, args.base_url)
    state = Path(args.state or repo_data_dir() / "sample_policy.md").read_text()
    client, model = runner.build_client(args.backend, args.model, args.base_url)
    counts = [int(c) for c in args.counts.split(",")]
    errors: list[tuple[int, str]] = []
    rows = batching.sweep(client, model, state, counts, errors=errors)
    print(batching.format_sweep(rows, model, len(state), errors=errors))


def cmd_compare(args: argparse.Namespace) -> None:
    for backend in ("jev", "anthropic"):
        _require_key(backend)
    items, questions = dataset()
    if args.limit:
        items = items[: args.limit]

    reports = {}
    for backend, model in (("jev", args.jev_model), ("anthropic", args.anthropic_model)):
        client, resolved = runner.build_client(backend, model)
        reports[backend] = runner.run_eval(client, backend, resolved, items, questions)
        print(runner.format_report(reports[backend]))

    jev, llm = reports["jev"], reports["anthropic"]
    print("=== head to head ===")
    for name in questions:
        agree = sum(
            1 for a, b in zip(jev.records[name], llm.records[name], strict=True) if a[1] == b[1]
        )
        print(f"{name:<14} argmax agreement {agree}/{len(items)} ({agree / len(items):.0%})")

    print()
    cost_ratio = llm.total_cost / jev.total_cost if jev.total_cost else 0.0
    speed_ratio = llm.pct(0.5) / jev.pct(0.5) if jev.pct(0.5) else 0.0
    print(
        f"cost   jev ${jev.total_cost:.6f}  vs  llm ${llm.total_cost:.6f}"
        f"   ({cost_ratio:.1f}x cheaper)"
    )
    print(
        f"p50    jev {jev.pct(0.5):.3f}s  vs  llm {llm.pct(0.5):.3f}s   ({speed_ratio:.1f}x faster)"
    )
    print(f"schema retries  jev {jev.total_schema_retries}  vs  llm {llm.total_schema_retries}")
    print(
        f"transient retries  jev {jev.total_transient_retries}"
        f"  vs  llm {llm.total_transient_retries}"
    )


def cmd_snake(args: argparse.Namespace) -> None:
    """A decision model plays Snake over `/v1/systemone`; see `levbench.snake`."""
    from contextlib import nullcontext

    from .snake import PlannerClient, play
    from .snake.ui import Keyboard, LiveDisplay, network_label, status_line

    if args.backend == "planner":
        client, model = PlannerClient(), "planner"
    else:
        _require_key(args.backend, args.base_url)
        client, model = runner.build_client(args.backend, args.model, args.base_url, args.timeout)

    live = not args.headless and LiveDisplay.available()
    if not args.headless and not live:
        print("no TTY (or rich missing: uv sync --extra demo); printing status lines", flush=True)
    display = LiveDisplay(alt_screen=not args.no_alt_screen) if live else None
    with display or nullcontext(), Keyboard() as keyboard:
        summary = play(
            client,
            args.backend,
            model,
            width=args.width,
            height=args.height,
            seed=args.seed,
            initial_length=args.initial_length,
            steps=args.steps,
            duration=args.duration,
            fps=args.fps,
            guarded=not args.unassisted,
            prompt=args.prompt,
            record=args.record,
            network=network_label(args.backend, args.base_url),
            display=display,
            keys=keyboard.read if live else None,
            on_step=None if live else (lambda g, d, st: print(status_line(g, d, st), flush=True)),
        )
    print("\n".join(summary.lines()))
    if args.record:
        print(f"recorded -> {args.record}   replay: uv run levbench replay {args.record}")


def cmd_replay(args: argparse.Namespace) -> None:
    """Play a recording back in the same display, at original speed."""
    from .snake import load_record, replay
    from .snake.ui import Keyboard, LiveDisplay

    metadata, frames = load_record(args.recording)
    if not LiveDisplay.available():
        sys.exit("replay needs a TTY with rich installed (uv sync --extra demo)")
    print(
        f"{len(frames)} frames, {frames[-1]['at']:.0f}s of {metadata['backend']} / "
        f"{metadata['model']}; Q quits",
        flush=True,
    )
    with LiveDisplay(alt_screen=not args.no_alt_screen) as display, Keyboard() as keyboard:
        shown = replay(args.recording, display, speed=args.speed, keys=keyboard.read)
    print(f"replayed {shown}/{len(frames)} frames")


def cmd_confidence(args: argparse.Namespace) -> None:
    """Work out which statistic the server's `confidence` field really is."""
    _require_key("jev", args.base_url)
    items, questions = dataset()
    if args.limit:
        items = items[: args.limit]
    client, model = runner.build_client("jev", args.model, args.base_url)

    answers: list = []
    for item in items:
        answers.extend(runner.call_once(client, item.state, questions).answers.values())

    samples = confidence_id.collect(answers)
    print(f"model: {model}   answers with a distribution: {len(samples)}\n")
    print(confidence_id.format_fits(confidence_id.identify(samples)))


BACKENDS = ["jev", "lev", "anthropic"]


def _add_base_url(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--base-url",
        default=None,
        help="override the endpoint (defaults to localhost:8000 for --backend lev)",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="levbench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    eval_parser = sub.add_parser("eval", help="accuracy and calibration on the labelled set")
    eval_parser.add_argument("--backend", choices=BACKENDS, default="jev")
    eval_parser.add_argument("--model", default=None)
    eval_parser.add_argument("--limit", type=int, default=None)
    eval_parser.add_argument(
        "--tasks",
        default=None,
        help=(
            "a task file, or a directory of them. Omit for the 24-item built-in "
            "fixture, which is a smoke test and cannot resolve better than "
            "+/-16 accuracy points. Generate a real one with `lev data eval`."
        ),
    )
    eval_parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="requests in flight at once; the lev server accepts several per container",
    )
    eval_parser.add_argument("--timeout", type=float, default=None, help="per-call seconds")
    _add_base_url(eval_parser)
    eval_parser.set_defaults(func=cmd_eval)

    sweep_parser = sub.add_parser("sweep", help="batched vs split cost and latency")
    sweep_parser.add_argument("--backend", choices=BACKENDS, default="jev")
    sweep_parser.add_argument("--model", default=None)
    sweep_parser.add_argument(
        "--state", default=None, help="a state file; default data/sample_policy.md in the repo"
    )
    sweep_parser.add_argument("--counts", default="1,2,4,8,13")
    _add_base_url(sweep_parser)
    sweep_parser.set_defaults(func=cmd_sweep)

    compare_parser = sub.add_parser("compare", help="run both backends and diff them")
    compare_parser.add_argument("--jev-model", default=None)
    compare_parser.add_argument("--anthropic-model", default="claude-opus-5")
    compare_parser.add_argument("--limit", type=int, default=None)
    compare_parser.set_defaults(func=cmd_compare)

    confidence_parser = sub.add_parser(
        "confidence", help="identify the server's confidence formula"
    )
    confidence_parser.add_argument("--model", default=None)
    confidence_parser.add_argument("--limit", type=int, default=None)
    _add_base_url(confidence_parser)
    confidence_parser.set_defaults(func=cmd_confidence)

    snake_parser = sub.add_parser("snake", help="a decision model plays Snake, one call per move")
    snake_parser.add_argument(
        "--backend",
        choices=[*BACKENDS, "planner"],
        default="lev",
        help="`planner` plays from the safety planner alone: no server, the reference row",
    )
    snake_parser.add_argument("--model", default=None)
    snake_parser.add_argument("--prompt", choices=("compact", "detailed"), default="compact")
    snake_parser.add_argument("--width", type=int, default=24)
    snake_parser.add_argument("--height", type=int, default=16)
    snake_parser.add_argument("--seed", type=int, default=7)
    snake_parser.add_argument("--initial-length", type=int, default=6)
    snake_parser.add_argument("--steps", type=int, default=None, help="stop after this many moves")
    snake_parser.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    snake_parser.add_argument(
        "--fps", type=float, default=None, help="pace moves to a budget; default waits on inference"
    )
    snake_parser.add_argument(
        "--unassisted",
        action="store_true",
        help="execute the model's raw first choice; no safety shield, deaths count",
    )
    snake_parser.add_argument("--headless", action="store_true", help="status lines, no display")
    snake_parser.add_argument(
        "--no-alt-screen",
        action="store_true",
        help="draw inline instead of on the alternate screen",
    )
    snake_parser.add_argument("--record", default=None, help="append a JSONL record of the run")
    snake_parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="per-call timeout in seconds; default 120 for --backend lev, the SDK's 10 otherwise",
    )
    _add_base_url(snake_parser)
    snake_parser.set_defaults(func=cmd_snake)

    replay_parser = sub.add_parser(
        "replay", help="play a `snake --record` file back at original speed"
    )
    replay_parser.add_argument("recording")
    replay_parser.add_argument("--speed", type=float, default=1.0, help="playback multiplier")
    replay_parser.add_argument("--no-alt-screen", action="store_true")
    replay_parser.set_defaults(func=cmd_replay)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
