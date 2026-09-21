"""Command line entry point: `levbench eval | sweep | compare`."""

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
    """Find the repo's `data/` by walking up from this file.

    A fixed `parents[n]` breaks the moment the package moves inside the repo,
    which is exactly what happened when levbench became a workspace member.
    """
    for parent in Path(__file__).resolve().parents:
        if (candidate := parent / "data").is_dir():
            return candidate
    raise FileNotFoundError("could not locate the repo's data/ directory")


DEFAULT_STATE = repo_data_dir() / "sample_policy.md"


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


def cmd_eval(args: argparse.Namespace) -> None:
    _require_key(args.backend, args.base_url)
    items, questions = dataset()
    if args.limit:
        items = items[: args.limit]
    client, model = runner.build_client(args.backend, args.model, args.base_url)
    report = runner.run_eval(client, args.backend, model, items, questions)
    print(runner.format_report(report))


def cmd_sweep(args: argparse.Namespace) -> None:
    _require_key(args.backend, args.base_url)
    state = Path(args.state).read_text()
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="levbench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("eval", help="accuracy and calibration on the labelled set")
    p_eval.add_argument("--backend", choices=["jev", "lev", "anthropic"], default="jev")
    p_eval.add_argument("--model", default=None)
    p_eval.add_argument("--limit", type=int, default=None)
    p_eval.add_argument(
        "--base-url",
        default=None,
        help="override the endpoint (defaults to localhost:8000 for --backend lev)",
    )
    p_eval.set_defaults(func=cmd_eval)

    p_sweep = sub.add_parser("sweep", help="batched vs split cost and latency")
    p_sweep.add_argument("--backend", choices=["jev", "lev", "anthropic"], default="jev")
    p_sweep.add_argument("--model", default=None)
    p_sweep.add_argument("--state", default=str(DEFAULT_STATE))
    p_sweep.add_argument("--counts", default="1,2,4,8,13")
    p_sweep.add_argument(
        "--base-url",
        default=None,
        help="override the endpoint (defaults to localhost:8000 for --backend lev)",
    )
    p_sweep.set_defaults(func=cmd_sweep)

    p_cmp = sub.add_parser("compare", help="run both backends and diff them")
    p_cmp.add_argument("--jev-model", default=None)
    p_cmp.add_argument("--anthropic-model", default="claude-opus-5")
    p_cmp.add_argument("--limit", type=int, default=None)
    p_cmp.set_defaults(func=cmd_compare)

    p_conf = sub.add_parser("confidence", help="identify the server's confidence formula")
    p_conf.add_argument("--model", default=None)
    p_conf.add_argument("--limit", type=int, default=None)
    p_conf.add_argument(
        "--base-url",
        default=None,
        help="override the endpoint (defaults to localhost:8000 for --backend lev)",
    )
    p_conf.set_defaults(func=cmd_confidence)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
