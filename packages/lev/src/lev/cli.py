"""`lev` — inspect the plan, serve the model, fit calibration.

Every subcommand that costs GPU time prints its budget first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_plan(args: argparse.Namespace) -> None:
    """Print the training budget for a preset without spending it."""
    from .train.config import PRESETS

    if args.preset not in PRESETS:
        sys.exit(f"unknown preset {args.preset!r}; have {', '.join(sorted(PRESETS))}")
    config = PRESETS[args.preset]
    config.validate()
    print(config.summary())


def cmd_route(args: argparse.Namespace) -> None:
    """Show which readout mode each question in a request would take."""
    from transformers import AutoTokenizer

    from .router import route_all
    from .types import SystemOneRequest

    request = SystemOneRequest.model_validate(json.loads(Path(args.request).read_text()))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    for name, r in route_all(request.questions, tokenizer, args.max_label_options).items():
        codes = f"  codes={r.codes[:6]}{'...' if r.codes and len(r.codes) > 6 else ''}"
        print(f"{name:<24} mode {r.mode.value}  ({r.reason}){codes if r.codes else ''}")


def cmd_check_data(args: argparse.Namespace) -> None:
    """Fail loudly if a training mixture touches an evaluation subset."""
    from .data import BLOCKED_SUBSETS, ContaminationError, assert_clean

    names = [n.strip() for n in Path(args.sources).read_text().splitlines() if n.strip()]
    try:
        assert_clean(names)
    except ContaminationError as exc:
        sys.exit(str(exc))
    print(f"{len(names)} sources clean against {len(BLOCKED_SUBSETS)} blocked subsets")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .server import create_app

    uvicorn.run(
        create_app(args.checkpoint, args.model_cache, args.calibration, args.model),
        host=args.host,
        port=args.port,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="lev", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan", help="print a preset's training budget")
    p.add_argument("--preset", default="4b")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("route", help="show the readout mode per question")
    p.add_argument("request", help="path to a /v1/systemone request JSON")
    p.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    p.add_argument("--max-label-options", type=int, default=None)
    p.set_defaults(func=cmd_route)

    p = sub.add_parser("check-data", help="contamination guard over a source list")
    p.add_argument("sources", help="file with one dataset name per line")
    p.set_defaults(func=cmd_check_data)

    p = sub.add_parser("serve", help="serve /v1/systemone")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    p.add_argument("--model-cache", default=None)
    p.add_argument("--calibration", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
