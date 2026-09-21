"""`lev` — inspect the plan, serve the model, fit calibration.

Every subcommand that costs GPU time prints its budget first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _measure_tokens(config, data_dir: str, sample: int = 1500) -> dict:
    """Tokenise real rendered prompts and report the length distribution.

    The budget rests on `avg_tokens_per_example`, and a guessed value is the
    easiest way to be wrong by an order of magnitude about how long a run takes.
    """
    import statistics

    from transformers import AutoTokenizer

    from .data.build import load_split
    from .data.splits import Split
    from .train.collate import ModeBatcher, render

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    rows = load_split(data_dir, Split.TRAIN)[:sample]
    batcher = ModeBatcher(tokenizer, batch_size=1)
    lengths = sorted(
        len(tokenizer.encode(render(e, batcher.route_for(e).codes), add_special_tokens=False))
        for e in rows
    )
    return {
        "n": len(lengths),
        "mean": round(statistics.mean(lengths)),
        "median": lengths[len(lengths) // 2],
        "p95": lengths[int(0.95 * len(lengths))],
        "max": lengths[-1],
    }


def cmd_plan(args: argparse.Namespace) -> None:
    """Print the training budget for a preset without spending it."""
    from .train.config import PRESETS

    if args.preset not in PRESETS:
        sys.exit(f"unknown preset {args.preset!r}; have {', '.join(sorted(PRESETS))}")
    config = PRESETS[args.preset]
    if args.data:
        measured = _measure_tokens(config, args.data)
        print(
            f"measured over {measured['n']:,} real prompts: "
            f"mean {measured['mean']}  median {measured['median']}  "
            f"p95 {measured['p95']}  max {measured['max']} tokens"
        )
        if measured["max"] > config.max_seq_len:
            print(
                f"  note: {measured['max']} > max_seq_len {config.max_seq_len}; "
                f"the longest prompts will be truncated"
            )
        config.avg_tokens_per_example = measured["mean"]
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


def _preset_names() -> list[str]:
    """Imported lazily: `train.config` is cheap, but keeping every CLI import
    behind its command is what lets `lev route` run without the train extra."""
    from .train.config import PRESETS

    return sorted(PRESETS)


def cmd_data_build(args: argparse.Namespace) -> None:
    """Download every source, split it, and write the mixture to disk."""
    from .data.build import build_dataset

    manifest = build_dataset(
        args.out,
        limit_per_source=args.limit_per_source,
        n_examples=args.n_examples,
        schema_first_fraction=args.schema_first_fraction,
        abstain_fraction=args.abstain_fraction,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    print(f"wrote {args.out}")
    for split, count in manifest["split_counts"].items():
        print(f"  {split:<12} {count:>8,}")
    ratio = manifest["oversample_ratio"]
    print(
        f"  unique train rows {manifest['unique_train_rows']:,} "
        f"(oversample {ratio}x)"
        + ("  <- raise --limit-per-source to lower this" if ratio > 2 else "")
    )
    print(f"  sources      {len(manifest['sources'])} ({', '.join(sorted(manifest['sources']))})")


def cmd_data_eval(args: argparse.Namespace) -> None:
    """Export the held-out split as levbench task files."""
    from .data.export_eval import export
    from .data.splits import Split

    index = export(args.data, args.out, Split(args.split), args.limit_per_source)
    print(f"wrote {index['total_items']:,} items from the {index['split']} split to {args.out}")
    for source, info in index["sources"].items():
        print(f"  {source:<20} {info['items']:>6,}  {info['type']}")
    print(f"\n  uv run levbench eval --backend lev --tasks {args.out}")


def cmd_train(args: argparse.Namespace) -> None:
    from .train.config import PRESETS
    from .train.loop import run_training

    config = PRESETS[args.preset]
    if args.output_dir:
        config.output_dir = args.output_dir
    summary = run_training(
        config, args.data, model_cache=args.model_cache, max_steps=args.max_steps
    )
    print(
        f"{summary['steps']} steps  "
        f"loss {summary['first_loss']:.4f} -> {summary['final_loss']:.4f}  "
        f"-> {summary['output_dir']}"
    )


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
    p.add_argument(
        "--data",
        default=None,
        help="measure token lengths from a built mixture instead of trusting the default",
    )
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
    p.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "a training output directory (or a step-N inside one). The LoRA "
            "adapter, the Mode B head and any calibration.json beside them are "
            "all picked up. Omit to serve the untrained base model."
        ),
    )
    p.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    p.add_argument("--model-cache", default=None)
    p.add_argument("--calibration", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("data", help="build the training mixture and the eval set")
    data_sub = p.add_subparsers(dest="data_command", required=True)

    d = data_sub.add_parser("build", help="download, split and write the mixture")
    d.add_argument("--out", default="data/mixture")
    d.add_argument(
        "--limit-per-source",
        type=int,
        default=20_000,
        help="rows sampled per source (shuffled first, never a head slice)",
    )
    d.add_argument("--n-examples", type=int, default=200_000)
    d.add_argument("--schema-first-fraction", type=float, default=0.5)
    d.add_argument("--abstain-fraction", type=float, default=0.1)
    d.add_argument("--seed", type=int, default=17)
    d.add_argument("--cache-dir", default=None)
    d.set_defaults(func=cmd_data_build)

    d = data_sub.add_parser("eval", help="export a held-out split as levbench task files")
    d.add_argument("--data", default="data/mixture")
    d.add_argument("--out", default="data/eval")
    d.add_argument("--split", default="test", choices=["test", "calibration"])
    d.add_argument("--limit-per-source", type=int, default=None)
    d.set_defaults(func=cmd_data_eval)

    p = sub.add_parser("train", help="run the LoRA fine-tune")
    p.add_argument("--preset", default="4b", choices=_preset_names())
    p.add_argument("--data", default="data/mixture")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--model-cache", default=None)
    p.add_argument("--max-steps", type=int, default=None, help="stop early; for smoke runs")
    p.set_defaults(func=cmd_train)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
