"""Collect checkpoint logits and fit temperatures on a dedicated calibration split."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

from ..calibrate import CalibrationProfile, fit, fit_for_transfer
from .checkpoints import CALIBRATION


def fit_profile(
    checkpoint_dir: str,
    data_dir: str,
    split: str = "calibration",
    on_complete: Callable[[], None] | None = None,
    config=None,
    method: str = "transfer",
) -> dict:
    """Fit bucket temperatures on calibration data; optionally select by family transfer.

    Save transfer comparisons and preserve the previous profile before replacing it."""
    rows = collect_logits(checkpoint_dir, data_dir, split, config=config)
    out = Path(checkpoint_dir) / CALIBRATION
    if out.is_file():
        shutil.copy2(out, out.with_name("calibration.previous.json"))

    report = None
    if method == "transfer":
        profile, report = fit_for_transfer(rows, split_name=split)
        out.with_name("calibration.report.json").write_text(json.dumps(report, indent=2) + "\n")
        for bucket, entry in sorted(report.items()):
            if "lofo_ece_rows" in entry:
                print(
                    f"  {bucket:<16} families={entry['families']:<3} "
                    f"rows T={entry['t_rows']:.3f} lofo ECE {entry['lofo_ece_rows']:.4f} | "
                    f"family T={entry['t_family']:.3f} lofo ECE {entry['lofo_ece_family']:.4f}"
                    f"  -> {entry['chosen']}",
                    flush=True,
                )
            else:
                print(
                    f"  {bucket:<16} families={entry['families']:<3} "
                    f"rows T={entry['t_rows']:.3f} (too few families)",
                    flush=True,
                )
    else:
        profile = fit({k: [(lg, y) for lg, y, _ in v] for k, v in rows.items()}, split_name=split)
    profile.save(out)
    if on_complete:
        on_complete()

    return {
        "profile": str(out),
        "temperatures": profile.temperatures,
        "n_samples": profile.n_samples,
        "report": report,
    }


def collect_logits(
    checkpoint_dir: str,
    data_dir: str,
    split: str,
    config=None,
    batch_size: int = 16,
    limit: int | None = None,
) -> dict[str, list[tuple[list[float], int, str]]]:
    """Collect raw logits and task families by calibration bucket, skipping abstain rows."""

    from ..data.build import load_split
    from ..data.sources import family_of
    from ..data.splits import Split
    from .config import PRESETS
    from .loop import load_for_eval, scored_rows

    config = config or PRESETS["4b"]
    model, tokenizer, head = load_for_eval(config, checkpoint_dir)

    rows = [e for e in load_split(data_dir, Split(split)) if not e.abstain]
    if limit:
        rows = rows[:limit]

    buckets: dict[str, list[tuple[list[float], int, str]]] = {}
    for example, logits, mode, width in scored_rows(
        model, tokenizer, head, config, rows, batch_size
    ):
        sample = (logits, example.target, family_of(example.source))
        # Banded for serving, unbanded as the fallback for thin bands.
        for key in {
            CalibrationProfile.key(example.question.type, mode, width),
            CalibrationProfile.key(example.question.type, mode),
        }:
            buckets.setdefault(key, []).append(sample)
    return buckets
