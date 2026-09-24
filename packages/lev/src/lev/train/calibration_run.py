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
    """Collect raw logits on `split`, fit one temperature per bucket.

    `method="transfer"` fits each bucket both by rows and by task family and
    keeps whichever has the lower leave-one-family-out ECE (ADR-028), writing
    the comparison to `calibration.report.json`. `method="rows"` is the
    original fit. An existing profile is kept as `calibration.previous.json`.
    """
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
    """Raw candidate logits per `CalibrationProfile.key` bucket, from the trained
    model under `no_grad`, before any temperature.

    Abstain rows are skipped: a temperature is fitted against a gold index. Each
    sample carries its task family (`sources.family_of`) for the transfer fit.
    """
    import torch

    from ..data.build import load_split
    from ..data.sources import family_of
    from ..data.splits import Split
    from ..prompt import Style
    from ..train.checkpoints import load_checkpoint
    from ..train.collate import DecisionCollator, ModeBatcher, RouteCache
    from ..train.config import PRESETS
    from ..train.loop import build_head, build_model, candidate_logits, device_of, to_device

    config = config or PRESETS["4b"]
    model, tokenizer = build_model(config, None)
    head = build_head(config, model) if config.train_mode_b_head else None
    load_checkpoint(model, head, checkpoint_dir)
    model.eval()
    if head is not None:
        head.eval()

    rows = [e for e in load_split(data_dir, Split(split)) if not e.abstain]
    if limit:
        rows = rows[:limit]

    routes = RouteCache(tokenizer, config.max_label_options, Style(config.prompt_style))
    collator = DecisionCollator(tokenizer, max_seq_len=config.max_seq_len, routes=routes)
    batcher = ModeBatcher(tokenizer, batch_size=batch_size, routes=routes)
    device = device_of(model)

    buckets: dict[str, list[tuple[list[float], int, str]]] = {}
    with torch.no_grad():
        for group in batcher(rows):
            batch = to_device(collator(group), device)
            logits = candidate_logits(model, batch, head)
            for i, example in enumerate(group):
                width = int((~batch.candidate_mask[i]).sum())
                sample = (logits[i, :width].tolist(), example.target, family_of(example.source))
                # Banded for serving, unbanded as the fallback for thin bands.
                for key in {
                    CalibrationProfile.key(example.question.type, batch.mode.value, width),
                    CalibrationProfile.key(example.question.type, batch.mode.value),
                }:
                    buckets.setdefault(key, []).append(sample)
    return buckets
