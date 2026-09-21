"""Fit per-bucket temperatures after training, on a dedicated split.

Separate from the fine-tune because re-fitting must not require re-training, and
because the split used here must be disjoint from *both* train and test.
`lev.calibrate.fit` refuses a split named test/eval/holdout for that reason.

NOT YET RUN.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..calibrate import fit


def fit_profile(
    checkpoint_dir: str,
    data_dir: str,
    split: str = "calibration",
    on_complete: Callable[[], None] | None = None,
) -> dict:
    """Collect raw logits on `split`, fit one temperature per (type, mode)."""
    buckets = collect_logits(checkpoint_dir, data_dir, split)
    profile = fit(buckets, split_name=split)

    out = Path(checkpoint_dir) / "calibration.json"
    profile.save(out)
    if on_complete:
        on_complete()

    return {
        "profile": str(out),
        "temperatures": profile.temperatures,
        "n_samples": profile.n_samples,
    }


def collect_logits(checkpoint_dir: str, data_dir: str, split: str) -> dict:
    """Raw (pre-softmax) candidate logits per bucket, keyed `"{type}:{mode}"`."""
    raise NotImplementedError(
        "Run the trained engine over the calibration split and collect raw scores. "
        "Depends on the same loaders as training; see docs/TRAINING.md."
    )
