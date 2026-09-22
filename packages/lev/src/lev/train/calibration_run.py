"""Fit per-bucket temperatures after training, on a dedicated split.

Separate from the fine-tune because re-fitting must not require re-training, and
because the split used here must be disjoint from *both* train and test.
`lev.calibrate.fit` refuses a split named test/eval/holdout for that reason.

One temperature per (question type, readout mode). Not one global scalar: a
Noul's nine-rating distribution and a 151-option Mode B distribution are not
miscalibrated in the same direction or by the same amount, and a single scalar
fitted across both lands between them and improves neither.
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
    config=None,
) -> dict:
    """Collect raw logits on `split`, fit one temperature per (type, mode)."""
    buckets = collect_logits(checkpoint_dir, data_dir, split, config=config)
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


def collect_logits(
    checkpoint_dir: str,
    data_dir: str,
    split: str,
    config=None,
    batch_size: int = 16,
    limit: int | None = None,
) -> dict[str, list[tuple[list[float], int]]]:
    """Raw (pre-softmax) candidate logits per bucket, keyed `"{type}:{mode}"`.

    Runs the *trained* engine over the calibration split under `no_grad`. The
    logits collected are pre-temperature by construction -- fitting a
    temperature on already-tempered scores would measure the previous fit.

    Abstain rows are skipped: they have no single correct answer, and a
    temperature is fitted against a gold index.
    """
    import torch

    from ..data.build import load_split
    from ..data.splits import Split
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

    routes = RouteCache(tokenizer, config.max_label_options)
    collator = DecisionCollator(tokenizer, max_seq_len=config.max_seq_len, routes=routes)
    batcher = ModeBatcher(tokenizer, batch_size=batch_size, routes=routes)
    device = device_of(model)

    buckets: dict[str, list[tuple[list[float], int]]] = {}
    with torch.no_grad():
        for group in batcher(rows):
            batch = to_device(collator(group), device)
            logits = candidate_logits(model, batch, head)
            for i, example in enumerate(group):
                width = int((~batch.candidate_mask[i]).sum())
                key = f"{example.question.type}:{batch.mode.value}"
                buckets.setdefault(key, []).append((logits[i, :width].tolist(), example.target))
    return buckets
