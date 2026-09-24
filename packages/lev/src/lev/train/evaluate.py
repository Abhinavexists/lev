"""Evaluate checkpoints through the training forward path, without fitting.

Report metrics per source before and after applying fitted temperatures.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..calibrate import CalibrationProfile, expected_calibration_error, softmax
from ..metrics import brier, log_loss


def accuracy_interval(accuracy: float, n: int, z: float = 1.96) -> float:
    """Normal-approximation confidence interval half-width; 95% at the default z."""
    if n <= 0:
        return 1.0
    return min(1.0, z * math.sqrt(max(accuracy * (1 - accuracy), 1e-9) / n))


@dataclass
class SourceScore:
    source: str
    question_type: str
    mode: str
    n: int
    accuracy: float
    ece: float
    mean_brier: float
    mean_log_loss: float
    temperature: float


@dataclass
class EvalReport:
    split: str
    checkpoint: str
    calibrated: bool
    per_source: list[SourceScore] = field(default_factory=list)

    def summary(self) -> str:
        head = (
            f"{self.checkpoint}  split={self.split}  "
            f"{'calibrated' if self.calibrated else 'UNCALIBRATED'}"
        )
        rows = [
            f"  {s.source:<18} {s.question_type:<6} {s.mode}  n={s.n:<5} "
            f"acc {s.accuracy:.3f} +/-{accuracy_interval(s.accuracy, s.n):.3f}  "
            f"ECE {s.ece:.4f}  brier {s.mean_brier:.4f}  "
            f"logloss {s.mean_log_loss:.4f}  T={s.temperature:.3f}"
            for s in self.per_source
        ]
        total = sum(s.n for s in self.per_source)
        if total:
            weighted_accuracy = sum(s.accuracy * s.n for s in self.per_source) / total
            weighted_ece = sum(s.ece * s.n for s in self.per_source) / total
            rows.append(
                f"  {'WEIGHTED':<18} {'':<6} {' '}  n={total:<5} "
                f"acc {weighted_accuracy:.3f} "
                f"+/-{accuracy_interval(weighted_accuracy, total):.3f}  "
                f"ECE {weighted_ece:.4f}"
            )
        rows.append(
            "  Intervals are 95% on accuracy alone. The run is not bit-reproducible; "
            "treat ECE moves below ~0.01 as noise."
        )
        return "\n".join([head, *rows])


def score(
    raw_logits: Sequence[tuple[list[float], int]], temperature: float
) -> tuple[float, float, float, float]:
    """Accuracy, ECE, mean Brier and mean log loss at one temperature."""
    return score_mixed([(logits, truth, temperature) for logits, truth in raw_logits])


def score_mixed(
    rows: Sequence[tuple[list[float], int, float]],
) -> tuple[float, float, float, float]:
    """`score` for rows that each carry their own temperature."""
    probs = [softmax(logits, t) for logits, _, t in rows]
    truths = [truth for _, truth, _ in rows]
    correct = sum(
        max(range(len(p)), key=p.__getitem__) == t for p, t in zip(probs, truths, strict=True)
    )
    return (
        correct / len(probs),
        expected_calibration_error(probs, truths),
        sum(brier(p, t) for p, t in zip(probs, truths, strict=True)) / len(probs),
        sum(log_loss(p, t) for p, t in zip(probs, truths, strict=True)) / len(probs),
    )


def evaluate_split(
    checkpoint_dir: str,
    data_dir: str,
    split: str = "test",
    config=None,
    profile: CalibrationProfile | None = None,
    limit_per_source: int | None = None,
    batch_size: int = 16,
) -> tuple[EvalReport, EvalReport]:
    """Return (uncalibrated, calibrated) reports over `split`.

    Both come from one forward pass, with the temperature applied to stored raw
    logits, so any difference is the temperature alone.
    """
    import torch

    from ..data.build import load_split
    from ..data.splits import Split
    from ..prompt import Style
    from .checkpoints import load_checkpoint, resolve_checkpoint
    from .collate import DecisionCollator, ModeBatcher, RouteCache
    from .config import PRESETS
    from .loop import build_head, build_model, candidate_logits, device_of, to_device

    config = config or PRESETS["4b"]
    profile = profile or CalibrationProfile()

    model, tokenizer = build_model(config, None)
    head = build_head(config, model) if config.train_mode_b_head else None
    load_checkpoint(model, head, checkpoint_dir)
    model.eval()
    if head is not None:
        head.eval()

    rows = [row for row in load_split(data_dir, Split(split)) if not row.abstain]
    if limit_per_source:
        seen: dict[str, int] = {}
        kept = []
        for row in rows:
            if seen.get(row.source, 0) < limit_per_source:
                seen[row.source] = seen.get(row.source, 0) + 1
                kept.append(row)
        rows = kept

    routes = RouteCache(tokenizer, config.max_label_options, Style(config.prompt_style))
    collator = DecisionCollator(tokenizer, max_seq_len=config.max_seq_len, routes=routes)
    # Bucketed like training: each row is scored once whatever the order.
    batcher = ModeBatcher(
        tokenizer, batch_size=batch_size, bucket_window=config.bucket_window, routes=routes
    )
    device = device_of(model)

    # (source, question type, mode, option count) -> (raw logits, gold index) rows.
    collected: dict[tuple[str, str, str, int], list[tuple[list[float], int]]] = {}
    with torch.no_grad():
        for group in batcher(rows):
            batch = to_device(collator(group), device)
            logits = candidate_logits(model, batch, head)
            for i, example in enumerate(group):
                width = int((~batch.candidate_mask[i]).sum())
                key = (example.source, example.question.type, batch.mode.value, width)
                collected.setdefault(key, []).append((logits[i, :width].tolist(), example.target))

    resolved = str(resolve_checkpoint(checkpoint_dir))
    plain = EvalReport(split=split, checkpoint=resolved, calibrated=False)
    tuned = EvalReport(split=split, checkpoint=resolved, calibrated=bool(profile.temperatures))
    # A source's rows may differ in option count, which sets the temperature, so
    # each group is scored at its own temperature and then reported per source.
    per_source: dict[tuple[str, str, str], list[tuple[list[float], int, float]]] = {}
    for (source, qtype, mode, width), samples in collected.items():
        t = profile.temperature(qtype, mode, width)
        per_source.setdefault((source, qtype, mode), []).extend((lg, y, t) for lg, y in samples)
    for (source, qtype, mode), rows_t in sorted(per_source.items()):
        samples = [(lg, y) for lg, y, _ in rows_t]
        temperatures = {t for _, _, t in rows_t}
        mixed_temperatures = len(temperatures) > 1
        shown_t = next(iter(temperatures)) if not mixed_temperatures else float("nan")
        for report, calibrated in ((plain, False), (tuned, True)):
            if calibrated and mixed_temperatures:
                accuracy, ece, mean_brier, mean_ll = score_mixed(rows_t)
            else:
                accuracy, ece, mean_brier, mean_ll = score(samples, shown_t if calibrated else 1.0)
            report.per_source.append(
                SourceScore(
                    source,
                    qtype,
                    mode,
                    len(samples),
                    accuracy,
                    ece,
                    mean_brier,
                    mean_ll,
                    shown_t if calibrated else 1.0,
                )
            )
    return plain, tuned
