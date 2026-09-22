"""Score a trained checkpoint on a held-out split.

Runs the same forward path as serving -- route, collate, read the candidate
logits -- and reports accuracy and calibration per source, with and without the
fitted temperature. Reporting both is the point: accuracy is unaffected by
temperature, so a single set of numbers cannot show whether calibration helped.

Evaluating on `test` is correct; *fitting* on it is not, which is why
`calibrate.fit` refuses that split and this module never calls it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..calibrate import CalibrationProfile, expected_calibration_error, softmax
from ..metrics import brier, log_loss


def accuracy_interval(accuracy: float, n: int, z: float = 1.96) -> float:
    """95% half-width on an accuracy estimate from `n` items.

    Printed beside every accuracy because the evaluation is not bit-reproducible:
    bf16 reductions and the Triton linear-attention kernels do not pin their
    reduction order, so logits differ in the last bits between runs and items
    near a decision boundary flip. Two runs of the same checkpoint on the same
    data have differed by one item in 200. Without the interval a reader takes
    that 0.5-point move for a result.
    """
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
            macro_acc = sum(s.accuracy * s.n for s in self.per_source) / total
            macro_ece = sum(s.ece * s.n for s in self.per_source) / total
            rows.append(
                f"  {'WEIGHTED':<18} {'':<6} {' '}  n={total:<5} "
                f"acc {macro_acc:.3f} +/-{accuracy_interval(macro_acc, total):.3f}  "
                f"ECE {macro_ece:.4f}"
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
    probs, truths = [], []
    for logits, truth in raw_logits:
        probs.append(softmax(logits, temperature))
        truths.append(truth)
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

    Both come from one forward pass: the temperature is applied to stored raw
    logits afterwards, so the two reports describe the same predictions and any
    difference between them is the temperature alone.
    """
    import torch

    from ..data.build import load_split
    from ..data.splits import Split
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

    routes = RouteCache(tokenizer, config.max_label_options)
    collator = DecisionCollator(tokenizer, max_seq_len=config.max_seq_len, routes=routes)
    # Bucketed like training: every row is scored exactly once whatever the
    # batch order, so the 4.4x padding saving is free here.
    batcher = ModeBatcher(
        tokenizer, batch_size=batch_size, bucket_window=config.bucket_window, routes=routes
    )
    device = device_of(model)

    # (source, question type, mode) -> raw logits and gold index per row.
    collected: dict[tuple[str, str, str], list[tuple[list[float], int]]] = {}
    with torch.no_grad():
        for group in batcher(rows):
            batch = to_device(collator(group), device)
            logits = candidate_logits(model, batch, head)
            for i, example in enumerate(group):
                width = int((~batch.candidate_mask[i]).sum())
                key = (example.source, example.question.type, batch.mode.value)
                collected.setdefault(key, []).append((logits[i, :width].tolist(), example.target))

    resolved = str(resolve_checkpoint(checkpoint_dir))
    plain = EvalReport(split=split, checkpoint=resolved, calibrated=False)
    tuned = EvalReport(split=split, checkpoint=resolved, calibrated=True)
    for (source, qtype, mode), samples in sorted(collected.items()):
        temperature = profile.temperature(qtype, mode)
        for report, t in ((plain, 1.0), (tuned, temperature)):
            accuracy, ece, mean_brier, mean_ll = score(samples, t)
            report.per_source.append(
                SourceScore(
                    source, qtype, mode, len(samples), accuracy, ece, mean_brier, mean_ll, t
                )
            )
    return plain, tuned
