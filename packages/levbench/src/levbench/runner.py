"""Evaluate backends through the shared system_one interface."""

from __future__ import annotations

import os
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

from . import metrics, pricing
from .tasks import FileItem, Item

load_dotenv()


@dataclass
class CallResult:
    answers: dict[str, Any]
    seconds: float
    served_by: str
    input_tokens: int
    output_tokens: int
    # Adapter retries; the TypeSafe SDK reports zero.
    schema_retries: int = 0
    transient_retries: int = 0


@dataclass
class EvalReport:
    backend: str
    model: str
    calls: list[CallResult] = field(default_factory=list)
    per_question: dict[str, metrics.Calibration] = field(default_factory=dict)
    records: dict[str, list] = field(default_factory=dict)
    wall_seconds: float = 0.0

    @property
    def total_cost(self) -> float:
        return sum(
            pricing.cost_usd(self.model, c.input_tokens, c.output_tokens) for c in self.calls
        )

    @property
    def latencies(self) -> list[float]:
        return sorted(c.seconds for c in self.calls)

    def pct(self, q: float) -> float:
        lat = self.latencies
        if not lat:
            return 0.0
        idx = min(int(q * len(lat)), len(lat) - 1)
        return lat[idx]

    @property
    def total_schema_retries(self) -> int:
        return sum(c.schema_retries for c in self.calls)

    @property
    def total_transient_retries(self) -> int:
        return sum(c.transient_retries for c in self.calls)

    @property
    def served_by(self) -> set[str]:
        return {c.served_by for c in self.calls}


DEFAULT_LOCAL_BASE_URL = "http://localhost:8000"


# Modal's 4B cold start measured 20-55 s, against the SDK's 10 s default (ADR-022).
DEFAULT_LOCAL_TIMEOUT = 120.0


def build_client(
    backend: str,
    model: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
):
    """Build a Jev, local lev, or Anthropic client and return its model name."""
    if backend in ("jev", "lev"):
        if backend == "lev":
            base_url = base_url or DEFAULT_LOCAL_BASE_URL
        from typesafe_sdk import TypeSafeClient

        resolved = model or ("jev-latest" if backend == "jev" else "local")
        kwargs: dict[str, Any] = {"model": resolved}
        if timeout is not None:
            kwargs["timeout"] = timeout
        elif backend == "lev":
            kwargs["timeout"] = DEFAULT_LOCAL_TIMEOUT
        if base_url:
            kwargs["base_url"] = base_url
            # Use local credentials for custom endpoints; empty keys produce invalid Bearer headers.
            kwargs["api_key"] = os.environ.get("LEVBENCH_LOCAL_API_KEY") or "local"
        return TypeSafeClient(**kwargs), resolved

    if backend == "anthropic":
        from system_one_adapter import SystemOneAdapterClient

        resolved = model or "claude-opus-5"
        client = SystemOneAdapterClient(
            structured_outputs=True,
            llm_answer_mode="probabilities",
            normalize_probabilities=True,
            n_retry_malformed_structure=2,
            provider="anthropic",
            model=resolved,
        )
        return client, resolved

    raise ValueError(f"Unknown backend {backend!r} (expected 'jev', 'lev' or 'anthropic')")


def _token_count(usage: Any, total_field: str, base_field: str) -> int:
    """Prefer billed retry totals; accept zero counts and reject missing counts."""
    for field_name in (total_field, base_field):
        value = getattr(usage, field_name, None)
        if value is not None:
            return int(value)
    raise ValueError(
        f"Usage reported neither {total_field} nor {base_field}; refusing to "
        f"assume 0 tokens, which would understate cost. Got: {usage!r}"
    )


def _usage_ints(usage: Any) -> tuple[int, int, int, int]:
    inp = _token_count(usage, "input_tokens_total", "input_tokens")
    out = _token_count(usage, "output_tokens_total", "output_tokens")
    # Adapter-only counters; absent on the SDK, where 0 is correct.
    schema = int(getattr(usage, "n_retries_malformed_structure", 0) or 0)
    transient = int(getattr(usage, "n_retries", 0) or 0)
    return inp, out, schema, transient


def call_once(client, state: Any, questions: dict[str, Any]) -> CallResult:
    started = time.perf_counter()
    response = client.system_one(state=state, questions=questions)
    elapsed = time.perf_counter() - started
    inp, out, schema, transient = _usage_ints(response.usage)
    return CallResult(
        answers=response.answers,
        seconds=elapsed,
        served_by=response.model,
        input_tokens=inp,
        output_tokens=out,
        schema_retries=schema,
        transient_retries=transient,
    )


def run_eval(
    client,
    backend: str,
    model: str,
    items: Sequence[Item | FileItem],
    questions: dict[str, Any],
    concurrency: int = 1,
) -> EvalReport:
    """Evaluate concurrently, returning results in item order and per-call latencies."""
    from concurrent.futures import ThreadPoolExecutor

    report = EvalReport(backend=backend, model=model)
    records: dict[str, list] = {name: [] for name in questions}
    started = time.perf_counter()

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(lambda it: call_once(client, it.state, questions), items))
    else:
        results = [call_once(client, item.state, questions) for item in items]
    report.wall_seconds = time.perf_counter() - started

    for item, result in zip(items, results, strict=True):
        report.calls.append(result)
        labels = item.labels
        for name, answer in result.answers.items():
            truth = labels[name]
            distribution = metrics.to_distribution(answer)
            records[name].append(
                (
                    distribution,
                    metrics.predicted_label(answer),
                    truth,
                    max(distribution.values()),
                )
            )

    served = report.served_by
    if backend == "lev" and len(served) == 1:
        # The label was a placeholder; the server knows what it loaded.
        report.model = served.pop()

    report.records = records
    report.per_question = {n: metrics.calibration(r) for n, r in records.items()}
    return report


def format_report(report: EvalReport) -> str:
    return "\n".join([*_format_summary(report), *_format_per_question(report)])


def _format_summary(report: EvalReport) -> list[str]:
    lines: list[str] = []
    n = len(report.calls)
    lines.append(f"=== {report.backend} / {report.model} ===")
    if report.wall_seconds:
        lines.append(
            f"{'wall time':<18} {report.wall_seconds:.1f}s  ({n / report.wall_seconds:.2f} items/s)"
        )
    lines.append(f"calls              {n}")
    if n:
        lines.append(f"latency p50        {report.pct(0.5):.3f}s")
        lines.append(f"latency mean       {statistics.mean(report.latencies):.3f}s")
        # Report the maximum for small samples, where p95 is dominated by connection setup.
        if n >= 100:
            lines.append(f"latency p95        {report.pct(0.95):.3f}s")
        else:
            lines.append(
                f"latency slowest    {report.latencies[-1]:.3f}s  (n={n}; too few for a p95)"
            )
    lines.append(f"input tokens       {sum(call.input_tokens for call in report.calls):,}")
    lines.append(f"output tokens      {sum(call.output_tokens for call in report.calls):,}")
    if pricing.is_self_hosted(report.model):
        lines.append("total cost         self-hosted (GPU time, not per-token)")
    else:
        lines.append(f"total cost         ${report.total_cost:.6f}")
    lines.append(f"schema retries     {report.total_schema_retries}")
    lines.append(f"transient retries  {report.total_transient_retries}")
    served = report.served_by
    if served and served != {report.model} and not pricing.is_self_hosted(report.model):
        lines.append(
            f"WARNING            requested {report.model!r} but server served "
            f"{sorted(served)} -- cost figures use the requested model's price"
        )
    lines.append("")
    return lines


def _format_per_question(report: EvalReport) -> list[str]:
    lines: list[str] = []
    for name, calibration in report.per_question.items():
        lines.append(f"-- {name}")
        lines.append(
            f"   accuracy {calibration.accuracy:.3f}   "
            f"log-loss {calibration.mean_log_loss:.4f}   "
            f"brier {calibration.mean_brier:.4f}   ECE {calibration.ece:.4f}"
        )
        records = report.records[name]
        for threshold in (0.5, 0.7, 0.9):
            accuracy, kept = metrics.selective_accuracy(records, threshold)
            lines.append(
                f"   conf>={threshold:.1f}: accuracy {accuracy:.3f} on {kept:.0%} of items"
            )
        occupied = [b for b in calibration.bins if b.n]
        if occupied:
            lines.append("   reliability (top-probability bin -> accuracy, n):")
            for b in occupied:
                lines.append(
                    f"     [{b.lo:.1f},{b.hi:.1f})  conf {b.mean_confidence:.3f}  "
                    f"acc {b.accuracy:.3f}  n={b.n}  gap {b.gap:+.3f}"
                )
        lines.append("")
    return lines
