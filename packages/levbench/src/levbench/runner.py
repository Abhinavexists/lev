"""Backend-agnostic evaluation runner.

Both backends expose the same `system_one(state, questions)` call and return the
same answer types, so the benchmark code below is written once and the client is
swapped. That interchangeability is the whole point of the vendor's adapter.
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from . import metrics, pricing
from .tasks import Item


@dataclass
class CallResult:
    answers: dict[str, Any]
    seconds: float
    # The model the server says answered. Compared against the requested model
    # so a silently-substituted default cannot masquerade as the one asked for.
    served_by: str
    input_tokens: int
    output_tokens: int
    # Adapter-only: retries the LLM needed to emit a schema-valid answer.
    # Always 0 for Jev, where schema conformance is structural.
    schema_retries: int = 0
    transient_retries: int = 0


@dataclass
class EvalReport:
    backend: str
    model: str
    calls: list[CallResult] = field(default_factory=list)
    per_question: dict[str, metrics.Calibration] = field(default_factory=dict)
    records: dict[str, list] = field(default_factory=dict)

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
        """Distinct models the server reported. More than one means the run
        is not a single-model measurement and must not be reported as one."""
        return {c.served_by for c in self.calls}


def build_client(backend: str, model: str | None = None, base_url: str | None = None):
    """Return `(client, model_name)` for `jev` or `anthropic`.

    `base_url` points the jev backend at any server speaking the same
    `/v1/systemone` wire schema -- the open reproductions (LitJev, OpenJev)
    implement it deliberately, so the same benchmark runs against them with no
    other change. Ignored for the anthropic backend, which is not an endpoint.
    """
    if backend == "jev":
        from typesafe_sdk import TypeSafeClient

        resolved = model or "jev-latest"
        # Must be passed at construction -- `system_one` defaults `model=None`,
        # which makes the server pick, so omitting it here would silently
        # benchmark whatever the default is while labelling it `resolved`.
        kwargs: dict[str, Any] = {"model": resolved}
        if base_url:
            kwargs["base_url"] = base_url
            # Never forward the real Jev credential to a non-default host. The
            # SDK always sends `Authorization: Bearer <key>`, so reusing
            # TYPESAFE_API_KEY here would hand it to whatever server the user
            # pointed at. Open reproductions need no credential at all; anyone
            # fronting one with auth sets LEVBENCH_LOCAL_API_KEY explicitly.
            kwargs["api_key"] = os.environ.get("LEVBENCH_LOCAL_API_KEY", "local")
        return TypeSafeClient(**kwargs), resolved

    if backend == "anthropic":
        from system_one_adapter import SystemOneAdapterClient

        resolved = model or "claude-opus-5"
        client = SystemOneAdapterClient(
            structured_outputs=True,
            # Ask for full distributions, not just an argmax -- calibration
            # metrics are meaningless without them.
            llm_answer_mode="probabilities",
            normalize_probabilities=True,
            n_retry_malformed_structure=2,
            provider="anthropic",
            model=resolved,
        )
        return client, resolved

    raise ValueError(f"Unknown backend {backend!r} (expected 'jev' or 'anthropic')")


def _token_count(usage: Any, total_field: str, base_field: str) -> int:
    """Read a token count, refusing to guess when the API reported nothing.

    The adapter's Usage adds `*_total` fields covering retried attempts, which
    is what actually gets billed; the real SDK's Usage has only the base
    fields, and types them `Optional[int]`. Both are checked with `is not
    None` rather than truthiness, because a genuine 0 is a real measurement
    and must not be confused with an absent one -- silently substituting 0
    would book the call as free and drag the headline cost down.
    """
    for field_name in (total_field, base_field):
        value = getattr(usage, field_name, None)
        if value is not None:
            return int(value)
    raise ValueError(
        f"Usage reported neither {total_field} nor {base_field}; refusing to "
        f"assume 0 tokens, which would understate cost. Got: {usage!r}"
    )


def _usage_ints(usage: Any) -> tuple[int, int, int, int]:
    """Pull tokens and retry counts, tolerating both Usage shapes."""
    inp = _token_count(usage, "input_tokens_total", "input_tokens")
    out = _token_count(usage, "output_tokens_total", "output_tokens")
    # Retry counters are adapter-only and genuinely absent on the real SDK,
    # where 0 is the correct reading: Jev needs no schema retries.
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
    items: list[Item],
    questions: dict[str, Any],
) -> EvalReport:
    report = EvalReport(backend=backend, model=model)
    records: dict[str, list] = {name: [] for name in questions}

    for item in items:
        result = call_once(client, item.state, questions)
        report.calls.append(result)
        for name, answer in result.answers.items():
            truth = item.labels[name]
            records[name].append(
                (
                    metrics.to_distribution(answer),
                    metrics.predicted_label(answer),
                    truth,
                    metrics.confidence(answer),
                )
            )

    report.records = records
    report.per_question = {n: metrics.calibration(r) for n, r in records.items()}
    return report


def format_report(report: EvalReport) -> str:
    lines: list[str] = []
    n = len(report.calls)
    lines.append(f"=== {report.backend} / {report.model} ===")
    lines.append(f"calls              {n}")
    if n:
        lines.append(f"latency p50        {report.pct(0.5):.3f}s")
        lines.append(f"latency mean       {statistics.mean(report.latencies):.3f}s")
        # A "p95" over a couple of dozen calls is just the second-slowest one,
        # and on a cold connection that is mostly setup jitter. Report the tail
        # honestly rather than dressing an order statistic up as a percentile.
        if n >= 100:
            lines.append(f"latency p95        {report.pct(0.95):.3f}s")
        else:
            lines.append(
                f"latency slowest    {report.latencies[-1]:.3f}s  (n={n}; too few for a p95)"
            )
    lines.append(f"input tokens       {sum(c.input_tokens for c in report.calls):,}")
    lines.append(f"output tokens      {sum(c.output_tokens for c in report.calls):,}")
    lines.append(f"total cost         ${report.total_cost:.6f}")
    lines.append(f"schema retries     {report.total_schema_retries}")
    lines.append(f"transient retries  {report.total_transient_retries}")
    served = report.served_by
    if served and served != {report.model}:
        lines.append(
            f"WARNING            requested {report.model!r} but server served "
            f"{sorted(served)} -- cost figures use the requested model's price"
        )
    lines.append("")

    for name, cal in report.per_question.items():
        lines.append(f"-- {name}")
        lines.append(
            f"   accuracy {cal.accuracy:.3f}   log-loss {cal.mean_log_loss:.4f}   "
            f"brier {cal.mean_brier:.4f}   ECE {cal.ece:.4f}"
        )
        recs = report.records[name]
        for thr in (0.5, 0.7, 0.9):
            acc, kept = metrics.selective_accuracy(recs, thr)
            lines.append(f"   conf>={thr:.1f}: accuracy {acc:.3f} on {kept:.0%} of items")
        occupied = [b for b in cal.bins if b.n]
        if occupied:
            lines.append("   reliability (confidence bin -> accuracy, n):")
            for b in occupied:
                lines.append(
                    f"     [{b.lo:.1f},{b.hi:.1f})  conf {b.mean_confidence:.3f}  "
                    f"acc {b.accuracy:.3f}  n={b.n}  gap {b.gap:+.3f}"
                )
        lines.append("")
    return "\n".join(lines)
