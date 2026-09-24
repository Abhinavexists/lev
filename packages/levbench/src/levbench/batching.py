"""The batching sweep: one state, N questions, one call versus N calls.

Measures the central economic claim: input tokens are billed per *request*,
so reading a large state is amortised across every question asked in the same
call. If it holds, batched cost stays roughly flat as N grows while split cost
grows linearly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from typesafe_sdk import Choice, Noul, Score

from . import pricing
from .runner import call_once

# One shared document; all three primitives, since output-token cost differs
# between them.
QUESTION_BANK: list[tuple[str, Any]] = [
    ("mentions_retention", Noul(instructions="The document discusses data retention periods.")),
    ("mentions_consent", Noul(instructions="The document discusses user consent.")),
    ("mentions_breach", Noul(instructions="The document describes breach notification duties.")),
    (
        "mentions_transfer",
        Noul(instructions="The document discusses international data transfers."),
    ),
    ("mentions_minors", Noul(instructions="The document addresses the personal data of children.")),
    ("mentions_dpo", Noul(instructions="The document refers to a data protection officer.")),
    (
        "audience",
        Choice(
            instructions="Who is the primary audience for this document",
            criteria={
                "engineers": "People implementing the system",
                "lawyers": "People assessing legal compliance",
                "end_users": "Members of the public using the service",
            },
        ),
    ),
    (
        "doc_type",
        Choice(
            instructions="What kind of document is this",
            criteria={
                "policy": "An internal or external policy statement",
                "contract": "A binding agreement between parties",
                "guide": "Explanatory or how-to material",
            },
        ),
    ),
    (
        "severity",
        Score(
            instructions="How strict the obligations described are",
            criteria=["Advisory only", "Recommended practice", "Mandatory with penalties"],
        ),
    ),
    (
        "clarity",
        Score(
            instructions="How clearly the document is written",
            criteria=["Dense and hard to follow", "Readable with effort", "Plain and clear"],
        ),
    ),
    ("mentions_encryption", Noul(instructions="The document requires encryption of stored data.")),
    ("mentions_audit", Noul(instructions="The document requires audit logging.")),
    ("mentions_erasure", Noul(instructions="The document grants a right to erasure.")),
]


@dataclass
class SweepRow:
    n_questions: int
    batched_cost: float
    batched_seconds: float
    split_cost: float
    split_seconds: float

    @property
    def cost_ratio(self) -> float:
        return self.split_cost / self.batched_cost if self.batched_cost else 0.0

    @property
    def speed_ratio(self) -> float:
        return self.split_seconds / self.batched_seconds if self.batched_seconds else 0.0


def sweep(
    client,
    model: str,
    state: str,
    counts: list[int],
    bank: list[tuple[str, Any]] | None = None,
    errors: list[tuple[int, str]] | None = None,
) -> list[SweepRow]:
    """Run the sweep, appending any per-count failure to `errors`."""
    bank = bank or QUESTION_BANK
    rows: list[SweepRow] = []
    errors = errors if errors is not None else []

    for n in counts:
        if n > len(bank):
            raise ValueError(f"Asked for {n} questions but the bank holds {len(bank)}")
        selected = dict(bank[:n])

        # The default sweep (1,2,4,8,13) sends 33 sequential requests; each count
        # is isolated so a late rate limit keeps the rows already paid for.
        try:
            batched = call_once(client, state, selected)
            batched_cost = pricing.cost_usd(model, batched.input_tokens, batched.output_tokens)

            split_cost = 0.0
            split_seconds = 0.0
            for key, question in selected.items():
                one = call_once(client, state, {key: question})
                split_cost += pricing.cost_usd(model, one.input_tokens, one.output_tokens)
                split_seconds += one.seconds
        except Exception as exc:  # noqa: BLE001 -- any API failure, report and move on
            errors.append((n, f"{type(exc).__name__}: {exc}"))
            continue

        rows.append(
            SweepRow(
                n_questions=n,
                batched_cost=batched_cost,
                batched_seconds=batched.seconds,
                split_cost=split_cost,
                split_seconds=split_seconds,
            )
        )
    return rows


def format_sweep(
    rows: list[SweepRow],
    model: str,
    state_chars: int,
    errors: list[tuple[int, str]] | None = None,
) -> str:
    out = [
        f"=== batching sweep / {model} / state {state_chars:,} chars ===",
        "",
        f"{'N':>3}  {'batched $':>11}  {'split $':>11}  {'cheaper':>8}  "
        f"{'batched s':>10}  {'split s':>9}  {'faster':>7}",
    ]
    for r in rows:
        out.append(
            f"{r.n_questions:>3}  {r.batched_cost:>11.6f}  {r.split_cost:>11.6f}  "
            f"{r.cost_ratio:>7.1f}x  {r.batched_seconds:>10.3f}  "
            f"{r.split_seconds:>9.3f}  {r.speed_ratio:>6.1f}x"
        )
    out.append("")
    for n, message in errors or []:
        out.append(f"  N={n} FAILED: {message}")
    if errors:
        out.append("")
    out.append(
        "If 'cheaper' grows roughly linearly with N, input tokens are billed "
        "per request and the shared-state economics hold."
    )
    return "\n".join(out)
