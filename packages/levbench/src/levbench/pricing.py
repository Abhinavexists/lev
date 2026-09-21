"""Per-million-token prices, and cost accounting for a single call.

Jev prices are from typesafe.ai (input $0.042/M, output free). Anthropic prices
are first-party API rates. Both are vendor list prices and go stale -- override
via `PRICES` rather than editing call sites.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float


PRICES: dict[str, Price] = {
    # TypeSafe. Output is genuinely $0 -- "too cheap to meter".
    "jev-latest": Price(0.042, 0.0),
    "jev-1.13": Price(0.042, 0.0),
    # Anthropic first-party rates.
    "claude-opus-5": Price(5.00, 25.00),
    "claude-sonnet-5": Price(2.00, 10.00),
    "claude-haiku-4-5": Price(1.00, 5.00),
    "claude-fable-5-1": Price(10.00, 50.00),
}


# A model you host yourself: the cost is GPU time, not tokens, so quoting any
# per-token figure would be actively misleading. Callers label it instead.
SELF_HOSTED = Price(0.0, 0.0)


def is_self_hosted(model: str) -> bool:
    """True for anything served locally rather than metered by a vendor."""
    return model not in PRICES and not model.startswith("jev")


def lookup(model: str) -> Price:
    """Resolve a price. Unknown models are treated as self-hosted, not an error."""
    if model in PRICES:
        return PRICES[model]
    if model.startswith("jev"):
        return PRICES["jev-latest"]
    # Self-hosted checkpoints (`Qwen/...`, a local adapter, ...) are not metered.
    return SELF_HOSTED


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price = lookup(model)
    return (input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok) / 1_000_000
