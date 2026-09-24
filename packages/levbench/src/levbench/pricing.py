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
    # TypeSafe bills input only.
    "jev-latest": Price(0.042, 0.0),
    "jev-1.13": Price(0.042, 0.0),
    # Anthropic first-party rates.
    "claude-opus-5": Price(5.00, 25.00),
    "claude-sonnet-5": Price(2.00, 10.00),
    "claude-haiku-4-5": Price(1.00, 5.00),
    "claude-fable-5-1": Price(10.00, 50.00),
}


# A self-hosted model costs GPU time, not tokens
SELF_HOSTED = Price(0.0, 0.0)


def is_self_hosted(model: str) -> bool:
    return model not in PRICES and not model.startswith("jev")


def lookup(model: str) -> Price:
    if model in PRICES:
        return PRICES[model]
    if model.startswith("jev"):
        return PRICES["jev-latest"]
    return SELF_HOSTED


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price = lookup(model)
    return (input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok) / 1_000_000
