"""Compute call costs from per-million-token list prices; override rates via PRICES."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float


# Every Jev version bills at the published Jev rate unless PRICES names it.
JEV = Price(0.042, 0.0)

PRICES: dict[str, Price] = {
    "claude-opus-5": Price(5.00, 25.00),
    "claude-sonnet-5": Price(2.00, 10.00),
    "claude-haiku-4-5": Price(1.00, 5.00),
    "claude-fable-5-1": Price(10.00, 50.00),
}


# Unknown non-Jev models are self-hosted and have zero token cost.
SELF_HOSTED = Price(0.0, 0.0)


def is_self_hosted(model: str) -> bool:
    return model not in PRICES and not model.startswith("jev")


def lookup(model: str) -> Price:
    if model in PRICES:
        return PRICES[model]
    if model.startswith("jev"):
        return JEV
    return SELF_HOSTED


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price = lookup(model)
    return (input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok) / 1_000_000
