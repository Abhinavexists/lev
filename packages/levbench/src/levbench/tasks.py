"""Built-in smoke fixture and file-backed benchmark tasks.

The 24 support tickets below are too few to measure training improvements.
Use held-out task files for benchmarks. Each JSON file has `questions` and
`items`; each item has a `state` and `labels` keyed by question name.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from typesafe_sdk import Choice, Noul, Score

DEPARTMENTS = {
    "billing": "Payment, invoice, refund or subscription issues",
    "technical": "Bugs, errors, outages or integration problems",
    "sales": "Pricing, plans, upgrades or pre-purchase questions",
    "account": "Login, password, permissions or profile changes",
}

FRUSTRATION_LEVELS = [
    "Calm and neutral; simply stating facts or asking a question",
    "Mildly annoyed or impatient, but still polite",
    "Openly angry; complaining, threatening to leave, or using strong language",
]


def questions() -> dict[str, Any]:
    return {
        "department": Choice(
            instructions="Which team should handle this ticket",
            criteria=DEPARTMENTS,
        ),
        "frustration": Score(
            instructions="How frustrated the customer appears",
            criteria=FRUSTRATION_LEVELS,
        ),
        "is_urgent": Noul(
            instructions="The customer is blocked right now or states an explicit deadline",
            criteria={
                "true": "Work is stopped, money is being lost, or a deadline is named",
                "false": "A question or request that can wait; no deadline stated",
            },
        ),
    }


@dataclass(frozen=True)
class Item:
    state: str
    department: str
    frustration: int
    is_urgent: bool

    @property
    def labels(self) -> dict[str, Any]:
        return {
            "department": self.department,
            "frustration": self.frustration,
            "is_urgent": self.is_urgent,
        }


ITEMS: list[Item] = [
    Item(
        "My card was charged twice for the October invoice. Please refund the duplicate.",
        "billing",
        1,
        False,
    ),
    Item(
        "The API returns a 500 on every POST to /v1/orders. Our checkout is completely down "
        "and we are losing sales right now.",
        "technical",
        2,
        True,
    ),
    Item(
        "Hi, could you tell me what the difference is between the Team and Business plans?",
        "sales",
        0,
        False,
    ),
    Item(
        "I can't log in. Password reset emails never arrive. I've checked spam.",
        "account",
        1,
        False,
    ),
    Item(
        "This is the third time I've written about the same billing error and nobody has "
        "replied. Absolutely unacceptable. Cancel my account.",
        "billing",
        2,
        False,
    ),
    Item("Quick question: does the Enterprise plan include SSO?", "sales", 0, False),
    Item(
        "Webhooks stopped firing at 03:00 UTC. Nothing changed on our side. "
        "We need this fixed before our 09:00 launch.",
        "technical",
        1,
        True,
    ),
    Item(
        "Please remove the old admin user from our workspace, they left the company.",
        "account",
        0,
        False,
    ),
    Item(
        "I was told I'd get a refund two weeks ago and I still see nothing on my statement.",
        "billing",
        2,
        False,
    ),
    Item("How much does it cost to add 10 more seats?", "sales", 0, False),
    Item(
        "Getting 'invalid signature' from the webhook verifier since the SDK update. "
        "Here's the stack trace.",
        "technical",
        0,
        False,
    ),
    Item(
        "Our whole team is locked out after the SSO migration. Nobody can work. This is urgent.",
        "account",
        2,
        True,
    ),
    Item("Can you send me a copy of last year's invoices for our accountant?", "billing", 0, False),
    Item(
        "The docs for the pagination parameter are wrong -- `cursor` is documented but the "
        "API expects `after`. Cost me an afternoon.",
        "technical",
        1,
        False,
    ),
    Item(
        "We're evaluating you against a competitor and need pricing for 500 seats by Friday.",
        "sales",
        0,
        True,
    ),
    Item("I need to change the email address on my account.", "account", 0, False),
    Item(
        "Why am I being billed for the Pro plan when I downgraded to Starter last month?",
        "billing",
        1,
        False,
    ),
    Item(
        "Rate limiting is returning 429 even though we're well under the documented quota. "
        "Production traffic is being dropped.",
        "technical",
        1,
        True,
    ),
    Item("Do you offer a discount for non-profits?", "sales", 0, False),
    Item(
        "Two-factor authentication is rejecting my codes after I changed phones.",
        "account",
        1,
        False,
    ),
    Item(
        "Your service has been down three times this month. We are losing customers and "
        "I want to speak to someone today.",
        "technical",
        2,
        True,
    ),
    Item("Just confirming: does my subscription auto-renew on the 1st?", "billing", 0, False),
    Item("I'd like to upgrade to the annual plan -- what's the process?", "sales", 0, False),
    Item(
        "Someone added a user to my workspace that I don't recognise. Please investigate "
        "immediately, this may be a breach.",
        "account",
        2,
        True,
    ),
]


@dataclass(frozen=True)
class FileItem:
    state: str
    _labels: dict[str, Any] = field(default_factory=dict)

    @property
    def labels(self) -> dict[str, Any]:
        return self._labels


_PRIMITIVES = {"choice": Choice, "score": Score, "noul": Noul}


def load_task_file(path: str | Path) -> tuple[list[FileItem], dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"task file {str(path)!r} does not exist. Write one with:\n"
            f"    uv run lev data eval --out {path}"
        )
    payload = json.loads(path.read_text())
    built: dict[str, Any] = {}
    for name, spec in payload["questions"].items():
        spec = dict(spec)
        kind = spec.pop("type")
        if kind not in _PRIMITIVES:
            raise ValueError(f"question {name!r} has unknown type {kind!r}")
        built[name] = _PRIMITIVES[kind](**spec)

    items = [FileItem(row["state"], row["labels"]) for row in payload["items"]]
    if not items:
        raise ValueError(f"{path} contains no items")

    unknown = {k for item in items for k in item.labels} - set(built)
    if unknown:
        raise ValueError(f"{path} labels questions that it does not define: {sorted(unknown)}")
    return items, built


def dataset(path: str | Path | None = None) -> tuple[list[Any], dict[str, Any]]:
    if path is not None:
        return load_task_file(path)
    return ITEMS, questions()


def detectable_difference(n: int, baseline: float = 0.8, z: float = 1.96) -> float:
    """Roughly the smallest accuracy change `n` items can distinguish.

    The half-width of the normal-approximation interval for one proportion;
    comparing two runs needs a wider margin still. The built-in 24-item fixture
    lands around +/-16 points.
    """
    if n <= 0:
        return 1.0
    return min(1.0, z * math.sqrt(baseline * (1 - baseline) / n))
