"""A small hand-labelled support-triage benchmark.

Label provenance: these labels were authored by hand for this repository. They
are not a vendor benchmark and not an independent public dataset. They exist so
calibration metrics have *some* ground truth -- without labels you can only
measure latency and cost. Items were chosen to be unambiguous; the `frustration`
levels are the softest of the three and should be read as the weakest signal.

Swap this module out for a real labelled set (the TypeSafe cookbooks list
several) before drawing conclusions about accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    """The three questions asked of every item, one per primitive type."""
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


def dataset() -> tuple[list[Item], dict[str, Any]]:
    return ITEMS, questions()
