"""Assemble the training mixture, gated by the contamination guard.

An example is one (state, question, answer) triple. The mixture controls three
things the architecture depends on, so they live here rather than in the loop:

  layout          50/50 state-first / schema-first, so both caches work at inference
  abstain         a fraction whose answer is not determinable from the state
  option scaling  some questions with large option sets, to exercise Mode B

NOT YET RUN against real datasets. `build_mixture` is the contract; the loaders it
calls are the part to fill in during the hacking phase.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field

from ..prompt import Layout
from ..types import Question
from .contamination import assert_clean


@dataclass
class Example:
    state: str | dict | list
    name: str
    question: Question
    target: int  # index into the candidate set
    layout: Layout
    source: str
    abstain: bool = False


@dataclass
class MixtureSpec:
    """Which sources to draw from, and in what proportion."""

    sources: dict[str, float] = field(default_factory=dict)
    n_examples: int = 200_000
    schema_first_fraction: float = 0.5
    abstain_fraction: float = 0.1
    seed: int = 17

    def validate(self) -> None:
        # Raises on any source colliding with an S1Bench evaluation subset. This
        # runs before a single example is loaded, because a contaminated run looks
        # *better* and would silently invalidate the only number we compete on.
        assert_clean(self.sources)
        if not self.sources:
            raise ValueError("mixture has no sources")
        if abs(sum(self.sources.values()) - 1.0) > 1e-6:
            raise ValueError(f"source weights must sum to 1, got {sum(self.sources.values())}")


def build_mixture(spec: MixtureSpec, loaders: dict[str, callable]) -> Iterator[Example]:
    """Yield `spec.n_examples`, respecting weights, layout split and abstain rate.

    `loaders` maps a source name to a callable returning an iterable of Examples.
    Injected rather than imported so tests can run without touching the network.
    """
    spec.validate()
    missing = set(spec.sources) - set(loaders)
    if missing:
        raise KeyError(f"no loader for sources: {sorted(missing)}")

    rng = random.Random(spec.seed)
    pools = {name: list(loaders[name]()) for name in spec.sources}
    for name, pool in pools.items():
        if not pool:
            raise ValueError(f"source {name!r} yielded no examples")

    names = list(spec.sources)
    weights = [spec.sources[n] for n in names]

    for _ in range(spec.n_examples):
        source = rng.choices(names, weights=weights, k=1)[0]
        example = rng.choice(pools[source])
        yield Example(
            state=example.state,
            name=example.name,
            question=example.question,
            target=example.target,
            layout=(
                Layout.SCHEMA_FIRST
                if rng.random() < spec.schema_first_fraction
                else Layout.STATE_FIRST
            ),
            source=source,
            abstain=rng.random() < spec.abstain_fraction,
        )
