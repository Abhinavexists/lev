"""Assemble the training mixture, gated by the contamination guard.

An example is one (state, question, answer) triple. The mixture controls three
things the architecture depends on, so they live here rather than in the loop:

  layout          50/50 state-first / schema-first, so both caches work at inference
  abstain         a fraction whose answer is genuinely not determinable from the
                  state, carrying a uniform target rather than a gold label
  option scaling  some questions with large option sets, to exercise Mode B

The loaders live in `lev.data.sources`; `build_mixture` takes them injected so the
tests never touch the network.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field

from ..prompt import Layout
from ..router import candidate_count
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
    # Present only on abstain examples, where there is no gold index to supervise
    # and the correct behaviour is spread mass, not pick one. The loss reads this
    # in preference to `target` when it is set.
    soft_target: list[float] | None = None


@dataclass
class MixtureSpec:
    """Which sources to draw from, and in what proportion."""

    sources: dict[str, float] = field(default_factory=dict)
    n_examples: int = 200_000
    schema_first_fraction: float = 0.5
    abstain_fraction: float = 0.1
    seed: int = 17
    # source -> sources whose states must not be used as abstain donors,
    # because they would in fact answer the question. See `sources.ADJACENT`.
    adjacent: dict[str, frozenset[str]] = field(default_factory=dict)

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
        layout = (
            Layout.SCHEMA_FIRST if rng.random() < spec.schema_first_fraction else Layout.STATE_FIRST
        )
        if rng.random() < spec.abstain_fraction:
            yield _abstain_from(example, pools, names, source, layout, rng, spec.adjacent)
        else:
            yield Example(
                state=example.state,
                name=example.name,
                question=example.question,
                target=example.target,
                layout=layout,
                source=source,
            )


def _abstain_from(example, pools, names, source, layout, rng, adjacent=None) -> Example:
    """Build an unanswerable example by pairing a question with a foreign state.

    Flagging an otherwise normal example `abstain=True` -- which is what an
    earlier version of this function did -- is worse than not augmenting at all:
    the state still determines the answer, so the only thing the model learns is
    to be unsure when it should not be. The question has to become genuinely
    unanswerable, and the way to do that is to take the state away.

    The substituted state is drawn from a *different and non-adjacent* source
    so it cannot accidentally answer the question, and the target is uniform
    because with no evidence every candidate is equally supported. That uniform
    vector is the calibrated answer, and teaching it is the point.

    "Different" alone is not enough. imdb and rotten_tomatoes are both movie
    reviews, so pairing one's question with the other's state produces a fully
    answerable example labelled uniform -- the precise mislabelling this is
    meant to avoid, on roughly 1% of rows. `spec.adjacent` carries the
    exclusions; an empty map means callers who have not declared any.
    """
    excluded = {source} | set((adjacent or {}).get(source, ()))
    others = [n for n in names if n not in excluded] or [n for n in names if n != source] or names
    donor_source = rng.choice(others)
    donor = rng.choice(pools[donor_source])
    n_candidates = candidate_count(example.question)
    return Example(
        state=donor.state,
        name=example.name,
        question=example.question,
        target=example.target,  # retained for bookkeeping; the loss uses soft_target
        layout=layout,
        source=source,
        abstain=True,
        soft_target=[1.0 / n_candidates] * n_candidates,
    )
