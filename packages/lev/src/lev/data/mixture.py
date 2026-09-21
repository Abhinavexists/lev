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
from collections.abc import Callable, Iterable, Iterator
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
        # *better* and would silently invalidate the only number this project
        # competes on.
        assert_clean(self.sources.keys())
        if not self.sources:
            raise ValueError("mixture has no sources")
        if abs(sum(self.sources.values()) - 1.0) > 1e-6:
            raise ValueError(f"source weights must sum to 1, got {sum(self.sources.values())}")


Loader = Callable[[], Iterable[Example]]


def build_mixture(spec: MixtureSpec, loaders: dict[str, Loader]) -> Iterator[Example]:
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
    weights = [spec.sources[name] for name in names]

    for _ in range(spec.n_examples):
        source = rng.choices(names, weights=weights, k=1)[0]
        example = rng.choice(pools[source])
        layout = (
            Layout.SCHEMA_FIRST if rng.random() < spec.schema_first_fraction else Layout.STATE_FIRST
        )

        abstain = rng.random() < spec.abstain_fraction
        state = example.state
        soft_target = None
        if abstain:
            # Replace the state, so the question becomes genuinely unanswerable,
            # and supervise a uniform distribution: with no evidence every
            # candidate is equally supported, and that is the calibrated answer
            # rather than a hedge. See ADR-012.
            state = _donor_state(pools, source, spec.adjacent, rng)
            n_candidates = candidate_count(example.question)
            soft_target = [1.0 / n_candidates] * n_candidates

        yield Example(
            state=state,
            name=example.name,
            question=example.question,
            # Kept for bookkeeping on abstain rows; the loss reads soft_target.
            target=example.target,
            layout=layout,
            source=source,
            abstain=abstain,
            soft_target=soft_target,
        )


def _donor_state(
    pools: dict[str, list[Example]],
    source: str,
    adjacent: dict[str, frozenset[str]],
    rng: random.Random,
):
    """A state borrowed from a source that cannot answer `source`'s question.

    "A different source" is not sufficient. imdb and rotten_tomatoes are both
    movie reviews, so pairing one's question with the other's state leaves the
    question perfectly answerable while it gets labelled uniform -- the exact
    mislabelling abstain augmentation exists to avoid. `adjacent` carries those
    exclusions; see `sources.ADJACENT`.

    Falls back to any other source, then to any source at all, so a narrow
    mixture still yields an example rather than raising.
    """
    excluded = {source} | set(adjacent.get(source, ()))
    eligible = (
        [name for name in pools if name not in excluded]
        or [name for name in pools if name != source]
        or list(pools)
    )
    return rng.choice(pools[rng.choice(eligible)]).state
