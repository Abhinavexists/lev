"""Sample typed training examples with layout, abstention and question variation.

Loaders are injected so the mixture can be built and tested offline.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

from ..labels import NOUL_RATING_TOKENS
from ..prompt import Layout
from ..router import candidate_count
from ..types import Choice, Noul, Question, Score
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
    # Abstain examples only: no gold index, so the loss reads this uniform target
    # instead of `target`.
    soft_target: list[float] | None = None


@dataclass(frozen=True)
class Augment:
    """How one source's questions may be varied in the training mixture.

    `negations` ask the opposite question, flipping the Noul target, so "yes"
    does not mean "good" in every row. `min_options` is the floor when a Choice
    is subsampled.
    """

    paraphrases: tuple[str, ...] = ()
    negations: tuple[str, ...] = ()
    min_options: int = 2
    # Fraction of Choice rows that keep their full option set (shuffle only).
    # For a large taxonomy this is the candidate-path head's training data.
    keep_full_fraction: float = 0.0


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
    # Training variation per source; empty means canonical questions (ADR-020).
    augment: dict[str, Augment] = field(default_factory=dict)
    paraphrase_fraction: float = 0.7
    negate_fraction: float = 0.5
    subsample_fraction: float = 0.5
    shuffle_fraction: float = 0.5
    description_dropout: float = 0.3

    def validate(self) -> None:
        # Before any example loads: a contaminated run silently looks better.
        assert_clean(self.sources.keys())
        if not self.sources:
            raise ValueError("mixture has no sources")
        if abs(sum(self.sources.values()) - 1.0) > 1e-6:
            raise ValueError(f"source weights must sum to 1, got {sum(self.sources.values())}")


Loader = Callable[[], Iterable[Example]]


def build_mixture(spec: MixtureSpec, loaders: dict[str, Loader]) -> Iterator[Example]:
    """Yield `spec.n_examples`, respecting weights, layout split and abstain rate.

    `loaders` maps a source name to a callable returning its Examples.
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
        question, target = example.question, example.target
        if (augment := spec.augment.get(source)) is not None:
            # Negation flips the target, so it must not touch an abstain row,
            # whose supervision is uniform regardless of polarity.
            question, target = _vary(
                question, target, augment, spec, rng, allow_negation=not abstain
            )

        soft_target = None
        if abstain:
            # Swap in an unrelated state and supervise a uniform distribution:
            # with no evidence that is the calibrated answer (ADR-012).
            state = _donor_state(pools, source, spec.adjacent, rng)
            n_candidates = candidate_count(question)
            soft_target = [1.0 / n_candidates] * n_candidates

        yield Example(
            state=state,
            name=example.name,
            question=question,
            # Kept for bookkeeping on abstain rows; the loss reads soft_target.
            target=target,
            layout=layout,
            source=source,
            abstain=abstain,
            soft_target=soft_target,
        )


def _vary(
    question: Question,
    target: int,
    augment: Augment,
    spec: MixtureSpec,
    rng: random.Random,
    allow_negation: bool,
) -> tuple[Question, int]:
    """Vary wording and Choice options while preserving the gold answer.

    Noul negation mirrors the rating target. Choice subsampling keeps the gold
    option and remaps its index. Score levels are never reordered.
    """
    instructions = question.instructions
    negate = isinstance(question, Noul) and augment.negations and allow_negation
    if negate and rng.random() < spec.negate_fraction:
        instructions = rng.choice(augment.negations)
        target = (len(NOUL_RATING_TOKENS) - 1) - target
        return Noul(instructions=instructions, criteria=question.criteria), target
    if augment.paraphrases and rng.random() < spec.paraphrase_fraction:
        instructions = rng.choice(augment.paraphrases)

    if isinstance(question, Noul):
        return Noul(instructions=instructions, criteria=question.criteria), target
    if isinstance(question, Score):
        return Score(instructions=instructions, criteria=question.criteria), target

    options = list(question.criteria.items())
    gold = options[target]
    keep_full = rng.random() < augment.keep_full_fraction
    if (
        not keep_full
        and len(options) > augment.min_options
        and rng.random() < spec.subsample_fraction
    ):
        k = rng.randint(augment.min_options, len(options) - 1)
        others = [o for i, o in enumerate(options) if i != target]
        options = [gold] + rng.sample(others, k - 1)
        rng.shuffle(options)
    elif rng.random() < spec.shuffle_fraction:
        rng.shuffle(options)
    if rng.random() < spec.description_dropout:
        options = [(key, None) for key, _ in options]
    return Choice(instructions=instructions, criteria=dict(options)), _index_of(options, gold[0])


def _index_of(options: list[tuple[str, object]], key: str) -> int:
    for i, (k, _) in enumerate(options):
        if k == key:
            return i
    raise AssertionError(f"gold option {key!r} lost during augmentation")


def _donor_state(
    pools: dict[str, list[Example]],
    source: str,
    adjacent: dict[str, frozenset[str]],
    rng: random.Random,
):
    """A state borrowed from a source that cannot answer `source`'s question.

    Another source is not enough: imdb's question is answerable from a
    rotten_tomatoes state, so `adjacent` (`sources.ADJACENT`) excludes such
    pairs. Falls back to any other source, then any source, so a narrow mixture
    still yields an example.
    """
    excluded = {source} | set(adjacent.get(source, ()))
    eligible = (
        [name for name in pools if name not in excluded]
        or [name for name in pools if name != source]
        or list(pools)
    )
    return rng.choice(pools[rng.choice(eligible)]).state
