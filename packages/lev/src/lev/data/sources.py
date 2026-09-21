"""The training sources: public classification datasets, cast as typed questions.

Every source here is a plain labelled-classification dataset. That is deliberate.
A Jev-like model is not learning world knowledge, it is learning to put *calibrated
mass on a candidate set* -- so the supervision we need is a gold label plus an
explicit option set, which is exactly what a classification corpus is.

Three things the registry has to get right:

  primitive coverage  Choice, Score and Noul all need supervision. A mixture of
                      only Choice teaches nothing about the 9-rating-token Noul
                      readout or the ordinal structure of Score.
  Mode B coverage     banking77 (77 intents) and clinc_oos (150) are the *only*
                      sources whose option sets overflow single-token label codes.
                      Without them Mode B -- the differentiator -- never trains.
                      See `MODE_B_SOURCES`; the default mixture weights them up.
  cleanliness         every id is checked against the ADR-009 block list at import
                      of the registry, not at load time.

Label names come from the dataset's own `ClassLabel` feature wherever possible
rather than being retyped here, because a silently reordered label list turns into
a wrong gold answer that no test would catch.

Every id here is parquet-backed and loads under `datasets>=5`, which no longer
executes dataset scripts. That ruled out three otherwise-good candidates:
`CogComp/trec` and `takala/financial_phrasebank` have no parquet mirror carrying
their label names, and are omitted rather than pinned to a reordered third-party
copy; `PolyAI/banking77` is script-backed, so we use `legacy-datasets/banking77`,
which is the same corpus with its `ClassLabel` intact. `test_sources.py` asserts
the whole registry stays loadable so this cannot rot silently.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from ..prompt import Layout
from ..types import Choice, Noul, Question, Score
from .contamination import assert_clean
from .mixture import Example

# Options beyond this many cannot get a single-token label code in any tokenizer
# we target (A..Z is 26). Sources above it are the Mode B training signal.
SINGLE_TOKEN_CODE_CEILING = 26


@dataclass(frozen=True)
class SourceSpec:
    """One HuggingFace dataset, and how to read a typed question out of it."""

    name: str
    hf_id: str
    text_field: str
    label_field: str
    primitive: str  # "choice" | "score" | "noul"
    instructions: str
    hf_config: str | None = None
    split: str = "train"
    # Only for sources whose ClassLabel names are absent or unusable.
    label_names: tuple[str, ...] | None = None
    descriptions: dict[str, str] = field(default_factory=dict)

    @property
    def is_mode_b(self) -> bool:
        n = len(self.label_names) if self.label_names else 0
        return n > SINGLE_TOKEN_CODE_CEILING or self.name in _KNOWN_LARGE


# Sources whose option count we know exceeds the ceiling but whose names are read
# from the dataset at load time, so `label_names` is None here.
_KNOWN_LARGE = frozenset({"banking77", "clinc_oos"})


REGISTRY: dict[str, SourceSpec] = {
    # ---- Choice: small, well-separated option sets -------------------------
    "ag_news": SourceSpec(
        name="ag_news",
        hf_id="fancyzhx/ag_news",
        text_field="text",
        label_field="label",
        primitive="choice",
        instructions="Which section of the paper does this story belong in?",
    ),
    "emotion": SourceSpec(
        name="emotion",
        hf_id="dair-ai/emotion",
        text_field="text",
        label_field="label",
        primitive="choice",
        instructions="Which emotion is the writer expressing?",
    ),
    "dbpedia_14": SourceSpec(
        name="dbpedia_14",
        hf_id="fancyzhx/dbpedia_14",
        text_field="content",
        label_field="label",
        primitive="choice",
        instructions="What kind of thing is this article about?",
    ),
    # ---- Mode B: option sets too large for single-token label codes --------
    "banking77": SourceSpec(
        name="banking77",
        hf_id="legacy-datasets/banking77",
        text_field="text",
        label_field="label",
        primitive="choice",
        instructions="What is this customer trying to do?",
    ),
    "clinc_oos": SourceSpec(
        name="clinc_oos",
        hf_id="clinc/clinc_oos",
        hf_config="plus",
        text_field="text",
        label_field="intent",
        primitive="choice",
        instructions="What is the user asking the assistant to do?",
    ),
    # ---- Score: genuinely ordered levels ------------------------------------
    "yelp_review_full": SourceSpec(
        name="yelp_review_full",
        hf_id="Yelp/yelp_review_full",
        text_field="text",
        label_field="label",
        primitive="score",
        instructions="How positive is this review?",
        label_names=("1 star", "2 stars", "3 stars", "4 stars", "5 stars"),
    ),
    "sst5": SourceSpec(
        name="sst5",
        hf_id="SetFit/sst5",
        text_field="text",
        label_field="label",
        primitive="score",
        instructions="How positive is this sentence?",
        label_names=(
            "very negative",
            "negative",
            "neutral",
            "positive",
            "very positive",
        ),
    ),
    # ---- Noul: a single yes/no proposition ----------------------------------
    "imdb": SourceSpec(
        name="imdb",
        hf_id="stanfordnlp/imdb",
        text_field="text",
        label_field="label",
        primitive="noul",
        instructions="Did this reviewer like the film?",
    ),
    "rotten_tomatoes": SourceSpec(
        name="rotten_tomatoes",
        hf_id="cornell-movie-review-data/rotten_tomatoes",
        text_field="text",
        label_field="label",
        primitive="noul",
        instructions="Is this review positive?",
    ),
}

# Checked once, at import. A contaminated id must never reach a loader.
assert_clean([spec.hf_id for spec in REGISTRY.values()])
assert_clean(REGISTRY)

MODE_B_SOURCES: frozenset[str] = frozenset(
    name for name, spec in REGISTRY.items() if spec.is_mode_b
)

# Sources close enough that one's state can answer another's question.
#
# This matters only for abstain augmentation (ADR-012), which builds an
# unanswerable example by pairing a question with a *foreign* state. Pair
# imdb's "Did this reviewer like the film?" with a rotten_tomatoes state and
# the question is fully answerable -- and it gets supervised as uniform, which
# is exactly the mislabelling the augmentation exists to avoid. All four
# sentiment corpora are mutually adjacent (SST-5 and Rotten Tomatoes are both
# movie-review sentiment, Yelp is the same task on another domain), and
# clinc_oos contains banking intents.
ADJACENT: tuple[frozenset[str], ...] = (
    frozenset({"imdb", "rotten_tomatoes", "sst5", "yelp_review_full", "emotion"}),
    frozenset({"banking77", "clinc_oos"}),
    frozenset({"ag_news", "dbpedia_14"}),
)


def adjacency_map() -> dict[str, frozenset[str]]:
    """source -> the sources whose states must not be used to make it unanswerable."""
    out: dict[str, frozenset[str]] = {}
    for name in REGISTRY:
        related = (
            frozenset().union(*(g for g in ADJACENT if name in g))
            if any(name in g for g in ADJACENT)
            else frozenset()
        )
        out[name] = related - {name}
    return out


def _noul_rating(label: int) -> int:
    """Map a binary class onto the ends of the 0-8 rating scale.

    A Noul is not read out as two options, it is read out as nine rating tokens
    and collapsed by `noul_probability`. So the supervision index is a *rating*,
    and passing the raw class through would train "yes" as rating 1 -- which
    `noul_probability` reads back as P(yes) = 0.125. The model would be learning
    to answer no on every positive example while the loss looked healthy.
    """
    from ..labels import NOUL_RATING_TOKENS

    return 0 if int(label) == 0 else len(NOUL_RATING_TOKENS) - 1


def humanise(label: str) -> str:
    """`card_arrival` -> `card arrival`; `Sci/Tech` is left alone.

    banking77 and clinc_oos ship snake_case intent ids. Feeding those to the
    model as option text trains it on a token distribution no real request uses.
    """
    return re.sub(r"[_\-]+", " ", label).strip()


def _label_names(spec: SourceSpec, dataset) -> list[str]:
    if spec.label_names:
        return list(spec.label_names)
    feature = dataset.features[spec.label_field]
    names = getattr(feature, "names", None)
    if not names:
        raise TypeError(
            f"source {spec.name!r} field {spec.label_field!r} is a "
            f"{type(feature).__name__} with no `names`; set `label_names` on the "
            f"SourceSpec so the option order is pinned rather than guessed"
        )
    return list(names)


def build_question(spec: SourceSpec, names: list[str]) -> Question:
    """Turn a label list into the typed question the model will be asked."""
    if spec.primitive == "noul":
        if len(names) != 2:
            raise ValueError(f"{spec.name}: a Noul needs exactly 2 labels, got {len(names)}")
        return Noul(instructions=spec.instructions)
    options = [humanise(n) for n in names]
    if spec.primitive == "score":
        return Score(instructions=spec.instructions, criteria=options)
    return Choice(
        instructions=spec.instructions,
        criteria={opt: spec.descriptions.get(raw) for opt, raw in zip(options, names, strict=True)},
    )


def load_source(
    spec: SourceSpec,
    limit: int | None = None,
    *,
    cache_dir: str | None = None,
    seed: int = 17,
    load_dataset: Callable | None = None,
) -> Iterator[Example]:
    """Yield Examples from one source, sampling rather than truncating.

    `limit` **shuffles first**. A head slice looks equivalent and is not: most of
    these corpora ship grouped by label, so `imdb[:400]` is 400 negative reviews
    and `dbpedia_14[:400]` is one class out of fourteen. Training on that teaches
    a prior on the label and it is invisible downstream, because every split
    drawn from it is biased the same way and the splits still agree with each
    other. The shuffle is seeded, so the sample stays reproducible.

    `load_dataset` is injectable purely so the tests can run without a network.
    """
    if load_dataset is None:
        from datasets import load_dataset  # noqa: PLC0415

    dataset = load_dataset(spec.hf_id, spec.hf_config, split=spec.split, cache_dir=cache_dir)
    names = _label_names(spec, dataset)
    if limit is not None and limit < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(limit))
    question = build_question(spec, names)
    n_labels = len(names)
    to_target = _noul_rating if spec.primitive == "noul" else int

    for i, row in enumerate(dataset):
        text = row[spec.text_field]
        target = int(row[spec.label_field])
        if not text or not text.strip():
            continue
        if not 0 <= target < n_labels:
            raise ValueError(
                f"{spec.name} row {i}: label {target} out of range for "
                f"{n_labels} classes -- the label column is not what we think"
            )
        yield Example(
            state=text.strip(),
            name=spec.name,
            question=question,
            target=to_target(target),
            layout=Layout.STATE_FIRST,  # reassigned by build_mixture
            source=spec.name,
        )


def default_weights() -> dict[str, float]:
    """Mode B gets a quarter of the mixture, split between its two sources.

    Not proportional to corpus size: proportional to what we need learned. Mode B
    is two sources out of eleven, so a size-weighted mixture would give it a few
    percent and it would not train. The remaining 0.75 is split evenly across the
    three primitives so none of the three readouts is starved.
    """
    mode_b = sorted(MODE_B_SOURCES)
    rest = [n for n in REGISTRY if n not in MODE_B_SOURCES]
    by_primitive: dict[str, list[str]] = {}
    for name in rest:
        by_primitive.setdefault(REGISTRY[name].primitive, []).append(name)

    weights = {name: 0.25 / len(mode_b) for name in mode_b}
    share = 0.75 / len(by_primitive)
    for group in by_primitive.values():
        for name in group:
            weights[name] = share / len(group)

    total = sum(weights.values())
    return {name: w / total for name, w in sorted(weights.items())}
