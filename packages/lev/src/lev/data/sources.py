"""Training dataset registry and readers for Choice, Score and Noul examples.

Label order comes from ClassLabel metadata unless explicitly pinned. Every
source id is checked against the evaluation block list at import. Tests use
fake loaders; they do not verify that upstream datasets remain available.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Literal

from ..labels import LABEL_OPTION_CAP, NOUL_RATING_TOKENS
from ..prompt import Layout
from ..types import Choice, JSONContent, Noul, Question, Score
from .contamination import assert_clean
from .mixture import Augment, Example

# Large taxonomies whose label names are read at load time, so the registry
# cannot count them.
_LARGE_OPTION_SETS = frozenset({"banking77", "clinc_oos"})

Primitive = Literal["choice", "score", "noul"]

# Reserve a share for large taxonomies so the Mode B head gets enough data (ADR-013).
MODE_B_SHARE = 0.25


# What a per-row reader returns: the state to show, the option texts if the row
# carries its own (None means the source's fixed label set), and the gold index.
Row = tuple[str | dict, list[str] | None, int]
# Readers may yield several examples from one row, or None to skip it.
Reader = Callable[[dict, random.Random], Row | list[Row] | None]


@dataclass(frozen=True)
class SourceSpec:
    """A dataset and its question, label mapping and optional row reader.

    Training can sample paraphrases or negate Noul questions. A custom reader
    handles per-row choices, sentence pairs and nested annotations.
    """

    name: str
    hf_id: str
    primitive: Primitive
    instructions: str
    text_field: str = "text"
    label_field: str = "label"
    hf_config: str | None = None
    split: str = "train"
    revision: str | None = None
    # Only for sources whose ClassLabel names are absent or unusable.
    label_names: tuple[str, ...] | None = None
    descriptions: dict[str, str] = field(default_factory=dict)
    paraphrases: tuple[str, ...] = ()
    negations: tuple[str, ...] = ()
    reader: Reader | None = None

    @property
    def is_mode_b(self) -> bool:
        """True for a large taxonomy (more than 26 options)."""
        if self.name in _LARGE_OPTION_SETS:
            return True
        return bool(self.label_names) and len(self.label_names) > LABEL_OPTION_CAP


def _read_race(row: dict, rng: random.Random) -> Row | None:
    options = [str(o) for o in row["options"]]
    index = ord(str(row["answer"]).strip().upper()[:1] or "?") - ord("A")
    if len(options) < 2 or not 0 <= index < len(options):
        return None
    return {"passage": row["article"], "question": row["question"]}, options, index


def _read_keyed_choices(question_field: str, context_field: str | None = None) -> Reader:
    """`choices={"label": [...], "text": [...]}` plus `answerKey` -- the shape
    commonsense_qa, openbookqa and ARC share. ARC keys some rows 1-4 rather
    than A-D, so the key is matched against the labels, never against A-Z."""

    def reader(row: dict, rng: random.Random) -> Row | None:
        labels = [str(x) for x in row["choices"]["label"]]
        texts = [str(x) for x in row["choices"]["text"]]
        key = str(row["answerKey"]).strip()
        if key not in labels or len(texts) < 2:
            return None
        question = row[question_field]
        state = (
            question
            if context_field is None or not row.get(context_field)
            else {"question": question, "context": row[context_field]}
        )
        return state, texts, labels.index(key)

    return reader


def _read_sciq(row: dict, rng: random.Random) -> Row | None:
    gold = row["correct_answer"]
    options = [gold, row["distractor1"], row["distractor2"], row["distractor3"]]
    options = [o for o in options if o and str(o).strip()]
    if len(options) < 2 or len(set(options)) != len(options):
        return None
    # The gold answer ships first; shuffle so its position carries no signal.
    rng.shuffle(options)
    support = row.get("support")
    state = {"question": row["question"], "context": support} if support else row["question"]
    return state, options, options.index(gold)


NLI_LABELS = ("entailment", "neutral", "contradiction")
NLI_DESCRIPTIONS = {
    "entailment": "the hypothesis must be true given the premise",
    "neutral": "the premise neither confirms nor rules out the hypothesis",
    "contradiction": "the hypothesis cannot be true given the premise",
}


def _read_nli(row: dict, rng: random.Random) -> Row | None:
    label = int(row["label"])
    # SNLI marks rows without annotator agreement as -1.
    if not 0 <= label < len(NLI_LABELS):
        return None
    return {"premise": row["premise"], "hypothesis": row["hypothesis"]}, None, label


def swap_two_words(text: str, rng: random.Random) -> str | None:
    """Exchange two distinct content words. The result keeps every token of the
    original -- maximal lexical overlap -- and no longer means the same thing.
    None when the sentence has too few candidates to swap."""
    words = text.split()
    slots = [i for i, w in enumerate(words) if len(w) > 3 and w.isalpha()]
    if len(slots) < 2:
        return None
    for _ in range(8):
        i, j = rng.sample(slots, 2)
        if words[i].lower() != words[j].lower():
            swapped = list(words)
            swapped[i], swapped[j] = words[j], words[i]
            return " ".join(swapped)
    return None


def _read_pair(first: str, second: str, adversarial: float = 0.0) -> Reader:
    """Read labelled sentence pairs, optionally adding word-swapped negatives.

    `adversarial` is the fraction of positive pairs that also yield a negative,
    reducing the usefulness of lexical overlap as a shortcut.
    """

    def reader(row: dict, rng: random.Random) -> list[Row] | None:
        a, b = row[first], row[second]
        label = int(row["label"])
        out: list[Row] = [({"sentence1": a, "sentence2": b}, None, label)]
        adversary = label == 1 and adversarial and rng.random() < adversarial
        if adversary and (swapped := swap_two_words(b, rng)) is not None:
            out.append(({"sentence1": a, "sentence2": swapped}, None, 0))
        return out

    return reader


FEVER_LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
FEVER_DESCRIPTIONS = {
    "SUPPORTS": "the evidence supports the claim",
    "REFUTES": "the evidence contradicts the claim",
    "NOT ENOUGH INFO": "the evidence neither supports nor contradicts the claim",
}


def _read_nli_fever(row: dict, rng: random.Random) -> Row | None:
    """FEVER as claim/evidence/verdict. The dataset's `premise` is the claim
    and its `hypothesis` the evidence sentence; the string label is used, not
    the integer, whose order differs from FEVER's own."""
    label = row.get("fever_gold_label")
    if label not in FEVER_LABELS:
        return None
    return {"claim": row["premise"], "evidence": row["hypothesis"]}, None, FEVER_LABELS.index(label)


def _read_parade(row: dict, rng: random.Random) -> Row | None:
    return (
        {"sentence1": row["Definition1"], "sentence2": row["Definition2"]},
        None,
        int(row["Binary labels"]),
    )


def _read_strategyqa(row: dict, rng: random.Random) -> Row | None:
    return {"question": row["question"], "facts": row["facts"]}, None, int(bool(row["answer"]))


def yes_no_from_choices(reader: Reader) -> Reader:
    """Turn each multiple-choice row into two Noul rows: gold and wrong proposals."""

    def wrapped(row: dict, rng: random.Random) -> list[Row] | None:
        produced = reader(row, rng)
        if produced is None:
            return None
        out: list[Row] = []
        for state, options, target in produced if isinstance(produced, list) else [produced]:
            if not options or len(options) < 2:
                continue
            base = state if isinstance(state, dict) else {"question": state}
            wrong = rng.choice([o for i, o in enumerate(options) if i != target])
            out.append(({**base, "proposed_answer": options[target]}, None, 1))
            out.append(({**base, "proposed_answer": wrong}, None, 0))
        return out or None

    return wrapped


def _read_toxic_chat(row: dict, rng: random.Random) -> Row | None:
    return row["user_input"], None, int(row["toxicity"])


def _read_toxigen(row: dict, rng: random.Random) -> Row | None:
    score = row.get("toxicity_human")
    if score is None:
        return None
    # Annotators rate 1-5; the dataset's own threshold for "toxic" is >= 3.
    return row["text"], None, int(float(score) >= 3.0)


def _read_beavertails(row: dict, rng: random.Random) -> Row | None:
    # `is_safe` True -> yes; the registry's negation keeps polarity from being
    # constant.
    return {"prompt": row["prompt"], "response": row["response"]}, None, int(bool(row["is_safe"]))


HELPFULNESS_LEVELS = (
    "not helpful",
    "slightly helpful",
    "partially helpful",
    "mostly helpful",
    "fully helpful",
)


def _read_ultrafeedback(row: dict, rng: random.Random) -> list[Row] | None:
    """One instruction, up to four rated completions -- each becomes a Score row."""
    out: list[Row] = []
    for completion in row.get("completions") or []:
        rating = ((completion.get("annotations") or {}).get("helpfulness") or {}).get("Rating")
        if rating is None:
            continue
        try:
            level = int(rating) - 1
        except (TypeError, ValueError):
            continue
        response = completion.get("response")
        if not response or not 0 <= level < len(HELPFULNESS_LEVELS):
            continue
        out.append(({"instruction": row["instruction"], "response": response}, None, level))
    return out or None


SNIPS_INTENTS = (
    "AddToPlaylist",
    "BookRestaurant",
    "GetWeather",
    "PlayMusic",
    "RateBook",
    "SearchCreativeWork",
    "SearchScreeningEvent",
)


def _read_snips(row: dict, rng: random.Random) -> Row | None:
    category = row["category"]
    if category not in SNIPS_INTENTS:
        return None
    return row["text"], None, SNIPS_INTENTS.index(category)


YES_NO = ("no", "yes")


REGISTRY: dict[str, SourceSpec] = {
    # Choice: small, well-separated option sets.
    "ag_news": SourceSpec(
        name="ag_news",
        hf_id="fancyzhx/ag_news",
        primitive="choice",
        instructions="Which section of the paper does this story belong in?",
        paraphrases=(
            "What is the topic of this news story?",
            "Under which heading would an editor file this article?",
            "Classify this story by subject.",
        ),
    ),
    "emotion": SourceSpec(
        name="emotion",
        hf_id="dair-ai/emotion",
        primitive="choice",
        instructions="Which emotion is the writer expressing?",
        paraphrases=(
            "How is the author feeling?",
            "What emotion does this message convey?",
            "Label the emotional tone of this text.",
        ),
    ),
    "dbpedia_14": SourceSpec(
        name="dbpedia_14",
        hf_id="fancyzhx/dbpedia_14",
        text_field="content",
        primitive="choice",
        instructions="What kind of thing is this article about?",
        paraphrases=(
            "Which category does the subject of this article fall into?",
            "What type of entity is being described?",
        ),
    ),
    # Choice with the question varying per row: the state alone cannot identify
    # the answer set, so the model has to read the question.
    "race": SourceSpec(
        name="race",
        hf_id="ehovy/race",
        hf_config="all",
        primitive="choice",
        instructions="Which option correctly answers the question about the passage?",
        paraphrases=(
            "Based on the passage, which answer is correct?",
            "Read the passage and answer the question.",
        ),
        reader=_read_race,
    ),
    "commonsense_qa": SourceSpec(
        name="commonsense_qa",
        hf_id="tau/commonsense_qa",
        primitive="choice",
        instructions="Which option best answers the question?",
        paraphrases=("Pick the most sensible answer.", "What is the right answer here?"),
        reader=_read_keyed_choices("question"),
    ),
    "sciq": SourceSpec(
        name="sciq",
        hf_id="allenai/sciq",
        primitive="choice",
        instructions="Which option answers the science question?",
        paraphrases=("Using the context if given, which answer is correct?",),
        reader=_read_sciq,
    ),
    "openbookqa": SourceSpec(
        name="openbookqa",
        hf_id="allenai/openbookqa",
        hf_config="additional",
        primitive="choice",
        instructions="Given the fact, which option completes the question correctly?",
        paraphrases=("Which answer follows from the fact and the question?",),
        reader=_read_keyed_choices("question_stem", context_field="fact1"),
    ),
    "arc_easy": SourceSpec(
        name="arc_easy",
        hf_id="allenai/ai2_arc",
        hf_config="ARC-Easy",
        primitive="choice",
        instructions="Which option correctly answers the question?",
        paraphrases=("Choose the correct answer.", "What is the right answer to this question?"),
        reader=_read_keyed_choices("question"),
    ),
    # Natural language inference: the same three options every time, but the
    # answer depends on a relation between two fields, not on either alone.
    "snli": SourceSpec(
        name="snli",
        hf_id="stanfordnlp/snli",
        primitive="choice",
        instructions="What is the relation between the premise and the hypothesis?",
        paraphrases=(
            "Does the premise entail, contradict, or say nothing about the hypothesis?",
            "Given the premise, is the hypothesis true, false, or undetermined?",
        ),
        label_names=NLI_LABELS,
        descriptions=NLI_DESCRIPTIONS,
        reader=_read_nli,
    ),
    "anli": SourceSpec(
        name="anli",
        hf_id="facebook/anli",
        split="train_r3",
        primitive="choice",
        instructions="What is the relation between the premise and the hypothesis?",
        paraphrases=(
            "Does the premise entail, contradict, or say nothing about the hypothesis?",
            "Given the premise, is the hypothesis true, false, or undetermined?",
        ),
        label_names=NLI_LABELS,
        descriptions=NLI_DESCRIPTIONS,
        reader=_read_nli,
    ),
    "nli_fever": SourceSpec(
        name="nli_fever",
        hf_id="pietrolesci/nli_fever",
        primitive="choice",
        instructions="Does the evidence support or refute the claim?",
        paraphrases=(
            "Given the evidence, is the claim supported, refuted, or undetermined?",
            "What verdict does the evidence give on the claim?",
        ),
        label_names=FEVER_LABELS,
        descriptions=FEVER_DESCRIPTIONS,
        reader=_read_nli_fever,
    ),
    "snips": SourceSpec(
        name="snips",
        hf_id="benayas/snips",
        primitive="choice",
        instructions="What does the user want to do?",
        paraphrases=("What is the user's intent?", "Which action is being requested?"),
        label_names=SNIPS_INTENTS,
        reader=_read_snips,
    ),
    # Large taxonomies: their full sets train the Mode B head.
    "banking77": SourceSpec(
        name="banking77",
        hf_id="legacy-datasets/banking77",
        primitive="choice",
        instructions="What is this customer trying to do?",
        paraphrases=(
            "Which banking intent does this message express?",
            "What does the customer need help with?",
            "Classify the customer's request.",
        ),
    ),
    "clinc_oos": SourceSpec(
        name="clinc_oos",
        hf_id="clinc/clinc_oos",
        hf_config="plus",
        label_field="intent",
        primitive="choice",
        instructions="What is the user asking the assistant to do?",
        paraphrases=(
            "What is the user's intent?",
            "Which command does this utterance correspond to?",
        ),
    ),
    # Score: genuinely ordered levels.
    "yelp_review_full": SourceSpec(
        name="yelp_review_full",
        hf_id="Yelp/yelp_review_full",
        primitive="score",
        instructions="How positive is this review?",
        paraphrases=("How many stars did this reviewer give?", "Rate the reviewer's satisfaction."),
        label_names=("1 star", "2 stars", "3 stars", "4 stars", "5 stars"),
    ),
    "sst5": SourceSpec(
        name="sst5",
        hf_id="SetFit/sst5",
        primitive="score",
        instructions="How positive is this sentence?",
        paraphrases=(
            "What is the sentiment of this sentence?",
            "Rate the tone from very negative to very positive.",
        ),
        label_names=(
            "very negative",
            "negative",
            "neutral",
            "positive",
            "very positive",
        ),
    ),
    "ultrafeedback": SourceSpec(
        name="ultrafeedback",
        hf_id="openbmb/UltraFeedback",
        primitive="score",
        instructions="How helpful is the response to the instruction?",
        paraphrases=(
            "Rate how well the response serves the request.",
            "How useful is this answer?",
        ),
        label_names=HELPFULNESS_LEVELS,
        reader=_read_ultrafeedback,
    ),
    # Noul: a single yes/no proposition. Every source carries a negation, so the
    # mixture asks both polarities and "yes" is not always the good outcome.
    "imdb": SourceSpec(
        name="imdb",
        hf_id="stanfordnlp/imdb",
        primitive="noul",
        instructions="Did this reviewer like the film?",
        paraphrases=("Is this a positive review?", "Would this reviewer recommend the film?"),
        negations=("Did this reviewer dislike the film?", "Is this a negative review?"),
    ),
    "rotten_tomatoes": SourceSpec(
        name="rotten_tomatoes",
        hf_id="cornell-movie-review-data/rotten_tomatoes",
        primitive="noul",
        instructions="Is this review positive?",
        paraphrases=("Did the critic like it?", "Is the sentiment favourable?"),
        negations=("Is this review negative?", "Did the critic pan it?"),
    ),
    # Paraphrase, with word-swapped negatives so overlap is not the answer.
    "mrpc": SourceSpec(
        name="mrpc",
        hf_id="SetFit/mrpc",
        primitive="noul",
        instructions="Do these two sentences mean the same thing?",
        paraphrases=("Are these sentences paraphrases of each other?",),
        negations=("Do these sentences differ in meaning?",),
        label_names=YES_NO,
        reader=_read_pair("text1", "text2", adversarial=0.6),
    ),
    "qqp": SourceSpec(
        name="qqp",
        hf_id="SetFit/qqp",
        primitive="noul",
        instructions="Are these two questions asking the same thing?",
        paraphrases=("Are these questions duplicates?",),
        negations=("Are these two questions asking different things?",),
        label_names=YES_NO,
        reader=_read_pair("text1", "text2", adversarial=0.6),
    ),
    "parade": SourceSpec(
        name="parade",
        hf_id="tasksource/parade",
        primitive="noul",
        instructions="Do these two definitions describe the same thing?",
        paraphrases=("Are these two definitions equivalent?",),
        negations=("Do these two definitions describe different things?",),
        label_names=YES_NO,
        reader=_read_parade,
    ),
    # Yes/no on a factual axis: a proposed answer to a grounded question.
    "strategyqa": SourceSpec(
        name="strategyqa",
        hf_id="ChilleD/StrategyQA",
        primitive="noul",
        instructions="Given the facts, is the answer to the question yes?",
        paraphrases=("Using the facts, answer the question.",),
        negations=("Given the facts, is the answer to the question no?",),
        label_names=YES_NO,
        reader=_read_strategyqa,
    ),
    "race_yesno": SourceSpec(
        name="race_yesno",
        hf_id="ehovy/race",
        hf_config="all",
        primitive="noul",
        instructions="Is the proposed answer to the question correct, according to the passage?",
        paraphrases=("Does the passage support the proposed answer?",),
        negations=("Is the proposed answer wrong, according to the passage?",),
        label_names=YES_NO,
        reader=yes_no_from_choices(_read_race),
    ),
    "sciq_yesno": SourceSpec(
        name="sciq_yesno",
        hf_id="allenai/sciq",
        primitive="noul",
        instructions="Is the proposed answer to the science question correct?",
        paraphrases=("Is this the right answer?",),
        negations=("Is the proposed answer incorrect?",),
        label_names=YES_NO,
        reader=yes_no_from_choices(_read_sciq),
    ),
    "openbookqa_yesno": SourceSpec(
        name="openbookqa_yesno",
        hf_id="allenai/openbookqa",
        hf_config="additional",
        primitive="noul",
        instructions="Given the fact, is the proposed answer correct?",
        paraphrases=("Does the fact make the proposed answer right?",),
        negations=("Given the fact, is the proposed answer wrong?",),
        label_names=YES_NO,
        reader=yes_no_from_choices(_read_keyed_choices("question_stem", context_field="fact1")),
    ),
    "toxic_chat": SourceSpec(
        name="toxic_chat",
        hf_id="lmsys/toxic-chat",
        hf_config="toxicchat0124",
        primitive="noul",
        instructions="Is this message toxic?",
        paraphrases=("Is this user message harmful or abusive?", "Should a moderator flag this?"),
        negations=("Is this message acceptable?", "Is this message harmless?"),
        label_names=YES_NO,
        reader=_read_toxic_chat,
    ),
    "toxigen": SourceSpec(
        name="toxigen",
        hf_id="toxigen/toxigen-data",
        hf_config="annotated",
        primitive="noul",
        instructions="Is this text toxic?",
        paraphrases=("Does this statement demean a group of people?",),
        negations=("Is this text benign?",),
        label_names=YES_NO,
        reader=_read_toxigen,
    ),
    "beavertails": SourceSpec(
        name="beavertails",
        hf_id="PKU-Alignment/BeaverTails",
        split="330k_train",
        primitive="noul",
        instructions="Is this response safe?",
        paraphrases=("Is the assistant's reply harmless?",),
        negations=("Is this response harmful?", "Does this reply violate safety norms?"),
        label_names=YES_NO,
        reader=_read_beavertails,
    ),
}

# Checked once, at import. A contaminated id must never reach a loader.
assert_clean(REGISTRY.keys())
assert_clean([spec.hf_id for spec in REGISTRY.values()])

MODE_B_SOURCES: frozenset[str] = frozenset(
    name for name, spec in REGISTRY.items() if spec.is_mode_b
)

# Sources close enough that one's state can answer another's question, and so
# cannot donate a state for abstain augmentation (ADR-012). The sentiment
# corpora are mutually adjacent -- SST-5 and Rotten Tomatoes are both
# movie-review sentiment -- and clinc_oos contains banking intents.
ADJACENT: tuple[frozenset[str], ...] = (
    frozenset({"imdb", "rotten_tomatoes", "sst5", "yelp_review_full", "emotion"}),
    frozenset({"banking77", "clinc_oos", "snips"}),
    frozenset({"ag_news", "dbpedia_14"}),
    frozenset({"snli", "anli", "nli_fever"}),
    frozenset({"mrpc", "qqp", "parade"}),
    frozenset({"toxic_chat", "toxigen", "beavertails"}),
    frozenset(
        {
            "race",
            "commonsense_qa",
            "sciq",
            "openbookqa",
            "arc_easy",
            "race_yesno",
            "sciq_yesno",
            "openbookqa_yesno",
            "strategyqa",
        }
    ),
)


def family_of(source: str) -> str:
    """The task family a source belongs to: its `ADJACENT` group, else itself.

    Holding out mrpc while qqp stays in is not a new task, so calibration
    transfer is measured over families, not sources.
    """
    for group in ADJACENT:
        if source in group:
            return "+".join(sorted(group))
    return source


def adjacency_map() -> dict[str, frozenset[str]]:
    """source -> the sources whose states must not be used to make it unanswerable."""
    return {
        name: frozenset().union(*(group for group in ADJACENT if name in group)) - {name}
        for name in REGISTRY
    }


def _noul_rating(label: int) -> int:
    """Map binary labels to ratings 0 and 8; rating 1 would mean P(yes)=0.125."""
    return 0 if int(label) == 0 else len(NOUL_RATING_TOKENS) - 1


def humanise(label: str) -> str:
    """Expand snake_case and camelCase labels; leave names such as Sci/Tech intact."""
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", label)
    return re.sub(r"[_\-]+", " ", spaced).strip()


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
        # `list` is invariant, so the element type has to be the one Score declares.
        levels: list[JSONContent] = [*options]
        return Score(instructions=spec.instructions, criteria=levels)
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
    """Yield examples, shuffling before applying a row limit.

    A head slice can contain only one class in a label-sorted corpus.
    `load_dataset` is injectable so tests run offline.
    """
    if load_dataset is None:
        from datasets import load_dataset  # noqa: PLC0415

    extra = {"revision": spec.revision} if spec.revision else {}
    dataset = load_dataset(
        spec.hf_id, spec.hf_config, split=spec.split, cache_dir=cache_dir, **extra
    )
    if limit is not None and limit < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(limit))

    # A fixed label set comes from the ClassLabel or the spec; a reader that
    # supplies options per row has none.
    names = _label_names(spec, dataset) if spec.reader is None or spec.label_names else None
    fixed_question = build_question(spec, names) if names is not None else None
    is_noul = spec.primitive == "noul"
    rng = random.Random(seed)

    def emit(state, question: Question, target: int, n_labels: int, i: int) -> Example:
        if not 0 <= target < n_labels:
            raise ValueError(
                f"{spec.name} row {i}: label {target} out of range for "
                f"{n_labels} classes -- the label column is not what we think"
            )
        return Example(
            state=state,
            name=spec.name,
            question=question,
            target=_noul_rating(target) if is_noul else target,
            layout=Layout.STATE_FIRST,  # reassigned by build_mixture
            source=spec.name,
        )

    for i, row in enumerate(dataset):
        if spec.reader is None:
            if names is None or fixed_question is None:
                raise ValueError(f"{spec.name}: no reader, so a fixed label set is required")
            text = row[spec.text_field]
            if not text or not text.strip():
                continue
            yield emit(text.strip(), fixed_question, int(row[spec.label_field]), len(names), i)
            continue

        produced = spec.reader(row, rng)
        if produced is None:
            continue
        rows = produced if isinstance(produced, list) else [produced]
        for state, options, target in rows:
            if _blank(state):
                continue
            if options is None:
                if fixed_question is None or names is None:
                    raise ValueError(f"{spec.name}: reader gave no options and spec has no labels")
                yield emit(state, fixed_question, target, len(names), i)
            else:
                # Duplicate options would collapse in the criteria map and shift
                # the gold index.
                if len(set(options)) != len(options):
                    continue
                question = Choice(
                    instructions=spec.instructions, criteria={opt: None for opt in options}
                )
                yield emit(state, question, target, len(options), i)


def _blank(state) -> bool:
    if isinstance(state, str):
        return not state.strip()
    return not any(str(v).strip() for v in state.values())


# Keep subsampled large taxonomies large enough to train Mode A beyond small sets.
LARGE_SET_MIN_OPTIONS = 15


def augmentation_map() -> dict[str, Augment]:
    """Per-source training augmentation, read off the registry.

    Large taxonomies keep their full option set half the time (the Mode B head's
    data) and are otherwise cut to `LARGE_SET_MIN_OPTIONS` or more, so Mode A
    also learns large lettered sets. Everything else may be cut to two (ADR-026).
    """
    return {
        name: Augment(
            paraphrases=spec.paraphrases,
            negations=spec.negations,
            min_options=LARGE_SET_MIN_OPTIONS if spec.is_mode_b else 2,
            keep_full_fraction=0.5 if spec.is_mode_b else 0.0,
        )
        for name, spec in REGISTRY.items()
    }


def default_weights() -> dict[str, float]:
    """Sampling weight per source: `MODE_B_SHARE` to Mode B, the rest by primitive.

    The remainder splits evenly across the three primitives, then within each,
    so no readout is starved by having few sources.
    """
    mode_b_sources = sorted(MODE_B_SOURCES)
    sources_by_primitive: dict[str, list[str]] = {}
    for name, spec in REGISTRY.items():
        if name not in MODE_B_SOURCES:
            sources_by_primitive.setdefault(spec.primitive, []).append(name)

    weights = {name: MODE_B_SHARE / len(mode_b_sources) for name in mode_b_sources}
    per_primitive = (1.0 - MODE_B_SHARE) / len(sources_by_primitive)
    for group in sources_by_primitive.values():
        for name in group:
            weights[name] = per_primitive / len(group)

    # Renormalise so float drift cannot fail the mixture's sum-to-one check.
    total = sum(weights.values())
    return {name: weight / total for name, weight in sorted(weights.items())}
