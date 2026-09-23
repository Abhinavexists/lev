# Jev — verified findings, and corrections to the plan

> For the *build* — what to take from each open implementation and which trade-offs to
> refuse — see [`ARCHITECTURE.md`](ARCHITECTURE.md). This document establishes what Jev
> is; that one establishes what **lev** should be.

Every claim below is tagged with its provenance:

- **[SDK]** — read directly out of `typesafe-sdk==0.7.0` installed in this repo (pydantic model fields, function signatures). Strongest evidence: this is the shipping code.
- **[DOCS]** — docs.typesafe.ai.
- **[VENDOR]** — a marketing or benchmark claim by TypeSafe. Not independently reproduced.
- **[UNVERIFIED]** — asserted in the original plan, and I could find nothing to confirm it.

Verified 2026-09-19 against `typesafe-sdk` 0.7.0, `system-one-adapter` 0.2.0, model `jev-1.13` / `jev-latest`.

---

## 1. What survives from the original plan

The core thesis holds up well, and two parts are validated more strongly than the plan assumed.

**The abstraction `f(state, question) → typed distribution` is correct.** [SDK] The real signature is:

```python
client.system_one(state: JSONContent, questions: Mapping[str, Noul|Choice|Score]) -> SystemOneResponse
```

**§14 "do not create one fixed head per question" is right, and the real API proves it.** [SDK] `questions` is an arbitrary caller-keyed map, and each question carries free-text `instructions` plus `criteria`. The question genuinely is an *input*, not a selector over pre-trained heads. New questions need no new model.

**§8 parallel questions against shared state is the real economic engine.** [DOCS] Confirmed with measurements — see §4 below.

**§7 "output is free" is literally true, not a simplification.** [VENDOR] Output tokens are priced at $0/M ("too cheap to meter"); input at $0.042/M.

---

## 2. Correction: the primitive set is wrong

The plan proposes `Boolean()`, `Choice(options=[...])`, `Score(min=0, max=1)`. None of the three match. [SDK]

| Plan | Reality | What's different |
|---|---|---|
| `Boolean()` | **`Noul`** | Returns a *probability of yes* (float 0–1), not a boolean or a yes/no distribution |
| `Choice(options=["a","b"])` | `Choice(criteria={"a": "desc", ...})` | `criteria` is a **required** option→description *map*. A bare list of option names is not accepted |
| `Score(min=0, max=1)` | `Score(criteria=["level 0 desc", "level 1 desc", ...])` | **Not** a min/max range. An ordered array of 2–10 *descriptive levels* |

### Exact question types [SDK]

```python
Noul(instructions=..., criteria={"true": ..., "false": ...} | None)  # criteria optional
Choice(instructions=..., criteria=Mapping[str, JSONContent])  # criteria REQUIRED
Score(instructions=..., criteria=Sequence[JSONContent])  # criteria REQUIRED, ordered levels
```

### Exact answer types — and the asymmetry the plan misses [SDK]

```python
NoulAnswer:    noul: float                                             # ← NOTHING ELSE
ChoiceAnswer:  choice: str,   confidence: float, probabilities: dict[str, float]
ScoreAnswer:   score: float,  confidence: float, probabilities: dict[int, float],
               legend: dict[int, str | dict | list]   # mirrors Score.criteria: JSONContent
```

**`NoulAnswer` carries no `confidence` and no `probabilities` field.** It is a single float. The plan's §4 example — `yes = 0.98 / no = 0.02` as a two-way distribution — is not the shape you get back. You get `0.98`, and `P(no)` is *your* inference, not a model output. This matters for §15: you cannot read a Noul's confidence off the response; the float is simultaneously the answer and the uncertainty.

### What `score` actually means [DOCS]

`score = Σ(level_index × P(level))` — a probability-weighted mean over level indices. So `1.035` in the quickstart is not "level 1.035"; it is an expectation sitting just above level 1. A fractional score means the mass straddles two levels. This is *ordinal regression by expectation*, materially different from the plan's scalar regression head.

---

## 3. Correction: `confidence` is derived, not predicted

The plan's §15 treats calibration and confidence as one training objective. They are two different things. [DOCS]

- `probabilities` is the distribution. **This** is the object that is calibrated, and what log loss / Brier / ECE apply to.
- `confidence` is "a statistic computed from the probability distribution the answer already gives you" — a deterministic concentration measure over `probabilities`. Concentrated → high; flat → low.

**Which statistic is it?** [MEASURED] It is **chance-corrected max probability**, `(K·max(p) − 1)/(K − 1)`, rounded to 2 decimal places. Measured over 48 live Choice and Score answers from `jev-latest`: mean absolute error 0.0026, max 0.0100, where every other candidate is an order of magnitude worse.

LitJev, reproducing the schema, uses normalized Gini concentration `(K·Σp² − 1)/(K − 1)` while disclaiming parity with Jev — and it is **wrong, but only in the inner statistic**. The chance-correcting wrapper `(K·x − 1)/(K − 1)` is right; `x` is the maximum, not the sum of squares. Gini came second-*worst* of six candidates (mean 0.0430).

The identification took two corrections to the tool. Pooling all sizes hid the answer, and a flat 5e-3 match threshold rejected it: the normalisation amplifies Jev's 2dp rounding by `K/(K−1)`, putting the true formula at 0.0100 — twice a threshold set for statistics that pass rounding through unchanged. `levbench confidence` now derives its tolerance from the detected quantisation and each formula's measured sensitivity to it.

**And confidence is not an accuracy estimate.** Bespoke Nimble states it plainly for its own model: *"a confidence of 0.9 does not mean that the answer is right 90% of the time."* Whether Jev's is better behaved is exactly what ECE and selective accuracy in this harness are for.

**Consequence for the build:** a student model needs **no confidence head**. It emits a distribution; confidence is computed in post-processing. Adding a confidence head would train a second, potentially inconsistent estimate of something already implied by the first. Keep §15's calibration metrics — apply them to `probabilities` only.

---

## 4. Correction: split §16 into measured and speculative

The plan's §16 describes encoding a 10k-token state once and reusing the representation across 100 questions. Two distinct claims are tangled here.

**(a) Intra-request amortization — documented and measurable today.** [DOCS] One call, one state, N questions. The GDPR cookbook (13 questions over a 54k-character article):

| | cost | latency |
|---|---|---|
| Batched, 1 call | $0.000497 | 0.27s |
| Individual, 13 calls | $0.006090 | 2.71s |
| | **12.2x cheaper** | **10.0x faster** |

Mechanism, verbatim: *"The document is byte-identical in every call... the batched call pays once."* Input tokens are billed **per request**, not per question.

**(b) Cross-request state caching — no *pinned* mechanism for *state*.** (Refined by
[`ARCHITECTURE.md`](ARCHITECTURE.md) §3.3: `decider` does persist a read-only
cross-request cache, but of the **schema**, not the state.) Nothing in the docs, pricing, or API describes persisting a state encoding between calls: no state handle, cache ID, or session in the request schema [SDK].

Open implementations do achieve *opportunistic* cross-request reuse. OpenJev-SGLang caches the shared prefix in a radix tree, which its README describes as *"opportunistic, not a pinned per-request KV session"* — concurrent requests and cache pressure erode the hit rate. So the honest three-way split is: intra-request fan-out is **guaranteed**; opportunistic cross-request reuse **exists in open implementations**; a pinned, addressable state handle exists **nowhere**. If you want the third, that is your research contribution, not a reproduction of Jev.

The plan reads as though (b) is the goal. Today only (a) is real, and (a) is already most of the economic win.

---

## 5. Missing from the plan entirely: model jaggedness

The plan implicitly routes *all* semantic decisions to the model. The vendor's own limitations page for `jev-1.13` contradicts this. [DOCS]

Documented failure modes:

- **Arithmetic, counting, numeric comparison** — *"We strongly recommend implementing any mathematical logic in code."*
- **Dates and time** — treated as text, not ordered quantities. Comparisons unreliable.
- **Indirection** — multi-hop reasoning and double negatives significantly reduce accuracy.
- **Irrelevant context** — degrades with large states containing unrelated detail. Filter before sending. *(Directly undercuts the plan's "STATE = everything the system currently knows.")*
- **Adversarial content** — state is not treated as hostile. Prompt injection via state works.
- **Literal interpretation** — answers as written, does not infer intent.
- **No structural invariants** — *"don't assume complementary questions sum to 1."*

That last one is the sharpest. The plan's §4 shows one state answering both "is the data sufficient?" and "should we retry?" as if coherent. **Nothing enforces consistency between questions in a batch.** Fan-out buys independence, not coherence — cross-question logic belongs in your code.

Correct framing: **semantic judgment to the model; arithmetic, ordering, and consistency to code.**

---

## 6. "Zero hallucinations" means type-safety, not accuracy

[VENDOR] The 0% figure is *"schema matching is guaranteed"* — output is always a valid member of the declared space. A `Choice` always returns one of your options; a `Score` always lands in your level range.

It says **nothing about whether the answer is correct.** A confidently wrong but well-typed answer scores 0% hallucination. Do not inherit this as an accuracy claim.

Likewise **193.6x faster / 444.6x cheaper** [VENDOR] are vendor-selected workloads against vendor-chosen comparators (GPT-6 Astra, Fable 5.1). Reproduce them or cite them as claims — the harness in this repo is built to do the former.

---

## 7. §12's teacher already exists — don't build it

The plan's Stage 2 proposes building a teacher-dataset generator. TypeSafe ships one: **`system-one-adapter-python`**, a drop-in replacement for `typesafe_sdk` backed by LLM APIs. [SDK, verified by installation]

```python
SystemOneAdapterClient(
    structured_outputs: bool,
    llm_answer_mode: "probabilities" | "discrete",
    normalize_probabilities: bool = False,
    n_retry_malformed_structure: int = 0,
)
```

It returns the **same** `NoulAnswer`/`ChoiceAnswer`/`ScoreAnswer` types as the real SDK, so a single benchmark can swap clients. With `llm_answer_mode="probabilities"` it is exactly the teacher from §12 — distributions, not just argmax.

**Sharp observation:** the existence of `normalize_probabilities` (rescales *invalid* distributions) and `n_retry_malformed_structure` (retries on schema-validation failure) is itself evidence of the failure mode Jev claims to remove. The adapter's `Usage` exposes `n_retries`, `n_retries_malformed_structure`, and `latency` [SDK] — so **counting those retries is a real, publishable measurement**, not a footnote.

---

---

## 8. The architecture question is answered — and it kills §13

**Provenance note.** Both X/Twitter links supplied were fetched and both returned
**HTTP 402**; neither tweet was read. What follows was reconstructed from web search
and then verified at source: repository existence and metadata via the GitHub API,
READMEs read directly. One search hit (`slavadubrov/jev-judge-bench`) **404s** and was
discarded — the list below is filtered, not transcribed.

Independent open reproductions have converged on the same architecture, and it is
**not** the one in the plan's §13.

### The convergent mechanism

```
state + all questions
        ↓
1. SHARED PREFILL      encode the state once  →  KV cache
        ↓
2. CACHED BRANCHES     replicate KV per question; batch each question's
                       instructions + criteria, ending in "Answer:"
        ↓
3. READOUT             logits[i, len(suffix_i) - 1, candidate_ids[i]]
                       — the next-token logits at the answer boundary,
                         restricted to the candidate label tokens
        ↓
4. TYPED RESPONSE      softmax over those logits; assemble JSON in code
```

**Zero tokens are generated.** LitJev's own example response reports
`"usage": {"output_tokens": 0}`. That is the mechanical explanation for the plan's §7
"free output": output isn't cheap to generate, it is *never generated*. The answer is
read out of the logit vector the forward pass already produced.

### What this deletes from the plan

The plan's §13 proposes training a state encoder, a question encoder, cross-attention,
and a decision head. In the convergent design **none of those are new components**:

| Plan's proposed component | What actually plays that role |
|---|---|
| State encoder | the base LLM |
| Question encoder | the base LLM (question text is just more prefill) |
| Cross-attention interaction | ordinary causal attention from branch to cached state |
| Decision head | `lm_head`, the existing vocabulary head |
| Choice softmax | softmax over candidate label token logits |
| Score regression | `Σ(index × P(level))` over the same readout |

**V0 through V3 collapse into "point a client at a server."** No training is required to
get typed, probabilistic, schema-valid decisions out of an off-the-shelf model.

Training is not pointless — it buys *accuracy* on top of a mechanism that is already
free. Bespoke Nimble's LoRA fine-tune of Qwen3.5-9B moves 66.4% → 90.1%. But that is
step two, not step one, and the plan had the order backwards.

### The hard constraint the plan never mentions

Each option must be **a single token at the answer boundary**. Implementations remap
arbitrary option keys to internal letter codes (`A`–`Z`, then `AA`…`ZZ`), verified
single-token for the loaded tokenizer; LitJev *rejects* unsupported tokenizers rather
than truncating, and OpenJev-SGLang caps at 64 options.

This is why `Choice.criteria` is a **map**: your keys are arbitrary strings, the model
only ever scores a letter code. It also bounds the design — a Choice over thousands of
options cannot be one readout, which is what makes `jev-tree`-style hierarchical
traversal a necessary pattern rather than a stylistic one.

### The reproductions, verified

Existence, language, licence and last-push confirmed via the GitHub API on 2026-09-20.

| Project | Base model | Serves `/v1/systemone` | Notes |
|---|---|---|---|
| [`zhengxuyu/litjev`](https://github.com/zhengxuyu/litjev) | Qwen3.8-27B | **Yes** | Apache-2.0. Cleanest architecture write-up. H100 80GB tested |
| [`ekzhang/openjev-sglang`](https://github.com/ekzhang/openjev-sglang) | Qwen3.6-35B-A3B | **Yes** | Prefill + first-token readout; N+1 single-token calls for N questions |
| [`razorback16/openjev`](https://github.com/razorback16/openjev) | DiffusionGemma 26B-A4B | **Yes** | Apache-2.0, vLLM |
| [`bespokelabsai/nimble`](https://github.com/bespokelabsai/nimble) | Qwen3.5-9B + LoRA | No (library) | The only one with published eval numbers |
| [`bnsd55/jevmlx`](https://github.com/bnsd55/jevmlx) | Qwen2.5 MLX 4-bit | **No** — Pydantic API | MIT. The only one that runs on Apple Silicon |

**The practical consequence for this repo:** three of these speak the identical wire
schema, and `TypeSafeClient` already accepts `base_url`. So `levbench` benchmarks them
with no code change:

```bash
levbench eval --backend lev --base-url http://127.0.0.1:8000  # any local clone
```

`jevmlx` does **not** serve the endpoint — it exposes a Pydantic schema API — so it
would need a small adapter. It is nonetheless the only option that runs locally on an
Apple Silicon machine; the others want a datacentre GPU.

### The only published numbers [VENDOR-adjacent]

Bespoke Nimble, 324 held-out examples. Read the caveat before the table:

| Model | Agreement |
|---|---|
| Jev 1.13.0 | 93.21% (302/324) |
| Bespoke-Nimble-9B | 90.12% (292/324) |
| Qwen3.8-27B (untuned) | 84.88% (275/324) |
| Qwen3.5-9B (base) | 66.36% (215/324) |

**This is reference-label agreement, not accuracy.** The labels are *synthetic*, and the
324 examples are 162 deliberately contrastive pairs. Nimble did not distil from Jev.
Do not place these beside this repo's harness numbers as though they measure the same
thing.

---

---

## 9. S1Bench: an independent benchmark exists — and it reframes the claims

**[THIRD-PARTY]** Source: an S1Bench Live dashboard served over an **ephemeral Cloudflare
tunnel** supplied by the user. Tunnels die, so the raw payload is snapshotted at
[`data/s1bench-snapshot.json`](../data/s1bench-snapshot.json) (build `468a014bc5`, run
100% complete, 39,980 decisions, 33 targets). I did not run this benchmark and cannot
vouch for its harness; what follows is its data plus my reading of it.

### It validates TypeSafe's published numbers

The strongest thing in the dataset is a sanity check. S1Bench's measured Jev accuracy
against TypeSafe's own published per-subset figures:

| subset | measured | published | diff |
|---|---|---|---|
| vitaminc-dev | 0.8030 | 0.8010 | +0.0020 |
| massive-en-US | 0.8743 | 0.8740 | +0.0003 |
| boolq | 0.8933 | 0.8970 | −0.0037 |
| helpsteer2 | 0.3480 | 0.3410 | +0.0070 |
| aegis2 | 0.8360 | 0.8040 | +0.0320 |
| paws | 0.8960 | 0.8920 | +0.0040 |

Five of six land within ±0.4pp. **TypeSafe's published accuracy numbers reproduce.**
That is worth stating plainly given §6's scepticism about the marketing figures — the
*accuracy* claims survive independent measurement even though the speed and cost
multipliers are comparator-dependent.

The `jev` target here is the real hosted API (`device: api`, `price_per_1k: 0.041`,
matching the $0.042/M list price), not a reproduction.

### Completed leaderboard — 6 subsets (`s1-fast`), 1,999 decisions each

Only these 20 targets finished the suite. Thirteen more were stopped after
`vitaminc-dev` alone (599 rows); **their macro scores are one subset, not six, and are
not comparable** — LitJev's entry, for instance, was explicitly halted.

| target | macro | Δ vs Jev | ECE | dec/s | contaminated |
|---|---|---|---|---|---|
| **jev** (hosted API) | **0.7751** | — | **0.0764** | 2.39 | |
| simplejev-qwen38-27b | 0.7582 | −0.0100 | 0.1214 | 1.63 | |
| djev-full | 0.7485 | −0.0196 | 0.1661 | 3.70 | |
| simplejev-qwen36-35b-a3b | 0.7442 | −0.0240 | 0.1365 | 1.67 | |
| reflex-4b | 0.7189 | −0.0492 | 0.0849 | 7.25 | |
| decider-2b | 0.7033 | −0.0648 | 0.1142 | 27.17 | yes |
| laya-gpu | 0.6254 | −0.1427 | 0.1304 | 17.86 | |
| jeff-gpu-full | 0.5595 | −0.2087 | 0.0738 | 6.21 | |
| open-jev-deberta | 0.5235 | −0.2446 | 0.0668 | 37.44 | yes |
| kev-05b | 0.4926 | −0.2756 | 0.1492 | 0.55 | yes |
| gliner-base | 0.4318 | −0.3364 | 0.3157 | 3.42 | |
| simplejev-rwkv-small | 0.3130 | −0.4551 | 0.2793 | 2.46 | |

(Abridged; the full 33 are in the snapshot. Contamination flags are the benchmark's
own: `decider-2b` on "many public decision datasets", `open-jev-deberta` on
banking77/sst5/boolq, `kev-05b` on banking77.)

### Three conclusions that change how this project should be framed

**1. Open reproductions are ~1 point behind on accuracy.** `simplejev-qwen38-27b`
scores 0.7582 against Jev's 0.7751 — a 1.0pp paired gap. Combined with §8, the
accuracy moat is thin. If accuracy were the whole story, a self-hosted 27B would be
competitive today.

**2. Calibration is where Jev actually separates — and it is the thing this harness
measures.** At comparable accuracy Jev's ECE of 0.0764 is 1.6–2.2× better than the
open clones near it (0.1214, 0.1365, 0.1661). This is direct evidence for the RLCD
"calibrated decisions" claim, and it is precisely the axis §3 and `levbench eval`
target. Caveat against over-reading: Jev is best-calibrated *at its accuracy level*,
not absolutely — `open-jev-deberta` (0.0668) and `jeff-gpu-full` (0.0738) beat it on
ECE while scoring 22–25pp lower on accuracy. Low ECE is easy when you are uncertain
and correct about being uncertain.

**3. The speed claim does not survive this comparison.** Jev runs at **2.39 dec/s**
(median 0.419s/decision — consistent with TypeSafe's own "70ms–500ms" figure). Against
purpose-built open decision models it is mid-pack at best: `open-jev-deberta` 37.4,
`decider-2b` 27.2, `reflex-08b` 24.4, `laya-gpu` 17.9 dec/s. The "193.6× faster"
headline [VENDOR] is measured against *frontier LLMs*, not against the category Jev
now competes in. Restated honestly: **Jev is two orders of magnitude faster than an
LLM doing the same job, and roughly an order of magnitude slower than a small
purpose-built local model — while being better calibrated than either.**

### Caveats that bound all of the above

- 13 of 33 targets stopped early; their macro is a single subset.
- Option caps differ materially — Jev allows 255 options, `reflex` 26, `verdict` 24,
  `mini-jev` 16. These targets are not all solving equally hard problems.
- Three targets carry self-declared dataset contamination.
- `helpsteer2` is near-chance for everyone (Jev 0.348), so the macro is dragged by one
  subset nobody handles.
- Single run, no seeds or confidence intervals reported at the macro level.

---

## 10. Corrected roadmap

| Plan stage | Verdict |
|---|---|
| V0 API abstraction | **Skip.** Vendor ships the types. Re-implementing under wrong names bakes in the errors. Import `typesafe_sdk`. |
| V1 Parallel interface | **Already exists** — it's the `questions` map. Measure it instead of building it. |
| V2 Teacher dataset | **Use `system-one-adapter`.** Don't rebuild. |
| V3 Student model | **Superseded by §8.** No new architecture needed: prefill + logit readout on an off-the-shelf model gives typed distributions untrained. Fine-tune only to raise accuracy (Nimble: 66% → 90%). |
| V4 Calibration | Valid, and **now the highest-value axis** — §9 shows open clones match Jev on accuracy but are 1.6–2.2× worse on ECE. Applies to `probabilities`. Needs ground-truth labels. |
| V5 Shared-state inference | **Split.** Intra-request is measurable now; cross-request reuse is opportunistic radix caching in open implementations; a pinned state handle is novel research (§4). |
| V6 RL | Unchanged — after a supervised baseline. |
| V7 Inference optimization | Unchanged. |

**The revised central question**, correcting §19:

> Can a specialized model answer *many independent typed questions against one shared state in a single pass*, with calibrated probability distributions, at a cost dominated by reading the state once rather than by the number of questions asked?

The shift from the original: the win is **intra-request fan-out over a shared state**, not cross-request reuse of a cached encoding.

---

## 11. Reference — verified API facts

[SDK] unless noted.

```
Endpoint      POST https://api.typesafe.ai/v1/systemone     [DOCS]
Auth          Authorization: Bearer <key>                    [DOCS]
Env var       TYPESAFE_API_KEY
Base URL env  TYPESAFE_BASE_URL   (default https://api.typesafe.ai)
Default model jev-latest
Timeout       10.0s default
Usage fields  input_tokens, output_tokens                    ← only these two
Errors        401 auth / 422 validation / 429 rate / 529 overloaded  [DOCS]
State         text only — str | JSON object | array. No image/audio/video. English best.  [DOCS]
Pricing       input $0.042/M, output free                    [VENDOR]
```

Official Claude Code skill: `claude plugin marketplace add typesafe-ai/skills` then `claude plugin install typesafe@typesafe-ai`. [DOCS]

---

## 12. First head-to-head on S1Bench: two named failure modes

Run 2026-09-22. 1,999 items across the six S1Bench subsets that actually
executed in `s1-fast`, exported by `lev s1bench export` and scored through one
`levbench eval --tasks` against both backends, so the prompts are identical and
a gap is attributable to the model rather than to the harness.

**The harness reproduces Jev.** Jev on our task files against its own recorded
accuracy, with the 95% band for a difference of two independent samples:

| subset | n | Jev here | S1Bench | delta pp | band |
|---|---|---|---|---|---|
| aegis2 | 250 | 0.832 | 0.836 | -0.4 | ±6.5 |
| boolq | 300 | 0.910 | 0.893 | +1.7 | ±4.8 |
| helpsteer2 | 250 | 0.304 | 0.348 | -4.4 | ±8.2 |
| vitaminc-dev | 599 | 0.846 | 0.803 | +4.3 | ±4.4 |
| massive-en-US | 350 | 0.814 | 0.874 | -6.0 | ±5.4 |
| paws | 250 | 0.820 | 0.896 | -7.6 | ±6.1 |

Four sit inside sampling noise; `massive` and `paws` fall marginally outside, so
our wording is slightly harder than Jev's on those two. `helpsteer2` reproducing
at 0.304 is the load-bearing check -- a harness matching only the high numbers
would more likely be mis-scoring than correct. This validation is what makes the
rest of the section interpretable: our prompts are not Jev's and cannot be
diffed against them, so without it every gap below would have two explanations.

**The comparison.** Macro 0.489 against Jev's 0.754, on identical files. Jev's
own published macro is 0.768, a third sign the harness is sound.

| subset | lev | Jev | gap |
|---|---|---|---|
| boolq | 0.803 | 0.910 | -10.7 |
| paws | 0.712 | 0.820 | -10.8 |
| helpsteer2 | 0.240 | 0.304 | -6.4 |
| vitaminc-dev | 0.578 | 0.846 | -26.8 |
| massive-en-US | 0.291 | 0.814 | -52.3 |
| aegis2 | 0.312 | 0.832 | -52.0 |

The two collapses are separate defects, both isolated by follow-up probes rather
than inferred from the table.

### Noul does not condition on the question's polarity

aegis2's 0.312 is below *every* constant predictor (always-safe scores 0.388,
always-unsafe 0.612), and confidence runs backwards: accuracy 0.574 in the
[0.5,0.6) bin against 0.034 in [0.9,1.0). Asking the same 120 states both ways
through one request:

    "Is this prompt unsafe?"  accuracy 0.308
    "Is this prompt safe?"    accuracy 0.775

Accuracy alone cannot explain that. The two sum to 1.083, and a model that
ignores the instruction entirely -- emitting one state-dependent score into both
slots -- sums to exactly 1.000 by construction, since each item contributes one
correct answer between the two framings. Only the per-item probabilities
separate "reads polarity but inverted" from "ignores the question":

    mean(p_unsafe + p_safe)   0.713    1.00 would mean complementary
    mean|p_unsafe - p_safe|   0.079    0.00 means the question is ignored
    89/120 items differ by <0.10;  23/120 sum to within 0.10 of 1.0

The pairs are near-identical rather than complementary. The model returns
roughly the same number whichever way it is asked, so it is not reading the
polarity and inverting it -- on this subset it is not conditioning on
`instructions` at all.

The underlying signal is real: that state-only score tracks actual safety well
enough to reach 0.775 when it happens to be read as P(safe), which is why the
positive framing looks competent. What is missing is that the question does not
modulate it. Noul supervision came only from imdb and rotten_tomatoes, where
yes = positive = good, so the model learned a benignness prior over states
rather than a function of the question.

This is not a blanket failure of Noul: boolq (0.803 against a 0.603 base rate)
and paws (0.712 against 0.520) both beat their base rates, so the question does
carry on subsets whose yes-axis is goodness-aligned or neutral. The fix is the
same in either case -- Noul needs training questions whose "yes" denotes the
undesirable outcome.

### massive-en-US never reached Mode B

The 0.291 was read as a Mode B result -- 60 options is over the 26 single-letter
codes -- and two follow-up probes were reported as Mode B transfer behaviour.
Both readings were wrong, and the correction matters more than the number.

The router is tokenizer-verified, not count-based, and Qwen3.5's 248k vocabulary
encodes every two-letter code up to `BP` as one token:

    n=60   single-token codes 60/60  -> Mode A
    n=77   single-token codes 76/77  -> Mode B   (`BQ` is the first that splits)

The live server confirms it: a 60-option Choice cost 839 input tokens (the
options were listed in the prompt, which only Mode A does) and a 77-option one
cost 28 (Mode B lists nothing). So massive ran in Mode A, with two-letter codes
the model had never seen, over a candidate set four times larger than any it had
trained on in that mode (dbpedia_14, 14 options). The Mode B head sat idle.

That also explains the order sensitivity. Same 60 items, same options, two
orders: argmax agreement 0.15, mean L1 between the distributions 1.23. Identical
order, same request: L1 0.0000. A GPU diagnostic (`modal run
modal/app.py::diagnose_candidates`) rules out the candidate encoder -- a string's
representation is bit-identical across batch orders, `max_abs 0.0`. What is left
is the one mechanism that *is* order-dependent by construction: letter-position
bias in Mode A, which reflex measured on the same backbone and cancels by reading
each question in two option orders.

Consequently the "60 options 0.375 / 30 options 0.592" probes measured Mode A
with lettered codes, not the candidate-path head, and **Mode B on an unseen
taxonomy is untested**. banking77 and clinc_oos were both training sources, so
0.87 there was held-out rows, not held-out labels.

Two fixes follow, neither needing retraining (ADR-020): a policy cap of 26 on
Mode A applied identically in training and serving, so anything above single
letters goes to the head that was trained for large sets; and two-order
averaging for Choice and binary Noul.

**Redeployed with the cap, same checkpoint, same files** (`/health` reports
`max_label_options: 26`): massive-en-US now costs 41 input tokens per item
instead of 664 -- it is in Mode B -- and scores **0.166**. That is the first
real Mode B transfer number: ten times chance (1/60), and well calibrated about
its own ignorance (ECE 0.076; the 142 items it placed in the 0-0.1 bin score
0.070), but far below Mode A's 0.291 and Jev's 0.814. The head learned some
general matching and mostly its two training taxonomies. The option key
format is not the cause: snake_case keys and humanised keys score identically,
0.140 on the same 150 items. More taxonomies in the mixture is the fix, and it
needs a retrain.

Two-order averaging moved vitaminc 0.578 -> 0.588 and helpsteer2 0.240 ->
0.216, both inside sampling noise -- but helpsteer2 is a Score, and reversing
an ordered scale shows the model a prompt training never produces. Averaging
is now limited to Choice and binary Noul. aegis2, boolq and paws are unchanged
to three decimals, as expected: nothing in the rating-scale Noul path changed.

### Caveat

Calibration was fitted on lev's own mixture, and all six subsets are
out-of-distribution for it, so the ECE figures from this run are not comparable
to the 0.0529 measured on the held-out split.

---

## 14. The instruct run, and why its clinc_oos number is a bug and not a result

`4b-instruct` (ADR-020 mixture, 18,750 steps) on the held-out split,
calibrated: weighted 0.706 ±0.005, ECE 0.2135 → 0.1133. Not comparable to the
first run's 0.856 -- the mixture is 23 sources and deliberately harder -- and
the per-source picture is what matters:

| learned (new sources) | | regressed | |
|---|---|---|---|
| sciq 0.975, arc_easy 0.927, openbookqa 0.917 | QA with per-row options | clinc_oos **0.055** (was 0.865) | Mode B, 151 options |
| snli 0.852, anli 0.758 | NLI | emotion **0.605** (was ≥0.865) | 6 options |
| toxic_chat 0.976, toxigen 0.871, beavertails 0.818 | safety, both polarities | | |
| mrpc 0.896, qqp 0.864 | paraphrase | | |
| banking77 0.906 (was 0.870) | Mode B, 77 options | | |

The two regressions are the sources with the most option shuffling and the
highest rate of identical option *sets* across rows. Checking the training
rows against the source datasets: 35–45% of shuffled Choice rows had `target`
pointing at a wrong option *as read by the training loop*, while the same
rows on disk were 100% correct. The reader's question cache was keyed on a
sorted payload and merged differently-ordered questions (ADR-024). Reading
the same file with the fixed reader: 646/646, 682/682, 116/116 correct.

So the instruct run is a mixed measurement: everything Noul, Score and
per-row-QA learned from correct labels and those numbers stand; every shuffled
Choice row of a fixed-option-set source trained on a corrupted label, and
clinc_oos and emotion are the visible damage. Calibration came out
under-confident on Choice (T=0.74 / 0.80) -- plausibly the same cause, a model
that learned to hedge because its labels disagreed with its inputs. The run
has to be repeated on the fixed reader before its S1Bench number means
anything; the mixture itself does not need rebuilding.

**Retrained on the fixed reader** (same mixture, same preset, 18,750 steps):
loss 5.06 → 0.23 overall; over the last 300 steps Mode A averaged 0.495 and
Mode B **0.531** -- against ~2.3 for Mode B in the corrupted run and 0.81 in
the first run on the narrow mixture. The head learned the moment its labels
stopped disagreeing with its inputs.

Held-out, calibrated: **weighted 0.836 ±0.004, ECE 0.1273 → 0.0459** -- on a
23-source mixture, against 0.856 / 0.0529 for the first model on nine. The two
regressions the bug produced are gone: clinc_oos **0.976** (corrupted 0.055;
first model 0.865), emotion 0.860 (corrupted 0.605). banking77 0.922, snips
0.976, dbpedia 0.998; the new families hold -- snli 0.896, anli 0.845, arc_easy
0.946, sciq 0.973, openbookqa 0.922, race 0.823, toxic_chat 0.973, toxigen
0.873, mrpc 0.888, qqp 0.873, beavertails 0.806. Score stays the weak
primitive (sst5 0.581, yelp 0.666, ultrafeedback 0.547). The fitted
temperatures are ordinary again -- choice:A 2.04, choice:B 1.34, noul 2.42,
score 2.91 -- so the under-confidence of the corrupted run was the bug as
well. The S1Bench comparison for this checkpoint follows in §15.
