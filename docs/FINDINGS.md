# Jev — verified findings, and corrections to the plan

> For the *build* — what to take from each open implementation and which trade-offs to refuse — see [`ARCHITECTURE.md`](ARCHITECTURE.md). This document establishes what Jev is; that one establishes what **lev** should be.

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
| --- | --- | --- |
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
| --- | --- | --- |
| Batched, 1 call | $0.000497 | 0.27s |
| Individual, 13 calls | $0.006090 | 2.71s |
| | **12.2x cheaper** | **10.0x faster** |

Mechanism, verbatim: *"The document is byte-identical in every call... the batched call pays once."* Input tokens are billed **per request**, not per question.

**(b) Cross-request state caching — no *pinned* mechanism for *state*.** (Refined by [`ARCHITECTURE.md`](ARCHITECTURE.md) §3.3: `decider` does persist a read-only cross-request cache, but of the **schema**, not the state.) Nothing in the docs, pricing, or API describes persisting a state encoding between calls: no state handle, cache ID, or session in the request schema [SDK].

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

**Provenance note.** Both X/Twitter links supplied were fetched and both returned **HTTP 402**; neither tweet was read. What follows was reconstructed from web search and then verified at source: repository existence and metadata via the GitHub API, READMEs read directly. One search hit (`slavadubrov/jev-judge-bench`) **404s** and was discarded — the list below is filtered, not transcribed.

Independent open reproductions have converged on the same architecture, and it is **not** the one in the plan's §13.

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

**Zero tokens are generated.** LitJev's own example response reports `"usage": {"output_tokens": 0}`. That is the mechanical explanation for the plan's §7 "free output": output isn't cheap to generate, it is *never generated*. The answer is read out of the logit vector the forward pass already produced.

### What this deletes from the plan

The plan's §13 proposes training a state encoder, a question encoder, cross-attention, and a decision head. In the convergent design **none of those are new components**:

| Plan's proposed component | What actually plays that role |
| --- | --- |
| State encoder | the base LLM |
| Question encoder | the base LLM (question text is just more prefill) |
| Cross-attention interaction | ordinary causal attention from branch to cached state |
| Decision head | `lm_head`, the existing vocabulary head |
| Choice softmax | softmax over candidate label token logits |
| Score regression | `Σ(index × P(level))` over the same readout |

**V0 through V3 collapse into "point a client at a server."** No training is required to get typed, probabilistic, schema-valid decisions out of an off-the-shelf model.

Training is not pointless — it buys *accuracy* on top of a mechanism that is already free. Bespoke Nimble's LoRA fine-tune of Qwen3.5-9B moves 66.4% → 90.1%. But that is step two, not step one, and the plan had the order backwards.

### The hard constraint the plan never mentions

Each option must be **a single token at the answer boundary**. Implementations remap arbitrary option keys to internal letter codes (`A`–`Z`, then `AA`…`ZZ`), verified single-token for the loaded tokenizer; LitJev *rejects* unsupported tokenizers rather than truncating, and OpenJev-SGLang caps at 64 options.

This is why `Choice.criteria` is a **map**: your keys are arbitrary strings, the model only ever scores a letter code. It also bounds the design — a Choice over thousands of options cannot be one readout, which is what makes `jev-tree`-style hierarchical traversal a necessary pattern rather than a stylistic one.

### The reproductions, verified

Existence, language, licence and last-push confirmed via the GitHub API on 2026-09-20.

| Project | Base model | Serves `/v1/systemone` | Notes |
| --- | --- | --- | --- |
| [`zhengxuyu/litjev`](https://github.com/zhengxuyu/litjev) | Qwen3.8-27B | **Yes** | Apache-2.0. Cleanest architecture write-up. H100 80GB tested |
| [`ekzhang/openjev-sglang`](https://github.com/ekzhang/openjev-sglang) | Qwen3.6-35B-A3B | **Yes** | Prefill + first-token readout; N+1 single-token calls for N questions |
| [`razorback16/openjev`](https://github.com/razorback16/openjev) | DiffusionGemma 26B-A4B | **Yes** | Apache-2.0, vLLM |
| [`bespokelabsai/nimble`](https://github.com/bespokelabsai/nimble) | Qwen3.5-9B + LoRA | No (library) | The only one with published eval numbers |
| [`bnsd55/jevmlx`](https://github.com/bnsd55/jevmlx) | Qwen2.5 MLX 4-bit | **No** — Pydantic API | MIT. The only one that runs on Apple Silicon |

**The practical consequence for this repo:** three of these speak the identical wire schema, and `TypeSafeClient` already accepts `base_url`. So `levbench` benchmarks them with no code change:

```bash
levbench eval --backend lev --base-url http://127.0.0.1:8000  # any local clone
```

`jevmlx` does **not** serve the endpoint — it exposes a Pydantic schema API — so it would need a small adapter. It is nonetheless the only option that runs locally on an Apple Silicon machine; the others want a datacentre GPU.

### The only published numbers [VENDOR-adjacent]

Bespoke Nimble, 324 held-out examples. Read the caveat before the table:

| Model | Agreement |
| --- | --- |
| Jev 1.13.0 | 93.21% (302/324) |
| Bespoke-Nimble-9B | 90.12% (292/324) |
| Qwen3.8-27B (untuned) | 84.88% (275/324) |
| Qwen3.5-9B (base) | 66.36% (215/324) |

**This is reference-label agreement, not accuracy.** The labels are *synthetic*, and the 324 examples are 162 deliberately contrastive pairs. Nimble did not distil from Jev. Do not place these beside this repo's harness numbers as though they measure the same thing.

---

---

## 9. S1Bench: an independent benchmark exists — and it reframes the claims

**[THIRD-PARTY]** Source: an S1Bench Live dashboard served over an **ephemeral Cloudflare tunnel** supplied by the user. Tunnels die, so the raw payload is snapshotted at [`data/s1bench-snapshot.json`](../data/s1bench-snapshot.json) (build `468a014bc5`, run 100% complete, 39,980 decisions, 33 targets). I did not run this benchmark and cannot vouch for its harness; what follows is its data plus my reading of it.

### It validates TypeSafe's published numbers

The strongest thing in the dataset is a sanity check. S1Bench's measured Jev accuracy against TypeSafe's own published per-subset figures:

| subset | measured | published | diff |
| --- | --- | --- | --- |
| vitaminc-dev | 0.8030 | 0.8010 | +0.0020 |
| massive-en-US | 0.8743 | 0.8740 | +0.0003 |
| boolq | 0.8933 | 0.8970 | −0.0037 |
| helpsteer2 | 0.3480 | 0.3410 | +0.0070 |
| aegis2 | 0.8360 | 0.8040 | +0.0320 |
| paws | 0.8960 | 0.8920 | +0.0040 |

Five of six land within ±0.4pp. **TypeSafe's published accuracy numbers reproduce.** That is worth stating plainly given §6's scepticism about the marketing figures — the *accuracy* claims survive independent measurement even though the speed and cost multipliers are comparator-dependent.

The `jev` target here is the real hosted API (`device: api`, `price_per_1k: 0.041`, matching the $0.042/M list price), not a reproduction.

### Completed leaderboard — 6 subsets (`s1-fast`), 1,999 decisions each

Only these 20 targets finished the suite. Thirteen more were stopped after `vitaminc-dev` alone (599 rows); **their macro scores are one subset, not six, and are not comparable** — LitJev's entry, for instance, was explicitly halted.

| target | macro | Δ vs Jev | ECE | dec/s | contaminated |
| --- | --- | --- | --- | --- | --- |
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

(Abridged; the full 33 are in the snapshot. Contamination flags are the benchmark's own: `decider-2b` on "many public decision datasets", `open-jev-deberta` on banking77/sst5/boolq, `kev-05b` on banking77.)

### Three conclusions that change how this project should be framed

**1. Open reproductions are ~1 point behind on accuracy.** `simplejev-qwen38-27b` scores 0.7582 against Jev's 0.7751 — a 1.0pp paired gap. Combined with §8, the accuracy moat is thin. If accuracy were the whole story, a self-hosted 27B would be competitive today.

**2. Calibration is where Jev actually separates — and it is the thing this harness measures.** At comparable accuracy Jev's ECE of 0.0764 is 1.6–2.2× better than the open clones near it (0.1214, 0.1365, 0.1661). This is direct evidence for the RLCD "calibrated decisions" claim, and it is precisely the axis §3 and `levbench eval` target. Caveat against over-reading: Jev is best-calibrated *at its accuracy level*, not absolutely — `open-jev-deberta` (0.0668) and `jeff-gpu-full` (0.0738) beat it on ECE while scoring 22–25pp lower on accuracy. Low ECE is easy when you are uncertain and correct about being uncertain.

**3. The speed claim does not survive this comparison.** Jev runs at **2.39 dec/s** (median 0.419s/decision — consistent with TypeSafe's own "70ms–500ms" figure). Against purpose-built open decision models it is mid-pack at best: `open-jev-deberta` 37.4, `decider-2b` 27.2, `reflex-08b` 24.4, `laya-gpu` 17.9 dec/s. The "193.6× faster" headline [VENDOR] is measured against *frontier LLMs*, not against the category Jev now competes in. Restated honestly: **Jev is two orders of magnitude faster than an LLM doing the same job, and roughly an order of magnitude slower than a small purpose-built local model — while being better calibrated than either.**

### Caveats that bound all of the above

- 13 of 33 targets stopped early; their macro is a single subset.
- Option caps differ materially — Jev allows 255 options, `reflex` 26, `verdict` 24, `mini-jev` 16. These targets are not all solving equally hard problems.
- Three targets carry self-declared dataset contamination.
- `helpsteer2` is near-chance for everyone (Jev 0.348), so the macro is dragged by one subset nobody handles.
- Single run, no seeds or confidence intervals reported at the macro level.

---

## 10. Corrected roadmap

| Plan stage | Verdict |
| --- | --- |
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

```text
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

> §§12–16 use this repo's own loaders for the six subsets, not the items S1Bench scores: massive-en-US there is 60 intents on the validation split, where S1Bench asks 18 scenarios on test, and all six use this repo's question wording, not S1Bench's. Their numbers compare runs with each other, not with the board. §17 re-runs on S1Bench's pinned items.

Run 2026-09-22. 1,999 items across the six S1Bench subsets that actually executed in `s1-fast`, exported by `lev s1bench export` and scored through one `levbench eval --tasks` against both backends, so the prompts are identical and a gap is attributable to the model rather than to the harness.

**The harness reproduces Jev.** Jev on our task files against its own recorded accuracy, with the 95% band for a difference of two independent samples:

| subset | n | Jev here | S1Bench | delta pp | band |
| --- | --- | --- | --- | --- | --- |
| aegis2 | 250 | 0.832 | 0.836 | -0.4 | ±6.5 |
| boolq | 300 | 0.910 | 0.893 | +1.7 | ±4.8 |
| helpsteer2 | 250 | 0.304 | 0.348 | -4.4 | ±8.2 |
| vitaminc-dev | 599 | 0.846 | 0.803 | +4.3 | ±4.4 |
| massive-en-US | 350 | 0.814 | 0.874 | -6.0 | ±5.4 |
| paws | 250 | 0.820 | 0.896 | -7.6 | ±6.1 |

Four sit inside sampling noise; `massive` and `paws` fall marginally outside, so our wording is slightly harder than Jev's on those two. `helpsteer2` reproducing at 0.304 is the load-bearing check -- a harness matching only the high numbers would more likely be mis-scoring than correct. This validation is what makes the rest of the section interpretable: our prompts are not Jev's and cannot be diffed against them, so without it every gap below would have two explanations.

**The comparison.** Macro 0.489 against Jev's 0.754, on identical files. Jev's own published macro is 0.768, a third sign the harness is sound.

| subset | lev | Jev | gap |
| --- | --- | --- | --- |
| boolq | 0.803 | 0.910 | -10.7 |
| paws | 0.712 | 0.820 | -10.8 |
| helpsteer2 | 0.240 | 0.304 | -6.4 |
| vitaminc-dev | 0.578 | 0.846 | -26.8 |
| massive-en-US | 0.291 | 0.814 | -52.3 |
| aegis2 | 0.312 | 0.832 | -52.0 |

The two collapses are separate defects, both isolated by follow-up probes rather than inferred from the table.

### Noul does not condition on the question's polarity

aegis2's 0.312 is below *every* constant predictor (always-safe scores 0.388, always-unsafe 0.612), and confidence runs backwards: accuracy 0.574 in the [0.5,0.6) bin against 0.034 in [0.9,1.0). Asking the same 120 states both ways through one request:

```text
    "Is this prompt unsafe?"  accuracy 0.308
    "Is this prompt safe?"    accuracy 0.775
```

Accuracy alone cannot explain that. The two sum to 1.083, and a model that ignores the instruction entirely -- emitting one state-dependent score into both slots -- sums to exactly 1.000 by construction, since each item contributes one correct answer between the two framings. Only the per-item probabilities separate "reads polarity but inverted" from "ignores the question":

```text
    mean(p_unsafe + p_safe)   0.713    1.00 would mean complementary
    mean|p_unsafe - p_safe|   0.079    0.00 means the question is ignored
    89/120 items differ by <0.10;  23/120 sum to within 0.10 of 1.0
```

The pairs are near-identical rather than complementary. The model returns roughly the same number whichever way it is asked, so it is not reading the polarity and inverting it -- on this subset it is not conditioning on `instructions` at all.

The underlying signal is real: that state-only score tracks actual safety well enough to reach 0.775 when it happens to be read as P(safe), which is why the positive framing looks competent. What is missing is that the question does not modulate it. Noul supervision came only from imdb and rotten_tomatoes, where yes = positive = good, so the model learned a benignness prior over states rather than a function of the question.

This is not a blanket failure of Noul: boolq (0.803 against a 0.603 base rate) and paws (0.712 against 0.520) both beat their base rates, so the question does carry on subsets whose yes-axis is goodness-aligned or neutral. The fix is the same in either case -- Noul needs training questions whose "yes" denotes the undesirable outcome.

### massive-en-US never reached Mode B

The 0.291 was read as a Mode B result -- 60 options is over the 26 single-letter codes -- and two follow-up probes were reported as Mode B transfer behaviour. Both readings were wrong, and the correction matters more than the number.

The router is tokenizer-verified, not count-based, and Qwen3.5's 248k vocabulary encodes every two-letter code up to `BP` as one token:

```text
    n=60   single-token codes 60/60  -> Mode A
    n=77   single-token codes 76/77  -> Mode B   (`BQ` is the first that splits)
```

The live server confirms it: a 60-option Choice cost 839 input tokens (the options were listed in the prompt, which only Mode A does) and a 77-option one cost 28 (Mode B lists nothing). So massive ran in Mode A, with two-letter codes the model had never seen, over a candidate set four times larger than any it had trained on in that mode (dbpedia_14, 14 options). The Mode B head sat idle.

That also explains the order sensitivity. Same 60 items, same options, two orders: argmax agreement 0.15, mean L1 between the distributions 1.23. Identical order, same request: L1 0.0000. A GPU diagnostic (`modal run modal/app.py::diagnose_candidates`) rules out the candidate encoder -- a string's representation is bit-identical across batch orders, `max_abs 0.0`. What is left is the one mechanism that *is* order-dependent by construction: letter-position bias in Mode A, which reflex measured on the same backbone and cancels by reading each question in two option orders.

Consequently the "60 options 0.375 / 30 options 0.592" probes measured Mode A with lettered codes, not the candidate-path head, and **Mode B on an unseen taxonomy is untested**. banking77 and clinc_oos were both training sources, so 0.87 there was held-out rows, not held-out labels.

Two fixes follow, neither needing retraining (ADR-020): a policy cap of 26 on Mode A applied identically in training and serving, so anything above single letters goes to the head that was trained for large sets; and two-order averaging for Choice and binary Noul.

**Redeployed with the cap, same checkpoint, same files** (`/health` reports `max_label_options: 26`): massive-en-US now costs 41 input tokens per item instead of 664 -- it is in Mode B -- and scores **0.166**. That is the first real Mode B transfer number: ten times chance (1/60), and well calibrated about its own ignorance (ECE 0.076; the 142 items it placed in the 0-0.1 bin score 0.070), but far below Mode A's 0.291 and Jev's 0.814. The head learned some general matching and mostly its two training taxonomies. The option key format is not the cause: snake_case keys and humanised keys score identically, 0.140 on the same 150 items. More taxonomies in the mixture is the fix, and it needs a retrain.

Two-order averaging moved vitaminc 0.578 -> 0.588 and helpsteer2 0.240 -> 0.216, both inside sampling noise -- but helpsteer2 is a Score, and reversing an ordered scale shows the model a prompt training never produces. Averaging is now limited to Choice and binary Noul. aegis2, boolq and paws are unchanged to three decimals, as expected: nothing in the rating-scale Noul path changed.

### Caveat

Calibration was fitted on lev's own mixture, and all six subsets are out-of-distribution for it, so the ECE figures from this run are not comparable to the 0.0529 measured on the held-out split.

---

## 13. Runtime: the serving path is correct, and where the time actually goes

Measured 2026-09-23 against the deployed `4b/step-18750`, with the engine profiled inside its container and the same requests timed from the client.

### The predictions are right; the gap is generalisation

The hypothesis was that an inference-side error -- a prompt or readout mismatch between training and serving -- was costing accuracy on top of what the mixture failed to teach. The test: in-distribution test-split rows, scored through the deployed HTTP path, against the in-container `evaluate` numbers (ADR-018), which go through the training collator. n=25 per source.

| source | HTTP | evaluate | | source | HTTP | evaluate |
| --- | --- | --- | --- | --- | --- | --- |
| banking77 (Mode B) | 0.920 | 0.870 | | imdb | 0.960 | 0.975 |
| clinc_oos (Mode B) | 0.880 | 0.865 | | rotten_tomatoes | 1.000 | 0.915 |
| sst5 | 0.760 | 0.545 | | yelp_review_full | 0.680 | 0.680 |
| ag_news / emotion / dbpedia_14 | 0.92 / 1.00 / 1.00 | >= 0.865 | | | | |

Every source lands at or above its `evaluate` figure, within the noise n=25 allows. Both readouts, both modes, the calibration, the option cap and two-order averaging all reach the model intact over HTTP. There is no serving bug to find: the S1Bench numbers are what the model knows. On sources the old checkpoint never trained on, the same run shows the shape of that: snli 0.44, race 0.80, toxic_chat 1.00.

### Where a call's 1.2 s goes

Client side, fresh connection, from this machine:

```text
    dns 0.003   tcp connect 0.28   tls done 0.58   first byte 0.91-1.27   (health, no model)
    real 1-Noul call: first byte 1.17-1.27
```

A 280 ms TCP round trip means the container is a continent away; TLS is two more of them. With the SDK's persistent connection the benchmark still saw ~0.9-1.2 s per call, so roughly 0.3-0.4 s of Modal ingress remains per request on top of the compute.

Inside the container, CUDA-synchronised medians over 20 calls:

```text
    fork (prefill + cache fork + suffix forward)   159-169 ms   flat: 1 noul = 8 nouls = 60-option Mode B
      of which cache fork (deepcopy)                  4.5 ms
      render + tokenise                               0.2-0.9 ms
    single batched forward over prefix+suffix       73-84 ms   also flat
```

Flat in the number of questions is the diagnosis: the forward is launch-bound -- ~200 kernel launches per pass through 32 layers, twice per call -- not FLOP-bound. The prefill-and-fork design saves prefix FLOPs that cost nothing to recompute and pays for them with a second full forward. One batched forward halves the compute at every shape measured, so it is now the default (ADR-023); the fork stays available for long states with many questions.

Two further measurements. `causal_conv1d` was never installed -- every training and serving log said so -- and the image now builds it from a CUDA 13.0 devel base (no wheel exists for `torch 2.14.0+cu130`): fork 159 → 143 ms, single 73 → 65 ms, about 10% each, which confirms the launch-bound diagnosis from a second angle. `torch.compile(mode="reduce-overhead", dynamic=True)` with shape buckets was then tried on the same container: 103–137 ms against 86–94 ms eager, a 104 s warmup, dynamo's recompile limit hit on the linear-attention layers, and a CUDA-graphs buffer overwrite on the Mode B shape. Rejected (ADR-023). decider's 8 ms comes from a static-shape engine built for its model; that is the remaining order of magnitude, and it is not a flag.

### After the fixes, from the client

Deployed with the single forward and the conv kernel (`prefix_mode: single`, `compiled: false` in `/health`), timed from the same laptop over the SDK's persistent connection -- the path the benchmark uses:

```text
    1 noul                  p50 464 ms   min 399 ms
    8 nouls, shared state   p50 457 ms   min 416 ms
    60-option Mode B        p50 489 ms   min 447 ms
```

Against 930-1,270 ms p50 in the S1Bench runs on the old path: about 2x end to end, with the compute share now ~75 ms of it and the rest the route. The three shapes still cost the same -- eight questions remain free.

The same subset through the benchmark, paws (250 items), same checkpoint:

```text
    old path, sequential          p50 928 ms   ~232 s wall
    new path, --concurrency 1     p50 417 ms    112 s wall   2.23 items/s
    new path, --concurrency 4     p50 604 ms     39 s wall   6.36 items/s
```

accuracy 0.712, log-loss 0.7535, ECE 0.1787 in all three -- identical to four decimals, which is the point: nothing here touched a probability. Under concurrency the per-call latency rises because forwards are serialised on the one GPU; wall time falls 2.85x because everything else overlaps. The full 1,999-item S1Bench pass now takes about five minutes instead of thirty.

### Why laya and reflex look so much faster

Two of the three reasons are not engineering. laya is a 421M-parameter *encoder* -- ten times fewer parameters than this model, one forward for every question, options scored at mask tokens -- and it runs in-process on the machine that reports the number, as does reflex (4B, 138 ms on a CPU) and every local target on the S1Bench board. Their latency contains no network. Ours is 86% network on the benchmark's path. Like for like, our in-container compute (73 ms single-forward, before the conv kernel) is already under reflex's 138 ms on the same class of model; what remains is the ingress route and the engineering the fast decoders did.

---

## 14. The instruct run, and why its clinc_oos number is a bug and not a result

`4b-instruct` (ADR-020 mixture, 18,750 steps) on the held-out split, calibrated: weighted 0.706 ±0.005, ECE 0.2135 → 0.1133. Not comparable to the first run's 0.856 -- the mixture is 23 sources and deliberately harder -- and the per-source picture is what matters:

| learned (new sources) | | regressed | |
| --- | --- | --- | --- |
| sciq 0.975, arc_easy 0.927, openbookqa 0.917 | QA with per-row options | clinc_oos **0.055** (was 0.865) | Mode B, 151 options |
| snli 0.852, anli 0.758 | NLI | emotion **0.605** (was ≥0.865) | 6 options |
| toxic_chat 0.976, toxigen 0.871, beavertails 0.818 | safety, both polarities | | |
| mrpc 0.896, qqp 0.864 | paraphrase | | |
| banking77 0.906 (was 0.870) | Mode B, 77 options | | |

The two regressions are the sources with the most option shuffling and the highest rate of identical option *sets* across rows. Checking the training rows against the source datasets: 35–45% of shuffled Choice rows had `target` pointing at a wrong option *as read by the training loop*, while the same rows on disk were 100% correct. The reader's question cache was keyed on a sorted payload and merged differently-ordered questions (ADR-024). Reading the same file with the fixed reader: 646/646, 682/682, 116/116 correct.

So the instruct run is a mixed measurement: everything Noul, Score and per-row-QA learned from correct labels and those numbers stand; every shuffled Choice row of a fixed-option-set source trained on a corrupted label, and clinc_oos and emotion are the visible damage. Calibration came out under-confident on Choice (T=0.74 / 0.80) -- plausibly the same cause, a model that learned to hedge because its labels disagreed with its inputs. The run has to be repeated on the fixed reader before its S1Bench number means anything; the mixture itself does not need rebuilding.

**Retrained on the fixed reader** (same mixture, same preset, 18,750 steps): loss 5.06 → 0.23 overall; over the last 300 steps Mode A averaged 0.495 and Mode B **0.531** -- against ~2.3 for Mode B in the corrupted run and 0.81 in the first run on the narrow mixture. The head learned the moment its labels stopped disagreeing with its inputs.

Held-out, calibrated: **weighted 0.836 ±0.004, ECE 0.1273 → 0.0459** -- on a 23-source mixture, against 0.856 / 0.0529 for the first model on nine. The two regressions the bug produced are gone: clinc_oos **0.976** (corrupted 0.055; first model 0.865), emotion 0.860 (corrupted 0.605). banking77 0.922, snips 0.976, dbpedia 0.998; the new families hold -- snli 0.896, anli 0.845, arc_easy 0.946, sciq 0.973, openbookqa 0.922, race 0.823, toxic_chat 0.973, toxigen 0.873, mrpc 0.888, qqp 0.873, beavertails 0.806. Score stays the weak primitive (sst5 0.581, yelp 0.666, ultrafeedback 0.547). The fitted temperatures are ordinary again -- choice:A 2.04, choice:B 1.34, noul 2.42, score 2.91 -- so the under-confidence of the corrupted run was the bug as well. The S1Bench comparison for this checkpoint follows in §15.

---

## 15. Second head-to-head: the retrained instruct model on S1Bench

Same 1,999 task files, same harness, run 2026-09-23 against the deployed `4b-instruct/step-18750` at concurrency 4 (321 s wall for the whole pass).

| subset | first 4b | **instruct** | Δ | frozen 4B | Jev | ECE instruct | ECE Jev |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aegis2 | 0.312 | **0.832** | +52.0 | 0.82 | 0.832 | 0.093 | 0.031 |
| helpsteer2 | 0.216 | **0.380** | +16.4 | 0.33 | 0.304 | 0.150 | 0.293 |
| vitaminc-dev | 0.588 | 0.751 | +16.3 | 0.75 | 0.846 | 0.129 | 0.069 |
| boolq | 0.803 | 0.863 | +6.0 | 0.82 | 0.910 | 0.050 | 0.028 |
| massive-en-US | 0.166 | 0.231 | +6.5 | 0.83 | 0.814 | 0.227 | 0.082 |
| paws | 0.712 | 0.612 | −10.0 | 0.77 | 0.820 | 0.349 | 0.038 |
| **macro** | 0.489 | **0.612** | +12.3 | 0.719 | 0.754 | | |
| massive-en-US, serving cap at the tokenizer limit (ADR-025) | 0.166 | **0.746** | +58.0 | 0.83 | 0.814 | 0.201 | 0.082 |
| **macro with that policy** | 0.489 | **0.697** | +20.8 | 0.719 | 0.754 | | |

Each of the three mixture changes built for a specific failure can be read off its subset. Negation-trained Noul: aegis2 from below-chance to **equal to Jev**, and the polarity probe of §12 no longer applies -- the model reads the question. A helpfulness rubric in training: helpsteer2 **above Jev** and above the frozen model, on the subset nobody handles. NLI and yes/no QA in training: vitaminc up 16 points to the frozen model's level, boolq past it.

Two subsets carry the remaining gap to the frozen backbone, and they are different kinds of problem.

**massive-en-US is a serving policy, not a training result.** The frozen instruct backbone scores 0.83 on it -- in Mode A, with 60 lettered options, which its tokenizer expresses in single tokens. Our cap of 26 (ADR-020) sends the same question to the Mode B head, which is now excellent on the taxonomies it trained on (clinc_oos 0.976, banking77 0.922) and still weak on one it has not seen. The cap was set when the Base model collapsed above 14 options; the instruct model evidently does not. Measured, one redeploy later, same items: **cap 26 → 0.231; Mode A up to the tokenizer limit → 0.746.** The serving default is now the tokenizer limit (ADR-025), the macro is **0.697**, and the 8 points still short of the frozen backbone's 0.83 mark the regime Mode A was under-trained in, not one it cannot do.

**paws regressed, and confidently (ECE 0.35).** The mixture gained paraphrase supervision from mrpc and qqp, and the model learned it (held-out 0.888 / 0.873) -- but those corpora reward lexical overlap, and PAWS is built from word-swapped pairs with *high* overlap and label "not a paraphrase". The model learned the shortcut the benchmark was designed to punish. Adversarial negatives (swapped-word pairs from the existing positives) are the training fix; PAWS itself and PAWS-X are blocked.

### The frozen backbone through our engine (Q11)

`Qwen/Qwen3.5-4B` with no adapter -- binary Noul, Mode A to the tokenizer limit, two-order averaging, no calibration -- on the same files:

| subset | frozen, our prompts | frozen, reflex | LoRA | LoRA − frozen |
| --- | --- | --- | --- | --- |
| vitaminc-dev | 0.715 | 0.75 | 0.751 | +3.6 |
| massive-en-US | 0.657 | 0.83 | 0.746 | +8.9 |
| boolq | 0.843 | 0.82 | 0.863 | +2.0 |
| aegis2 | 0.708 | 0.82 | 0.832 | +12.4 |
| paws | 0.752 | 0.77 | 0.612 | −14.0 |
| helpsteer2 | 0.244 | 0.33 | 0.380 | +13.6 |
| **macro** | **0.653** | **0.719** | **0.697** | |

Two things this separates. The LoRA adds 2–14 points on five subsets and costs 14 on paws, which is the shortcut §15 describes and ADR-026 targets. And the same frozen weights score 0.653 through our prompts against 0.719 through reflex's -- 6.6 points, 17 on massive alone, that are prompt and readout design rather than weights.

Most of that is the dressing. Rendering the same request as the ChatML turns the instruct model was trained on -- system prompt, `# Evidence` / `# Criterion` / `# Options`, `A. option` lines, "respond with only the letter", an empty think block opening the assistant turn -- and reading the bare letter that follows:

| subset | frozen, plain | frozen, chat | Δ | ECE plain → chat |
| --- | --- | --- | --- | --- |
| vitaminc-dev | 0.715 | 0.733 | +1.8 | 0.444 → 0.164 |
| massive-en-US | 0.657 | 0.734 | +7.7 | 0.494 → 0.116 |
| boolq | 0.843 | 0.860 | +1.7 | 0.058 → 0.051 |
| aegis2 | 0.708 | 0.776 | +6.8 | 0.045 → 0.064 |
| paws | 0.752 | 0.756 | +0.4 | 0.026 → 0.057 |
| helpsteer2 | 0.244 | 0.400 | +15.6 | 0.152 → 0.155 |
| **macro** | **0.653** | **0.710** | +5.7 | |

Same weights, same items. The frozen backbone in its own format already beats the trained plain-style LoRA (0.697), and its zero-shot calibration on the two hardest subsets improves fourfold. The instruct preset trains in this style from here (ADR-027); prompts are rendered at training time, so the mixture did not change.

Calibration: Jev is better on five of six; we are better on helpsteer2, where Jev is confidently wrong at 0.29. The temperatures were fitted on the training mixture's held-out split and these subsets are out of distribution for it; fitting on a held-out *task family* remains open (ADR-020).

---

## 16. Third run: the four gaps, in the backbone's chat format

ADR-026 mixture (29 sources: FEVER, word-swapped paraphrase negatives, parade, yes/no recasts, StrategyQA, large lettered sets in Mode A) trained in the `chat` prompt style (ADR-027), resumed once from step 6,000 after the Mode A logits OOM (fixed by projecting only answer positions). Same 1,999 S1Bench items, same harness, 314 s wall at concurrency 4.

| subset | run 2 | **run 3** | Δ | frozen (chat) | Jev | ECE run 3 | ECE Jev |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aegis2 | 0.832 | **0.864** | +3.2 | 0.776 | 0.832 | 0.080 | 0.031 |
| massive-en-US | 0.746 | **0.791** | +4.5 | 0.734 | 0.814 | **0.050** | 0.082 |
| boolq | 0.863 | 0.880 | +1.7 | 0.860 | 0.910 | 0.083 | 0.028 |
| paws | 0.612 | 0.716 | +10.4 | 0.756 | 0.820 | 0.235 | 0.038 |
| vitaminc-dev | 0.751 | 0.738 | −1.3 | 0.733 | 0.846 | 0.246 | 0.069 |
| helpsteer2 | 0.380 | 0.360 | −2.0 | 0.400 | 0.304 | 0.155 | 0.293 |
| **macro** | 0.697 | **0.725** | +2.8 | 0.710 | 0.754 | | |

Past the frozen backbone and past reflex (0.719) for the first time; above Jev on aegis2 and helpsteer2, better calibrated than Jev on massive (the option-count bands: `choice:A:large` T=1.66 fitted on 1,515 rows). Held-out weighted 0.807, ECE 0.180 → 0.061, on a harder 29-source split.

> **The S1Bench ECE figures in this section are not a valid lev-vs-Jev comparison.** They were measured before levbench binned ECE on each answer's top probability: lev's rows binned on its Gini `confidence`, Jev's on its chance-corrected maximum, two different statistics. The held-out ECE (0.061) comes from `lev.train.evaluate` and is unaffected. The S1Bench comparison has to be re-run to be quoted.

What worked, by target: word-swapped negatives took paws from 0.612 to 0.716 (the shortcut is mostly unlearned, not gone -- still 4 under frozen, and confidently wrong at ECE 0.235); large lettered sets took massive to within 2.3 of Jev; the yes/no recasts moved boolq +1.7.

What did not: **vitaminc did not move** despite FEVER learning well held-out (nli_fever 0.872). VitaminC is contrastive -- pairs of near-identical evidence revisions with opposite verdicts -- and FEVER's evidence never differs by one number or date. That is the remaining 10.8 points, and a data gap of a specific kind: minimal-edit evidence pairs.

And one held-out regression the S1Bench subsets do not show: **banking77 0.922 → 0.765** in Mode B (clinc_oos 0.976 → 0.939). With half the large-taxonomy rows cut into Mode A, the head saw fewer full sets and its loss ended at 1.16 against 0.53. banking77's 77 options sit past this tokenizer's single-token limit of 68 (the 69th code, `BQ`, splits), so it is served in Mode B; a code scheme that skips split codes routes it to Mode A. Measured: 0.818 → 0.980 on 400 held-out rows (ADR-028).

### After ADR-028: skipped label codes and transfer-selected calibration

Same checkpoint, redeployed. Serving skips split label codes, so 77- and 151-option questions read in Mode A: banking77 0.818 → **0.980** and clinc_oos 0.953 → 0.968 on 400 held-out rows each. Calibration chosen by leave-one-family-out ECE moved the temperatures little and S1Bench mean ECE from 0.1415 to 0.1359 (vitaminc 0.246 → 0.207); S1Bench accuracy is unchanged (no subset exceeds 68 options). Run sequentially from the same laptop as the Jev run, per-call p50 is 414–463 ms against Jev's 344–357 ms.

The logs for this run are in `docs/charts/logs/pre-nimble/`.

---

## 17. All 13 S1Bench subsets, on the pinned items

Run 2026-09-24. S1Bench's items come from Bespoke Labs' Nimble public-benchmark manifests (commit `62076b4`): per subset, the upstream dataset, the exact ids, the label counts, a SHA-256 of the records, and the one question every record asks. `lev s1bench export` rebuilds each subset from its Hugging Face source, keeps exactly the pinned ids and refuses the export if an id is missing or the label counts differ. The rebuilt records match the manifest SHA-256 on 12 of 13 subsets; multinli differs only in reference metadata the model never sees (Nimble reads annotator votes from the original NYU zip). Record counts match the board on 12 of 13; helpsteer2 has 249 against the board's 250. 3,880 items. The vendored definitions are in `packages/lev/src/lev/data/s1bench_subsets/`.

Both backends were run sequentially from the same laptop, through the same task files: Jev on the hosted API (served `jev-1.13.0`), lev on the deployed `4b-instruct/step-18750` (chat prompts, skipped label codes, transfer-selected calibration). ECE is binned on each answer's top probability for both, so this is the first valid lev-vs-Jev calibration comparison on S1Bench.

**The harness reproduces Jev.** On every subset our Jev run lands within 0.8 points of TypeSafe's published figure; its 13-subset macro is 0.761 against the published 0.760. Against the board's own measurements, five of the six measured subsets agree within 0.7 points. The exception is aegis2: 0.804 here, exactly the published figure, against the board's 0.836 -- the board is the outlier there, already +3.2 against published in §9.

| subset | n | lev | Jev | Jev published | majority label | ECE lev | ECE Jev |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vitaminc-dev | 599 | 0.668 | **0.801** | 0.801 | 0.503 | 0.141 | 0.099 |
| massive-en-US | 350 | 0.857 | **0.874** | 0.874 | 0.163 | **0.056** | 0.071 |
| massive-de-DE | 350 | 0.823 | **0.871** | 0.869 | 0.163 | 0.067 | 0.059 |
| boolq | 300 | 0.827 | **0.893** | 0.897 | 0.580 | 0.126 | 0.035 |
| squad2 | 299 | 0.813 | **0.836** | 0.829 | 0.502 | 0.103 | 0.036 |
| paws | 250 | 0.776 | **0.900** | 0.892 | 0.516 | 0.178 | 0.028 |
| multinli | 299 | **0.890** | 0.836 | 0.829 | 0.361 | **0.037** | 0.057 |
| civil_comments | 300 | 0.760 | **0.803** | 0.810 | 0.893 | 0.118 | 0.044 |
| aegis2 | 250 | 0.800 | **0.804** | 0.804 | 0.568 | 0.134 | 0.050 |
| helpsteer2 | 249 | **0.386** | 0.341 | 0.341 | 0.422 | **0.104** | 0.258 |
| summeval-relevance | 240 | 0.358 | 0.358 | 0.350 | 0.458 | **0.136** | 0.230 |
| summeval-consistency | 144 | 0.271 | **0.812** | 0.812 | 0.840 | 0.174 | 0.086 |
| pubmedqa | 250 | 0.732 | **0.764** | 0.772 | 0.532 | **0.121** | 0.136 |
| **macro, all 13** | | 0.689 | **0.761** | 0.760 | | 0.115 | 0.091 |
| **macro, board's six** | | 0.719 | **0.769** | 0.768 | | 0.123 | 0.090 |

"Majority label" is the accuracy of always answering the most common label. A per-subset difference needs roughly 5–9 points to clear sampling noise (95%).

**Where lev stands on the board.** On the six subsets every board model completed, lev's 0.7190 is level with reflex-4b's 0.7189 (the same Qwen3.5-4B backbone), behind Jev (0.775 on the board) and three open models of 26B–35B. Board models were scored by S1Bench's harness and lev by ours, on the same items; our Jev run is 0.6 points under the board's Jev on these six, all of it aegis2.

**What the table says.**

- **Clear losses:** paws (−12.4) and vitaminc (−13.3), the minimal-edit pairs of §16, and summeval-consistency (−54.1). boolq (−6.6) is just outside the band.
- **Leads at the edge of noise:** multinli (+5.4) and helpsteer2 (+4.5).
- **Four subsets beat both models with a constant.** civil_comments is 89% non-toxic, and on the three 5-level rating subsets the most common level wins too. Neither backend has learned these rating scales better than their base rates.
- **Calibration:** lev is better calibrated on 5 of 13 (massive-en-US, multinli, helpsteer2, summeval-relevance, pubmedqa), worse on the rest, and worst where it is also least accurate (paws 0.178, consistency 0.174).
- **The earlier definitions flattered and hid different things.** On the old loaders lev led aegis2 by 3.2; on the pinned items it trails by 0.4. The 18-scenario massive is much easier for lev (0.857) than the 60-intent version (0.791).

**summeval-consistency is a hedging failure.** 121 of the 144 summaries carry the top rating, "every statement is supported". Probed on the deployed endpoint, lev answered level 4 on 34 of them, level 3 on 66 and level 2 on 17; its mean probability on level 4 was 0.236 against 0.338 on level 3. Jev's 0.812 is close to the 0.840 of always answering 4.

The serving path is not the cause and training is. Run on Modal over the same 144 items, the release gives identical predictions with raw softmax and without order averaging (0.271 each; temperature cannot move an argmax, and the probabilities with and without order averaging were identical). The frozen Qwen3.5-4B backbone, through the same engine and chat prompts, answers level 4 on 138 of 144 and scores **0.826**. That is below the 0.840 of always answering 4: like Jev, it is near-constant rather than discriminating. Fine-tuning taught lev to rate one level lower: a regression of 55 points on this subset, and a data problem to look for in the mixture's rating sources rather than a readout to fix.

**Latency is not comparable across these two runs.** lev's per-call p50 was 600–654 ms against 414–463 ms in §16, on the same checkpoint and configuration; Jev's was 335–346 ms, unchanged. A bare `GET /health` from the laptop took 0.42 s median, and Modal's request log records about 108 ms of execution even for that handler, which does no work. Twenty warm sequential aegis2 calls afterwards (states of about 470 tokens): 589 ms at the laptop, 381 ms of request duration and 287 ms of execution in Modal's log. The 69 ms compute figure (ADR-023) is for `profile_engine`'s short reference state, measured inside the container; it is not the per-call cost on S1Bench's longer states. Why the end-to-end path moved by ~180 ms between runs is open.

Logs: `docs/charts/logs/{jev,lev}-s1bench-2026-09-24.txt`. Comparison charts: `docs/charts/lev-vs-jev.html`, rendered by `docs/charts/build.py` from those logs.
