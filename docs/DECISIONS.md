# Decision log

One record per choice that would be expensive to reverse. Each states what was decided, what the alternatives were, and **what evidence settled it** — so a future reader can reopen a decision when the evidence changes, rather than guessing at intent.

Status key: **Accepted** · **Superseded** · **Open**

---

## ADR-001 — Reproduce the abstraction, not the model

**Accepted.**

Jev's weights, training data and RLCD details are not public. Attempting to reproduce *Jev* is unfalsifiable; reproducing the **computational abstraction** — `f(state, question) → calibrated typed distribution`, with the state understood once and reused across many questions — is a concrete, testable goal.

**Evidence:** [`FINDINGS.md`](FINDINGS.md) §10 — the public API surface is fully specified and wire-compatible reimplementation is routine; the model is not.

---

## ADR-002 — Wire compatibility with `/v1/systemone` is non-negotiable

**Accepted.**

Our server speaks TypeSafe's exact request/response schema.

**Why it matters more than it looks:** it makes the benchmark honest. `levbench` measures us and Jev through *the same code path*, with one changed flag. Any divergence in our favour would otherwise be unfalsifiable. It also lets anyone swap us in behind an existing `typesafe-sdk` client with a `base_url` change.

**One deliberate divergence:** our `NoulAnswer` adds `probabilities` and `confidence`. Jev's Noul is a bare float, which makes it the one question type you cannot calibrate from a response. The `noul` field is unchanged, so existing clients are unaffected. See ADR-007.

---

## ADR-003 — Backbone: `Qwen/Qwen3.5-4B-Base`

**Accepted.** This is the decision everything else rests on.

### Why not Qwen3.8, which is newer?

Asked directly, and the answer is not preference — **Qwen3.8 has no 4B**. Enumerating the family on the Hub returns exactly:

```text
Qwen3.8-2.4T-A95B     Qwen3.8-27B     Qwen3.8-Flash-Next    (+ FP8 variants)
```

Three further facts decide it:

1. **No `-Base` checkpoints exist in the 3.8 family at all** — only instruct-tuned. For a logit-readout task you want the base model; decider and Nimble both used `-Base`. An instruct tune has already been shaped toward generating text, which is precisely the behaviour we are bypassing.
2. **The real 3.8 option is the 27B**, and on one H100 it is LoRA-only at 54 GB of weights, leaving 25.7 GB of headroom, at **~108 h per 3-epoch run**. That is 4.5 days for a single run — it eliminates the ablation budget entirely.
3. **Qwen3.5-4B is architecturally the same thing one tier down.** Verified from its `config.json`, not assumed:

```
32 layers = 24 linear_attention + 8 full_attention   (full_attention_interval 4)
hidden 2560 · head_dim 256 · 16 heads / 4 KV (GQA) · vocab 248320 · tied embeddings
max_position_embeddings 262144 · image_token_id 248056 (natively multimodal)
```

### What that config buys, for free

| Property | Consequence |
| --- | --- |
| Only 8 of 32 layers hold K/V | decider's persistent prefix cache **without pretraining a hybrid** — the other 24 carry small conv/recurrent state |
| 262 k context | The 512-token trap that caps laya and open-jev-deberta cannot occur |
| Native image/video tokens | Multimodal states at no extra cost |
| GQA 16/4 | Small per-state K/V footprint — and the state cache is the thing we fork N ways |

`Qwen3.5-2B-Base` is **24 layers = 18 linear + 6 full**, exactly the "6 full-attention layers, 18 delta-net layers" decider describes. We are one tier up on a validated choice.

### Why 4B and not 2B or 9B

Measured, on S1Bench's completed runs: reflex-4b (Qwen3.5-4B) **0.7189**; decider-2b (full fine-tune) **0.7033**. ~1.6 pp for double the parameters — and 4B still fits comfortably under LoRA. 9B doubles memory and time for accuracy we do not need, because **our target is calibration, not the accuracy crown** (ADR-006).

### Feasibility table, one H100 80 GB, bf16

| backbone | weights | full-FT state | LoRA state | headroom | h / 3 epochs | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-2B-Base | 4 GB | 32 GB | 4.3 GB | 75.7 GB | 8 | fallback |
| **Qwen3.5-4B-Base** | **8 GB** | 64 GB | **8.3 GB** | **71.7 GB** | **16** | **chosen** |
| Qwen3.5-9B-Base | 18 GB | 144 GB | 18.3 GB | 61.7 GB | 36 | if 4B underfits |
| Qwen3.8-27B | 54 GB | 432 GB | 54.3 GB | 25.7 GB | 108 | no Base ckpt, no ablations |

**Reopen this if:** a Qwen3.8 `-Base` appears at ≤9B, or you get more than one H100.

---

## ADR-004 — LoRA, not a full fine-tune

**Accepted.**

4B full fine-tune needs ~64 GB of weights + gradients + AdamW state, leaving ~16 GB for activations — too tight at long context. LoRA r32 needs ~8.3 GB, leaving ~72 GB.

**The non-obvious part that made this safe:** Mode B's matching head is *new* parameters and trains at full precision regardless of LoRA freezing the backbone. Freezing the backbone does not prevent training a new head, which is what initially made this look like a trade-off and turns out not to be one.

**Consequence:** a full run is ~2 h once measured (see ADR-016), so a dozen is cheap. Budget the H100 for **ablations, not one heroic run**.

---

## ADR-005 — Dual-mode readout (the differentiator)

**Accepted.** This is the one genuinely novel piece.

Every implementation surveyed picks exactly one readout family:

- **A — label-token:** map options to single tokens, read logits at `Answer:`. Zero parameters, works untrained. Hard option ceiling; decider measured **−5 to −24 points** on 50–219 option sets.
- **B — candidate-path:** encode each candidate's *text*, score with a trained matching head. No ceiling. Costs a head; NanoJev shows the candidates batch into one forward ("44 candidate paths, 1 backbone forward").

**Nobody does both.** We route per question, and critically: **the boundary is tokenizer-verified single-token-ness, not an option count.** Where LitJev *rejects* a tokenizer that cannot express the codes, we fall through to Mode B. Their hard failure becomes our second mode.

**Two obligations this creates**, both treated as first-class rather than assumed away:

1. A and B produce distributions by different mechanisms, so they get **separate fitted temperatures** (ADR-006).
2. Where both are valid we **measure agreement**. Material disagreement makes the router a correctness hazard, not merely a capacity switch.

---

## ADR-006 — Calibration is the product

**Accepted.** The most important decision after the backbone.

S1Bench, completed runs only:

| | macro | ECE |
| --- | --- | --- |
| jev | 0.7751 | **0.0764** |
| simplejev-qwen38-27b | 0.7582 | 0.1214 |
| djev-full | 0.7485 | 0.1661 |
| **reflex-4b** | 0.7189 | **0.0849** |
| qwen3-8b-full (untuned) | 0.5346 | 0.4252 |

Two readings settle the project's direction:

1. **The accuracy gap is ~1 point.** Open reproductions have essentially caught Jev on accuracy. Competing there is a losing, expensive fight.
2. **reflex-4b is the only open model with both accuracy and calibration** — and it gets there *not by architecture* but by fitting **one scalar** post-hoc. Untuned: 0.4252. With a temperature: 0.0849. Jev: 0.0764.

**Calibration is a cheap bolt-on that almost none of the clones bothered with.** That is the winnable fight, and it is why `lev.calibrate` fits **per (question type, readout mode)** — one global scalar under-serves at least one bucket — and why `calibrate.fit()` **raises** on a split named test/eval/holdout.

### First measured head-to-head

Jev `jev-1.13.0` against lev on `Qwen3.5-4B-Base` (untuned, Mode A, no calibration), same 24 items, same questions, `levbench eval`:

| question | type | Jev acc | lev acc | Jev ECE | lev ECE |
| --- | --- | --- | --- | --- | --- |
| department | Choice | 0.958 | **0.958** | **0.0250** | 0.2320 |
| frustration | Score | 0.750 | 0.500 | 0.1350 | 0.4635 |
| is_urgent | Noul | 0.917 | 0.292 | 0.0804 | 0.5261 |

**Choice is a dead tie on accuracy and a 9× gap on calibration.** That is ADR-006's whole argument, reproduced on our own data at the first attempt: an untuned 4B already matches a frontier decision model at picking the right option, and loses entirely on knowing how sure it is.

Three further readings:

- **lev's Choice is *under*confident, not over.** Every reliability bin scores accuracy 1.000 while reporting 0.15–0.93 confidence — all gaps negative. Fitting a temperature gives **T = 0.409** (sharpening, not flattening). That is the easy direction to fix and further evidence for ADR-006.
- **Score is Jev's weak primitive too.** Its `frustration` log loss is **1.9484** against `ln(3) = 1.099` for a uniform guess — worse than chance despite 75% accuracy, meaning it is confidently wrong on the quarter it misses. Consistent with S1Bench, where the Score-shaped `helpsteer2` subset sat at 0.348 for everyone including Jev.
- **Measured ECE depends on which statistic the server calls `confidence`.** lev reports normalised Gini (LitJev's choice); scoring the *same* distributions by max-probability instead gives department ECE 0.0878 rather than 0.2320. So part of the gap above is a statistic mismatch rather than a worse distribution — which is exactly why open question Q1 has to be settled before this table is read too hard.

Jev also reported **1,920 output tokens** across the 24 calls (~80 per call), billed at $0. lev reported 0: it genuinely generates nothing.

---

## ADR-007 — Noul from nine rating tokens

**Accepted.**

Jev's `NoulAnswer` is a bare float: no `probabilities`, no `confidence` (verified at runtime against the real SDK, [`FINDINGS.md`](FINDINGS.md) §2). It is therefore the one question type whose calibration cannot be measured from a response.

Following simple-jev, we read Noul from a **9-level rating scale** and report `p(yes) = Σ (i/8)·p_i` alongside the full distribution. Finer resolution, and Noul becomes calibratable like Choice and Score. The `noul` field itself is unchanged.

### Measured caveat: the 9-point scale does not survive an untuned base model

First real run, `Qwen3.5-4B-Base`, Mode A, no fine-tune, no calibration, on the 24-item triage set:

| question | type | accuracy | ECE |
| --- | --- | --- | --- |
| department | Choice | **0.958** | 0.232 |
| frustration | Score | 0.500 | 0.464 |
| is_urgent | Noul | **0.292** | 0.526 |

0.292 is exactly 7/24 — the positive base rate. The model answered *yes to every item*. Probing it directly shows why: the rating distribution is pinned at the endpoints with P(8) ≈ 0.8 regardless of content (0.839 on a clearly non-urgent question, 0.797 on a clearly urgent one — flat, and slightly inverted).

**Choice works zero-shot on a base checkpoint; a 9-point rating scale does not.** A base model has no instruction-following prior for "Rate 0-8", so the digits after `Answer:` reflect token priors rather than judgment. This does not invalidate ADR-007 — the scale is still the right *trained* target, and it is the only way to make Noul calibratable — but it means the untrained baseline in the build order cannot use it. Options, in preference order:

1. Run the zero-shot baseline on an **instruct** checkpoint (`Qwen3.5-4B`, not `-Base`), keeping `-Base` for the fine-tune where it belongs.
2. Fall back to a 2-token yes/no Noul until the scale is trained.

This is the clearest empirical support so far for ADR-011's "shipping untuned is not an option" — and it identifies *which* primitive fails first.

---

## ADR-008 — Both prompt layouts, trained 50/50

**Accepted.**

- **state-first** — cache the state, fork across questions. The shared-state win.
- **schema-first** — cache the question catalogue across many states.

Random layout per example buys the choice at inference. decider's measured cost of schema-first is inherited as a **routing rule**, not a preference:

| workload | cost |
| --- | --- |
| fixed label set | −1.5 pts (median −0.7, calibration equal) |
| options vary per example | −5 pts |
| 50–219 options, or multi-thousand-token states | −5 to −24 pts |

So: schema-first only for high-volume fixed-schema batch work; state-first by default.

---

## ADR-009 — All thirteen evaluation subsets are banned from training

**Accepted.** Enforced in code, not documentation.

```
ran in s1-fast (6)     vitaminc-dev  massive-en-US  boolq  helpsteer2  aegis2  paws
intended, unrun (7)    massive-de-DE  squad2  multinli  civil_comments
                       summeval-relevance  summeval-consistency  pubmedqa
```

Three S1Bench entries self-declare contamination and their numbers are compromised. Our single differentiating claim is calibration measured on exactly these subsets, so contamination would not merely weaken the result — **it would silently improve it**, which is worse.

The block list covers all **13** subsets in the snapshot's `published_jev`, not the 6 that happened to execute in the `s1-fast` suite. The other 7 are evaluation data that simply did not run; training on them would contaminate any later full-suite comparison and we would not find out until the number was already published. Blocking only what ran is how you get a result that looks good and means nothing. The cost of the wider list is four otherwise-usable sources (squad2, multinli, civil_comments, pubmedqa) — cheap next to an uninterpretable headline.

`lev.data.contamination` resolves aliases (`tals/vitaminc`, `paws-x`, `google/boolq`, `nvidia/HelpSteer2`, `rajpurkar/squad_v2`, `nyu-mll/multi_nli`, …) and **raises**. It runs before any data loads. decider's ~95-dataset registry contains several of the thirteen and must be filtered.

---

## ADR-010 — Two packages, one workspace

**Accepted.**

`packages/levbench` (measurement) and `packages/lev` (the model) are separate distributions. The benchmark predates the model, is useful on its own against Jev and any compatible server, and **must not depend on our model** — a measuring instrument that imports the thing it measures is not an instrument.

The lev **core** (schema, prompt, router, calibration, contamination) imports without torch, so it is fully testable on a laptop; everything needing a GPU sits behind the `[train]` extra.

---

## ADR-011 — Rejected trade-offs

**Accepted.** Each refusal has evidence; see [`ARCHITECTURE.md`](ARCHITECTURE.md) §4.

| Rejected | Evidence |
| --- | --- |
| 512-token context | Structurally destroys the shared-state premise. kotoba measured DeBERTa-v3-large as not fitting 100 questions in 512 |
| Diffusion backbone | djev: accuracy 0.7485 at ECE 0.166–0.178 (2.2× Jev). kotoba clocked a dLLM at 846 ms vs 19–34 ms for ModernBERT-base |
| Pure linear/recurrent | simplejev-rwkv holds the bottom three slots (0.313–0.380). Hybrid yes, pure RWKV no |
| Shipping untuned | ECE 0.4252 completed, 0.366–0.493 stopped. Uncalibrated confidence is worthless, and confidence is the product |
| Optimising for speed | `verdict`: 101 dec/s at ECE 0.4622. Fast and confidently wrong is not a decision model |

---

## ADR-012 — Abstain means taking the state away

**Accepted.**

10% of training examples are unanswerable. The first implementation set a flag on an otherwise ordinary example and changed nothing else, which is worse than no augmentation at all: the state still determines the answer, so the only thing the model learns is to be unsure when it should not be. That is a calibration *regression* dressed as a calibration aid.

An abstain example now pairs a question with a state drawn from a **different source**, and supervises a **uniform** distribution over the candidate set. With no evidence, every candidate is equally supported, so the uniform vector is the calibrated answer rather than a hedge — and teaching it is the point.

This is why `decision_loss` takes soft targets. The hard rows are unaffected: with a one-hot target the soft cross-entropy is exactly `F.cross_entropy`, which is asserted in `test_loss.py`.

Abstain rows are training-only. They are excluded from the exported eval set, because scoring them would measure our abstention rather than our accuracy, and the two are different numbers.

---

## ADR-013 — Mode B is weighted by what must be learned, not by corpus size

**Accepted.**

Only two of the nine sources — `banking77` (77 intents) and `clinc_oos` (151) — have option sets that overflow single-token label codes, so they are the *only* Mode B training signal. Every other source trains Mode A and nothing else.

Weighting the mixture by corpus size would give the pair a few percent of the examples, and the head that is supposed to be this project's differentiator ([ADR-005](#adr-005--dual-mode-readout-the-differentiator)) would ship untrained. `default_weights()` gives them **25%**. The remaining 75% splits evenly across the three primitives, so no readout starves.

The cost is accepted knowingly: those two sources are over-represented relative to any natural distribution, which will bias Mode A's option-set prior toward short intent phrases. `test_data_pipeline.py` asserts the 25% floor so a later "tidy-up" of the weights cannot quietly undo it.

---

## ADR-014 — Split before mixing, into three splits

**Accepted.**

Three splits, not two: a temperature fitted on the test set is not a measurement. `calibrate.fit()` already refuses a split named test/eval/holdout; the `calibration` split is the other half of that guarantee — the thing it can legitimately accept.

Assignment is a **blake2b hash of a stable per-row key**, not an RNG draw. The same row lands in the same split on any machine, in any dataset order, on any re-run, which is what makes a resumed or repeated experiment comparable to the original. Python's `hash()` is salted per process and would not reproduce tomorrow.

Splitting happens **before** the mixture is drawn. Mixing first would let one underlying row appear in train and in test wearing two different layouts — a contamination leak with our own data rather than S1Bench's. Subtler than ADR-009's, and flattering in exactly the same way.

Two coverage checks, because one is not enough. Every label *observed* must appear in train, or its error rate measures nothing. And every label the *question offers* must be observed at all — the check that catches an undersampled banking77, where 3 intents out of 77 satisfy the first check and the mixture is still junk. Both raise.

---

## ADR-015 — The 24-item task set is a fixture, not a benchmark

**Accepted.**

At n=24 the 95% interval on an accuracy estimate is about ±16 points. A training run that moved accuracy by 5 points is indistinguishable from one that moved it by nothing, so the hand-labelled support set cannot answer the only question training raises.

It stays, as a smoke fixture: does the server answer, are the types right, does a distribution come back. The real eval is exported from the mixture's held-out test split — 2,760 items across nine sources, ±4 points per source — and `levbench eval --tasks` reads it.

One file per source, because each source carries exactly one question. A single combined file would make the harness ask every question of every item, i.e. ask "how positive is this review?" of a banking ticket, and score the answer.

The exporter lives in `lev`, not in `levbench`. levbench must not import the thing it measures (ADR-010), so the handoff is a JSON file and the writer sits on the model side of the fence.

---

## ADR-016 — Sequence length is measured, and the budget was wrong by 10x

**Accepted.**

`avg_tokens_per_example` was `1200`. It was a guess, made before any data existed. The measured mean over 1,500 rendered prompts from the real mixture is **121 tokens** — median 73, p95 390, longest 1,104.

| Source | Mean tokens |
| --- | --- |
| clinc_oos | 38 |
| banking77 | 39 |
| rotten_tomatoes | 65 |
| emotion | 71 |
| sst5 | 77 |
| ag_news | 98 |
| dbpedia_14 | 164 |
| yelp_review_full | 223 |
| imdb | 305 |

The 4B budget therefore read **16.0 hours** when it is closer to **1.7**. That error is not academic: it is the number the H100 is booked against, and it was wrong in the direction that makes you plan for one run when you can afford ten.

Three things changed as a result:

- The default is `128`, with the measurement and its provenance in the docstring, and `lev plan --data <dir>` re-measures against a real mixture rather than trusting it.
- `tokens_per_step` was `max_seq_len × batch`, which bills the truncation cap rather than the work. The collator pads to the longest row in the batch, so at a 4,096 cap over 121-token data that over-counted by ~16x. It is now `avg_tokens_per_example × batch`, and `steps_per_epoch` counts **examples**, because an epoch is a pass over the data and how many optimiser steps that takes is a function of the batch, not of a token budget.
- `per_device_batch` went 8 → 32. At a 121-token mean, a batch of 8 makes 75,000 optimiser steps and the run is bound by step overhead long before it is bound by FLOPs. `max_seq_len` dropped 4,096 → 2,048, which still clears the longest prompt seen with room to spare.

The general lesson, and the reason this is an ADR rather than a commit message: **every number in the budget that was not measured was wrong.** The FLOPs constant and the H100 throughput figure are still assumptions, so Q5 stays open.

---

## ADR-017 — Batches are length-bucketed, and the budget was wrong again

**Accepted.** Measured on the first real H100 run, not predicted.

The first `train --preset 4b` on Modal reported:

```
step 25/18750  0.1%  A=5.7621  B=5.1222  lr=4.45e-06  0.22 it/s  826 tok/s  eta 23:13:22  mem 29.3G
```

**23 hours against a 24-hour timeout**, and 826 tok/s where ADR-016's arithmetic implied ~12,000. Two causes, both measured afterwards:

**Padding, 4.4x.** A batch is padded to its longest row, so the model computes on the rectangle rather than on the real tokens. The mixture is bimodal — a banking intent is ~39 tokens and an imdb review reaches 1,300 — so one long row in a batch of 32 drags the whole batch up to it. Through the real collator on the real mixture:

| batching | real tokens | padded | waste |
| --- | --- | --- | --- |
| random | 380,794 | 1,686,163 | **4.43x** |
| length-bucketed | 380,794 | 542,779 | **1.43x** |

`ModeBatcher` now sorts by length inside a shuffled window of `bucket_window` (64) batches, then shuffles the batch order. Sorting globally would feed every short example before every long one and correlate length with step number; sorting within a window keeps the order effectively random. The proxy is character count, because tokenising twice would cost more than the padding it saves — the resulting buckets measure 1.43x against a 1.03x theoretical floor.

**Kernels, the rest.** 24 of Qwen3.5-4B's 32 layers are linear-attention, and without `flash-linear-attention` transformers runs `chunk_gated_delta_rule` on its reference PyTorch path — three quarters of the model on the slow route. That layer is now in the image. It had been left out on the grounds that a failed build is worse than a slow run; a run that does not fit its timeout changes that trade.

`causal-conv1d` is **not** included, and the first attempt to add it alongside proved why the two are not interchangeable. It is a CUDA *source* build that needs `nvcc`, which `debian_slim` does not ship, so the image build dies at `Getting requirements to build wheel` with `NameError: name 'bare_metal_version' is not defined` — a confusing way to say "no compiler". Including it would mean an `nvidia/cuda:*-devel` base and a far heavier image, to accelerate only the short depthwise convolution. `flash-linear-attention` carries the linear-attention core and is pure Python and Triton, so it needs no compiler. The lesson is narrower than "install the kernels": **a pure-Python wheel and a CUDA source build are different decisions**, and bundling them into one turned a correct call into a failed build.

**The trade accepted:** length-bucketed batches are more homogeneous in source, because length correlates with source here. Over an epoch the shuffled batch order distributes them evenly, and the alternative is paying 3x. Worth revisiting if the loss curves look source-periodic.

**A caveat on the 23 h figure itself.** It was a *cumulative* rate read at step 25, so it was dominated by startup — weight loading, allocator warmup, and Triton JIT. The 0.8B smoke makes the size of that effect concrete: 4.68 s/step over the first 25 steps, 0.40 s/step after, reported cumulatively as 0.33 it/s against a true 2.50. `ProgressLog` now measures rate, throughput and ETA over the window since the last report. The padding finding stands on its own — 4.43x was measured offline through the collator, not inferred from the ETA — but the 23 h number was inflated by an unknown amount and should not be quoted as a baseline.

**The general lesson, restated from ADR-016 because it recurred:** the estimate was built from a token count that was correct and a padding factor that was assumed to be 1. Both ADR-016 and this one are the same failure — an unmeasured term in the budget — and the throughput readout that exposed it only existed because the loop had been made to print (see `ProgressLog`). Q5 stays open until a full run lands.

---

## ADR-018 — What the first trained checkpoint actually shows

**Accepted.** Measured on 1,800 held-out items, `step-18750`, `modal run modal/app.py::evaluate`.

Training: 18,750 steps, ~4h50 on one H100, every loss finite. Mode A 1.754 → 0.233, Mode B 4.843 → 0.813 from an `ln(151) = 5.02` chance start.

| | uncalibrated | calibrated |
| --- | --- | --- |
| accuracy | 0.856 ±0.016 | 0.856 ±0.016 |
| ECE | 0.1162 | **0.0529** |

Four findings, in descending order of how much they change the design.

**Q6 is closed: the Noul collapse was a base-checkpoint artefact.** Untrained, the model answered "urgent" for everything — 0.292, `P(8) ≈ 0.8` regardless of content (ADR-007). Trained: imdb 0.975, rotten_tomatoes 0.915. Mapping the binary class onto the *ends* of the 0-8 rating scale rather than passing it through as a class index is what fixed it.

**Mode B works, and it is the differentiator.** banking77 0.870 over 77 options, clinc_oos 0.865 over **151** — within 6 points of Mode A's average, with no option ceiling. Every Family-A implementation surveyed in ARCHITECTURE.md §2 either caps the option count or rejects the request at this point.

**Per-bucket calibration was the right call, and now there is evidence.** `choice:B` fitted to T=1.033 — the candidate-path head came out essentially calibrated on its own — while Mode A needed 2.084 (choice), 2.981 (noul) and 4.096 (score). A single global temperature of ~2.5 would have actively damaged Mode B. That rule was argued from first principles in `calibrate.py`; it is now measured.

Two sources got *worse* ECE under calibration: dbpedia_14 0.0049 → 0.0149 and imdb 0.0244 → 0.0427. Both were already near-perfectly calibrated and were over-softened, because the bucket key is `(type, mode)` and is shared across every source in the bucket. Log loss improved on both, so the temperature is net-positive even there, but per-source calibration is the obvious refinement.

**Score is the weak spot.** sst5 0.545, yelp_review_full 0.680; everything else is ≥0.865. Both are 5-level ordered scales where adjacent levels are genuinely ambiguous. sst5 also had the worst uncalibrated ECE at 0.4336. A next run should target Score specifically.

**These numbers are not comparable to Jev's 0.7751 / 0.0764.** This is a held-out split of the same nine corpora the model trained on — in-distribution. S1Bench is thirteen different subsets, deliberately blocked from training (ADR-009) and still untouched. Reporting 0.856 against Jev's 0.7751 would be the exact error the contamination guard exists to prevent.

---

## ADR-019 — The evaluation is not bit-reproducible, so the report states its interval

**Accepted.**

Two runs of `evaluate` over the same checkpoint and the same data disagreed: yelp_review_full accuracy 0.685 → 0.680 (one item in 200), clinc_oos ECE 0.0515 → 0.0421. Accuracy on the other eight sources was identical.

The cause is GPU kernel non-determinism: bf16 reductions and the Triton linear-attention kernels do not pin their reduction order, so logits differ in the last bits and items near a decision boundary flip. clinc_oos moved most because a 151-way softmax has many near-ties sitting on bin edges and ECE is a *binned* statistic; log loss barely moved (0.4871 → 0.4873), which is the tell that nothing real changed.

`torch.use_deterministic_algorithms(True)` was rejected: it would likely refuse the linear-attention kernels and give back most of the speedup ADR-017 bought, to remove a difference that is far inside the sampling interval anyway.

Instead the report prints its own resolution — a 95% half-width on every accuracy, ±0.048 at n=200 and ±0.016 at n=1,800, with a standing note that ECE moves below ~0.01 are noise. A four-decimal table that is reproducible to two is worse than no table, because a reader takes the fourth decimal for a result.

---

## ADR-020 — The trained model lost to the frozen one; what changes and what does not

**Accepted.** Measured on 1,999 S1Bench items through identical task files (FINDINGS.md §12), plus targeted probes against the live server and one GPU diagnostic.

### What was measured

`reflex-4b` serves **frozen Qwen3.5-4B** -- no adapter, no training -- and scores 0.719 macro, ECE 0.085, 138 ms on CPU. Our LoRA fine-tune of the same backbone scored **0.489**, below the frozen model on every one of the six subsets. Its README names the failure before we hit it: every adapter they trained "won on data shaped like its training mix and lost general judgement".

Four causes, each with a probe behind it:

1. **The mixture confounded state with answer set.** Nine sources, one fixed instruction each, one question per row. The state alone identified the label set, so the model never had to read the question -- and does not: on aegis2, "is this unsafe?" and "is this safe?" return the same number (mean |Δp| = 0.079 over 120 items; 89 differ by <0.10).
2. **Base checkpoint, decider's cost without decider's data.** ADR-003 chose `-Base` on decider's precedent. decider trained it on 1.47M examples, 455M tokens/epoch, ~95 datasets, full fine-tune, teacher-written questions. We used 200k, 24M, 9, LoRA r32 -- 7x, 19x, 11x less -- so every bit of judgement had to come from sentiment and topic labels, and a model with no zero-shot floor (Noul 0.292 untrained, ADR-007) ended below one that never trained.
3. **60 options routed to Mode A.** The router is tokenizer-verified, and this tokenizer has single tokens for every two-letter code up to `BP`, so massive-en-US (60 intents) ran as Mode A with codes and a set size the model had never seen, while the trained Mode B head sat idle (839 input tokens listed vs 28 for a 77-option Mode B question). Reordering the options changed the argmax on 85% of items; a GPU diagnostic shows candidate representations are bit-identical across orders, so this is Mode A letter-position bias, not the encoder.
4. **The latency is the deployment, not the model.** `/health` alone takes 0.9 s from the client; one Noul takes 1.16 s and eight Nouls sharing a state take 1.16 s. The cache fork works; ~0.9 s is Modal ingress from this machine.

Two things were caught on the way that are bugs regardless of the above: the contamination guard matched blocked *subset* names inside longer ids but not *aliases*, so `SetFit/amazon_massive_intent_en-US` and `nvidia/Aegis-...-1.0` passed; and hwu64 -- a candidate Mode B source -- is the 64-intent schema MASSIVE inherited via SLURP, so training on it would turn massive-en-US from an unseen-taxonomy test into a seen one.

### Decisions that need no retraining

- **`LABEL_OPTION_CAP = 26`**, applied by `TrainConfig` and `EngineConfig` alike. The cap draws the Mode A / Mode B line, not the tokenizer. Serving and training must agree or the model is asked at serve time to do what it never trained for.
- **Two-order averaging** for Choice and binary Noul under the state-first layout: two suffix rows per question in the same batched forward, probabilities averaged in canonical order. reflex's correction, for reflex's measured reason. Not for Score or the rating scale: their order is the meaning, training never reorders them, and reversing Score measured -2.4 points on helpsteer2.
- **Binary Noul** (`A: yes / B: no`) as the readout when no adapter is loaded. The 0-8 scale is the trained target and the only calibratable form, but a stock checkpoint pins it (ADR-007). `LEV_SERVE_MODEL=Qwen/Qwen3.5-4B modal serve` serves the instruct checkpoint frozen this way -- the zero-shot baseline every trained run must now beat.
- **Guard:** bare aliases match as segment sets; hwu64 is blocked under `massive-en-US` for its schema.
- **Eval export** skips sources whose question varies per row rather than writing them under the first row's option set.

### Decisions that do need retraining

- **Instruction is no longer constant per source.** Every source carries paraphrases; every Noul source carries negations that flip the target, so "yes" is no longer always the good outcome. Applied to the train split only; held-out splits keep the canonical wording, so exported eval files and fitted temperatures describe what is served.
- **Options vary per row.** Choice sets are subsampled (gold kept), shuffled, and sometimes stripped of descriptions. Mode B sources keep >= 27 options so they stay Mode B; everything else may go down to two. Score levels are never reordered.
- **Fourteen new sources**, all passing `assert_clean`: QA with per-row options (race, commonsense_qa, sciq, openbookqa, arc_easy), NLI (snli, anli), paraphrase (mrpc, qqp), safety in both polarities (toxic_chat, toxigen, beavertails), a helpfulness rubric (ultrafeedback), and a small intent set (snips). cosmos_qa, social_i_qa, strategy-qa and mtop were candidates and are script-backed, which `datasets>=5` refuses.
- **`4b-instruct` preset**: `Qwen/Qwen3.5-4B` at lr 5e-5. Zero-shot judgement becomes the floor rather than zero. Reflex's forgetting result is the risk; breadth in the mixture is the first mitigation, a KL anchor to the frozen model the second if breadth is not enough.

Not done, and named so it is not mistaken for done: multi-question training rows (the product serves N questions per state; training still shows one), a KL anchor, `torch.compile`/CUDA graphs on the engine, and calibration fitted on a held-out *task family* rather than held-out rows.

**Reopen this if:** the frozen instruct baseline through our engine lands far from reflex's 0.719, which would put the gap in our prompts or engine rather than in training; or if massive-en-US under the cap -- the first real Mode B transfer number -- comes out near chance, which would mean the head memorised its two training taxonomies. *Reopened on the second point by ADR-025: the head scored 0.166 then 0.231 on massive, and the serving cap now follows the tokenizer, not the training cap.*

---

## ADR-021 — A checkpoint carries the training state, and a run resumes by default

**Accepted.** Supersedes the weights-only resume that `run_training`'s docstring defended.

The old position: restore the adapter and head only, let the optimiser and cosine schedule start over, because at ~2 h per run a repeated warmup is cheap and a half-restored optimiser is hard to reason about. Two things changed it. Runs are longer -- the 4B mixture rebuilt under ADR-020 carries long states (passages, prompt+response pairs), and the first-window ETA on the instruct run read 53 h -- and nothing ever *used* the checkpoints: `train` only resumed when handed a path, so a preemption followed by `make train` replayed the run from step 0 with 2,000-step checkpoints sitting unused on the volume. Writing state nobody reads is not durability.

What a checkpoint now carries beside the weights: optimiser moments, scheduler position, step, epoch, the batch count within the epoch, and the Python RNG state captured before the epoch's shuffle. The resumed run reproduces that epoch's order, skips the batches it had already consumed, continues the schedule from the same position and appends to `history.json`. Tested through the real loop: the resumed run processes exactly the batches the uninterrupted one would have, across an epoch boundary too.

`train` resumes from the newest checkpoint in the preset's directory unless told `--fresh`; `--resume <path>` still names one. `smoke` always starts fresh. A checkpoint without a state file -- any written before this -- restores weights only and behaves as before, so nothing already on the volume is stranded. The state file is written after the weights. Writes are not atomic: an interruption can leave incomplete weights or state, so a partial checkpoint may need to be set aside before resuming from the previous complete one.

---

## ADR-022 — A release is one flat directory, and the demo is a client

**Accepted.**

**Releases.** Weights leave the training tree as a single directory: adapter, Mode B head, tokenizer, `calibration.json`, `lev_release.json`, model card. The manifest names the base model, the step and preset, and the serving policy the weights were trained under (option cap, Noul readout). A LoRA adapter is meaningless without its base, the head without its adapter, the temperatures without their readout, and the cap without the training that assumed it; shipping them apart is how a "model" on the Hub stops matching the numbers in the paper. `lev serve` reads the manifest, so a release serves from disk or from a Hub id with no other argument, and `/health` says what loaded. The optimiser state stays behind: it is for resuming, not serving.

**The Snake demo lives in levbench, not lev.** It is a client of `/v1/systemone` and nothing else, so the same game runs against lev, Jev, or a frozen baseline through the client `levbench eval` uses, and its latency includes the network the benchmark's does. laya-mlx's rules, planner and prompt wording are kept verbatim (Apache-2.0) so runs are comparable to theirs, and its display, loop (rounds, keyboard, pacing) and recording format are laya's too, so a recording replays in either. Two additions: the planner knows the true answer to both Noul questions, so the demo scores them live — in the compact prompt the state text states the answers, which makes this the cheapest possible probe of the "ignores its question" failure that cost lev 52 points on aegis2 — and `--backend planner` plays from the planner alone, the reference row every model is measured against and a way to see the display with no server up.

**Deploy, don't serve, for anything you point a run at.** Every ephemeral app of `modal/app.py` — `modal serve` and every `modal run` — registers the same `lev-serve-dev` web label; a run steals it from a live dev server and the URL 404s when the run exits. Measured the hard way: an export run took the URL out from under a benchmark. `make deploy` owns a stable label.

**Self-hosted servers get a 120 s client timeout.** The SDK's default is 10 s; Modal's cold start for the 4B model measured 20–55 s. The first move of a demo or the first item of an eval must not fail on a container scaling from zero.

---

## ADR-023 — One batched forward, not prefill-and-fork

**Accepted.** Reverses the default of ADR-005's serving path on measurement.

The engine was built around prefill-once-fork-per-question: compute the state prefix once, fork the hybrid cache, run every question's suffix against it. The argument was FLOPs -- N questions cost one prefix instead of N. Profiled in the container on the 4B checkpoint (`modal run modal/app.py::profile_engine`), CUDA-synchronised, medians of 20:

```text
    fork    1 noul 167 ms   3 mixed 169 ms   8 nouls 159 ms   60-option Mode B 159 ms
    single  1 noul  73 ms   3 mixed  81 ms   8 nouls  84 ms   60-option Mode B  78 ms
```

Both are flat in the number of questions, which says the forward is bound by kernel launches, not arithmetic, at these sizes. The fork's saved prefix FLOPs are therefore worth nothing, and its second forward costs a full second pass. A single right-padded batch of prefix+suffix rows halves the latency at every shape measured, and the two strategies agree on every probability (checked in the same profile).

`EngineConfig.prefix_mode` defaults to `"single"`; `"fork"` remains for the regime the original design assumed -- long states with many questions, where repeating the prefix per row would dominate. That crossover has not been measured and should be before anyone relies on it. The two paths agree to a maximum probability difference of 0.0075 across every answer profiled, which is the bf16 kernel-path noise ADR-019 already documents, not a semantic gap.

With `causal_conv1d` present (the image now builds it against `torch 2.14.0+cu130`): fork 140-156 ms, single 65-75 ms -- about 10% each. That the kernel bought so little is the diagnosis confirmed from a second angle: the time is in launching ~200 kernels twice, not in any one of them. What removes launches is CUDA-graph replay, so the engine gains `EngineConfig.compile` (`torch.compile(mode="reduce-overhead", dynamic=True)`) with inputs padded to shape buckets -- width to 32 tokens, rows to a power of two -- so the set of recorded graphs stays small, and a `warmup()` the server runs at startup so no request pays a capture. Padding changes no real row's logits: right padding is masked and rows are independent. The GPU forward is taken under a lock, because graph replay is not thread-safe and the server now accepts concurrent requests; parsing, tokenising and the network overlap around it.

**Compile: measured, rejected.** On the same container, eager single-forward ran 86–94 ms and the compiled path 103–137 ms, after a 104 s warmup; dynamo hit its recompile limit (8) on the hybrid's linear-attention layers, so much of the forward stayed eager with extra guard overhead, and the Mode B shape then failed with a CUDA-graphs output-buffer overwrite when hidden states from one replay were read after the next. The flag stays (`LEV_SERVE_COMPILE=1`) so the measurement can be repeated on a future transformers/FLA release, and defaults off. Getting decider's 8 ms would mean a static-shape engine built for this model, not `torch.compile` over the stock one.

Also decided here, from the same session's measurements (FINDINGS.md §13): the image builds `causal_conv1d` from a CUDA devel base, since no wheel exists for `torch 2.14.0+cu130` and every log said the reference path was in use; and the serving path was verified correct end to end -- in-distribution accuracy over HTTP matches the collator-path `evaluate` on every source -- so no inference-side error is hiding behind the S1Bench gap.

---

## ADR-024 — The mixture reader merged shuffled questions, and the instruct run trained on it

**Accepted.** A bug report as much as a decision; recorded because it decides what the `4b-instruct` numbers mean.

`read_jsonl` validates each row's question once per distinct payload and shares the object, to avoid building 200,000 pydantic models. The cache key was `json.dumps(payload, sort_keys=True)`. Sorting the payload sorts the `criteria` map, so two rows with the same options in different orders hashed to the same key and the second row received the first row's question object -- while its `target`, an index into *its own* order, was left alone. Every shuffled Choice row of a source therefore trained on a label chosen by the first-seen order: measured on the local mixture, 224 of 646 clinc_oos rows, 282 of 682 banking77 rows and 54 of 116 emotion rows pointed at a wrong option. The file on disk was correct in every case; only the reader was wrong.

Before ADR-020 no source shuffled its options, so no two rows ever collided and the bug had no effect. The augmentation that fixed the "ignores its question" failure exposed it. Held-out splits keep canonical order, so calibration and evaluation read correctly; per-row QA sources have distinct option sets, so they were untouched; Noul and Score have no order to lose.

**What it did to the run.** `4b-instruct` (18,750 steps, 5h53) trained on those labels. Its evaluation: weighted 0.706, ECE 0.2135 → 0.1133; the new sources learned (sciq 0.975, arc_easy 0.927, snli 0.852, toxic_chat 0.976, mrpc 0.896); banking77 0.906; but clinc_oos 0.055 and emotion 0.605, the two sources with the highest collision rates, against 0.865 and >= 0.865 before. Those two numbers are the bug, not the model, and the Mode B training loss of ~2.3 against 0.8 in the first run is the same thing seen from the other side.

**Decisions.** The key preserves order (`sort_keys=False`; the payload comes from an ordered dump). `verify_round_trip` re-reads every written split at build time and refuses a file whose rows do not come back exactly as written -- a check that costs seconds and would have failed this build before the run started. The test suite re-introduces the bug and asserts the guard catches it.

**Consequence.** The mixture on the volume is correct and did not need rebuilding. The run was repeated with the fixed reader (`FRESH=1`, which now also moves the superseded run aside): Mode B loss 0.531 over the last 300 steps against ~2.3, held-out weighted 0.836 with ECE 0.0459, clinc_oos 0.976. FINDINGS.md §14.

---

## ADR-025 — Serving routes Mode A up to the tokenizer limit; training keeps its cap

**Accepted.** Partly reverses ADR-020's "same cap in training and serving".

ADR-020 capped Mode A at 26 options on both sides because the Base model, sent 60 lettered options it had never trained on, scored 0.291 with a letter-position bias that flipped 85% of answers on reordering. That was the right reading of that model. The instruct backbone is a different model: frozen, it scores 0.83 on the same subset in Mode A at 60 options.

Measured on the retrained instruct LoRA, same deployment, same 350 items, only the serving cap changed:

```text
    cap 26  → Mode B head      massive-en-US 0.231
    cap 76  → Mode A, 60 codes massive-en-US 0.746   (two-order averaged)
```

The head is excellent on the taxonomies it trained on (clinc_oos 0.976, banking77 0.922) and weak on one it has not seen; the backbone's zero-shot label-token reading survives the LoRA well enough to be worth 51 points on an unseen one. So serving now routes Mode A whenever the tokenizer expresses the codes (`EngineConfig.max_label_options = None`; 68 for this tokenizer -- the 69th code, `BQ`, is two tokens; this ADR first said 76, which was wrong), and Mode B above that. Training kept `LABEL_OPTION_CAP = 26` at the time this was written; ADR-026 moves training to the same tokenizer-limit routing and feeds the head from the large sources' full option sets instead. `LEV_SERVE_MAX_LABEL_OPTIONS` reproduces any other policy; `/health` reports the one in effect.

Two costs, stated. The LoRA gave back ~8 points against the frozen backbone in this regime (0.746 vs 0.83) -- Mode A was trained on at most 14 options, so the regime is under-trained rather than untrained, and option subsampling into the 15-68 range for the Mode B sources would close it. And calibration there is unfitted: `choice:A`'s temperature comes from ≤26-option questions, and at 60 options the model is under-confident (ECE 0.20, accuracy 0.953 on the 60% of items it puts above 0.5). A temperature bucket keyed on option count needs held-out rows in that range, which the mixture does not yet have.

Macro on the six S1Bench subsets with this policy: **0.697**, from 0.612.

---

## ADR-026 — The third run: data for the four remaining gaps

**Accepted.** Every point still short of Jev after ADR-025 is data (FINDINGS §15), so this run changes the mixture and nothing about the model.

| gap | source of the error | change |
| --- | --- | --- |
| paws −20.8 | paraphrase training rewarded lexical overlap | mrpc/qqp positives also yield a word-swapped negative (60%); `tasksource/parade` |
| vitaminc −9.5 | no fact-verification data | `pietrolesci/nli_fever`, FEVER's own labels |
| massive −6.8 vs Jev, −8 vs frozen | Mode A never trained above 14 options | large-taxonomy sources cut to 15+ options half the time; routing by tokenizer in training too |
| boolq −4.7 | little passage-grounded yes/no | race / sciq / openbookqa recast as "is the proposed answer correct?"; `ChilleD/StrategyQA` |

Three consequences follow. `TrainConfig.max_label_options` is None: a training row routes to Mode A whenever the tokenizer expresses its codes, as the server does, so the model trains on every set size it serves; Mode B's data is the large sources' full sets, kept half the time. The calibration split now varies option *sets* (never wording), so temperatures can be fitted per option-count band -- `choice:{mode}:{small|mid|large}` at 8 and 26 options -- with the unbanded bucket as fallback; the 60-option under-confidence of §15 was one scalar fitted on 3-14 options. And the yes/no recasts put "yes" on a factual axis in four more sources, so polarity is spread across sentiment, safety, paraphrase and fact.

The mixture must be rebuilt (`modal run modal/app.py::build_data`); the sources changed. Expected on the six S1Bench subsets: past the frozen backbone (0.719) and at Jev's level on four of six.

---

## ADR-027 — The prompt is dressed in the backbone's own format

**Accepted.** Closes Q11 with a mechanism, not just a number.

The frozen instruct backbone scored 0.653 on S1Bench through our prompts and 0.719 through reflex's (FINDINGS §15). reflex's difference is not wording alone: it wraps every request in ChatML -- the system/user/assistant turns the instruct model was trained on -- with headed sections, `A. option` lines, an explicit "respond with only the letter", and Qwen's empty `<think>` block opening the assistant turn. Ours was `Context: ... Question: ... Answer:`.

`prompt.Style` now offers both. Measured on the same frozen weights, same 1,999 items, the style changed and nothing else:

```text
    plain  0.653     chat  0.710     (reflex 0.719; our trained plain LoRA 0.697)
```

Every subset rose; helpsteer2 by 15.6 points to 0.400, above Jev. Zero-shot calibration changed more than accuracy did: ECE 0.49 → 0.12 on massive-en-US, 0.44 → 0.16 on vitaminc. A backbone answering in its native format is also a backbone that knows how sure it is.

Consequences. The style is a property of the weights: `TrainConfig.prompt_style` sets it, the release manifest records it, the server reads it from the manifest and `/health` reports it, and training and serving cannot disagree by accident. The label token changes with the style -- a space-prefixed letter after `Answer:`, a bare letter at the start of an assistant turn -- so the router, collator and readout take the prefix from the style rather than assuming one. The `4b-instruct` preset trains in `chat`; the Base presets stay `plain`, which is all a base checkpoint knows. The first measurement of this was a false negative -- the override never reached the container and the "chat" deployment served `plain`, bit for bit -- so the A/B now refuses to run unless `/health` confirms the style it was asked for.

**Expected for the ADR-026 run:** the adapter starts from 0.710 rather than 0.653 and trains in the format it will be served in, which is the most likely cure for the 8 points it gave back against the frozen backbone on massive.

---

## ADR-028 — Skip split label codes when serving; calibrate for families the model has not seen

**Accepted.**

**Codes.** The label-code scheme is A..Z, AA..ZZ, and the router required the *first n* codes all to be single tokens. For Qwen3.5 the 69th, `BQ`, is two tokens, so the Mode A limit was 68 (not 76, as ADR-025 first said; its cap-76 experiment was unaffected -- massive has 60 options). banking77's 77 options therefore went to the Mode B head. `skip_multi_token_codes` passes over codes that split and takes the next single-token ones. Measured on the same 400 held-out rows of each, production against a separately deployed variant:

```text
    banking77   77 options   Mode B 0.818   Mode A, skipped codes 0.980
    clinc_oos  151 options   Mode B 0.953   Mode A, skipped codes 0.968
```

Serving now skips by default; training does not, so Mode B keeps its data. Mode A reads 151 options well although training never showed it more than 68. The ADR-026 training cuts of 69-76 options went to Mode B, not Mode A as that ADR implied.

**Calibration for new families.** Temperatures fitted on held-out rows are fitted where the model is most reliable, and transfer badly: ECE 0.06 in-distribution against ~0.14 across the six S1Bench subsets. Two fits per bucket -- every row weighted equally, and every task family (an `ADJACENT` group, else a source) weighted equally -- are compared by leave-one-family-out ECE on the calibration split: fit on all families but one, measure on that one, average. **Selection rule, fixed before S1Bench was consulted:** per bucket, the fit with the lower leave-one-family-out ECE is kept; buckets drawn from fewer than three families keep the row fit. S1Bench is reported for the result, and for the previous profile (`calibration.previous.json`) beside it, but is not used to choose. A per-bucket temperature cannot fix a model that is confidently wrong through a learned shortcut (paws, vitaminc); the expectation is a partial improvement, not Jev's 0.08.

**Result.** The rule chose the family fit in the three buckets drawn from at least three families (`choice:A`, `choice:A:small`, `noul:A`), and it barely differs from the row fit: T 1.924 → 1.790, 1.826 → 1.767, 2.363 → 2.333, with leave-one-family-out ECE 0.032 → 0.028, 0.028 → 0.027, 0.047 → 0.046. Across the mixture's own families the row fit already transfers well. On S1Bench, reported after the choice: accuracy unchanged to four decimals (0.7248), mean ECE 0.1415 → 0.1359, almost all from vitaminc (0.246 → 0.207). S1Bench is further from training than any held-out family the calibration split can simulate, so the remaining out-of-distribution overconfidence is a property of the model on unfamiliar tasks, not of the temperature -- the lever for it is data (as paws showed: +10 points and a large ECE drop from training on its failure mode), not refitting.

---

## Open questions

| # | Question | How it gets settled |
| --- | --- | --- |
| ~~Q1~~ | ~~Is Jev's `confidence` normalised Gini?~~ | **Closed: no.** It is chance-corrected *max probability*, `(K·max − 1)/(K − 1)`, rounded to 2dp — mean abs error 0.0026 over 48 live answers. Gini shares the wrapper and has the wrong inner statistic. See FINDINGS.md §confidence |
| **Q2** | Do Mode A and Mode B agree where both are valid? | Explicit eval ([ADR-005](#adr-005--dual-mode-readout-the-differentiator)). A correctness gate, not a nice-to-have |
| **Q3** | Can a *state* cache persist across requests? | decider persists a **schema** cache; persisting state is unclaimed and is the genuinely novel direction |
| **Q4** | Does Mode B cost accuracy under the ceiling? | Ablation: Mode B forced on small option sets vs Mode A |
| **Q5** | How long does a 4B run actually take? | Read once at 23 h before bucketing, a cumulative-rate artefact (ADR-017). Sequence length and padding are measured, and the image installs both kernels (`flash-linear-attention`, `causal-conv1d`); record the wall clock of the next full run here |
| ~~Q6~~ | ~~Does an instruct checkpoint fix zero-shot Noul?~~ | **Closed by ADR-018, reopened by ADR-020.** Training fixed it in-distribution (0.975/0.915) and broke it out of distribution (aegis2 0.312, below every constant predictor). The instruct checkpoint became the starting point (ADR-020), and the released model reads Noul from the trained rating scale |
| **Q7** | Do the public training corpora transfer to support-triage states? | Train, then eval on both the generated set *and* the 24-item fixture. Agreement between them is the signal; the fixture alone cannot resolve it |
| ~~Q9~~ | ~~How does lev compare to Jev on S1Bench?~~ | **Answered on identical task files**, harness validated against Jev's own numbers. On the earlier six-subset definitions: 0.489 macro in the first run (FINDINGS.md §12, ADR-020), 0.725 in the third, against Jev's 0.754 (§16). On all 13 subsets as S1Bench pins them: 0.689 against Jev's 0.761, and 0.719 on the board's six, level with reflex-4b (§17) |
| ~~Q10~~ | ~~Does Mode B generalise to an unseen taxonomy?~~ | **Weakly: 0.166 on massive-en-US** under the cap, 10x chance and well calibrated (ECE 0.076), against Mode A's 0.291 and Jev's 0.814. Key format ruled out (0.140 = 0.140). Two training taxonomies were not enough; the rebuilt mixture is the fix. FINDINGS.md §12 |
| ~~Q11~~ | ~~Does the frozen instruct checkpoint match reflex's 0.719 through our engine?~~ | **No: 0.653.** Same weights, our prompts; reflex's prompts get 0.719. Six points of prompt/readout design, 17 on massive. FINDINGS.md §15 |
| **Q8** | Is 25% the right Mode B share? | Ablation at 10% / 25% / 40%, read on banking77 and clinc_oos accuracy against Mode A sources' regression |
