# Decision log

One record per choice that would be expensive to reverse. Each states what was
decided, what the alternatives were, and **what evidence settled it** — so a future
reader can reopen a decision when the evidence changes, rather than guessing at intent.

Status key: **Accepted** · **Superseded** · **Open**

---

## ADR-001 — Reproduce the abstraction, not the model

**Accepted.**

Jev's weights, training data and RLCD details are not public. Attempting to reproduce
*Jev* is unfalsifiable; reproducing the **computational abstraction** —
`f(state, question) → calibrated typed distribution`, with the state understood once
and reused across many questions — is a concrete, testable goal.

**Evidence:** [`FINDINGS.md`](FINDINGS.md) §10 — the public API surface is fully
specified and wire-compatible reimplementation is routine; the model is not.

---

## ADR-002 — Wire compatibility with `/v1/systemone` is non-negotiable

**Accepted.**

Our server speaks TypeSafe's exact request/response schema.

**Why it matters more than it looks:** it makes the benchmark honest. `levbench` measures
us and Jev through *the same code path*, with one changed flag. Any divergence in our
favour would otherwise be unfalsifiable. It also lets anyone swap us in behind an
existing `typesafe-sdk` client with a `base_url` change.

**One deliberate divergence:** our `NoulAnswer` adds `probabilities` and `confidence`.
Jev's Noul is a bare float, which makes it the one question type you cannot calibrate
from a response. The `noul` field is unchanged, so existing clients are unaffected.
See ADR-007.

---

## ADR-003 — Backbone: `Qwen/Qwen3.5-4B-Base`

**Accepted.** This is the decision everything else rests on.

### Why not Qwen3.8, which is newer?

Asked directly, and the answer is not preference — **Qwen3.8 has no 4B**. Enumerating
the family on the Hub returns exactly:

```
Qwen3.8-2.4T-A95B     Qwen3.8-27B     Qwen3.8-Flash-Next    (+ FP8 variants)
```

Three further facts decide it:

1. **No `-Base` checkpoints exist in the 3.8 family at all** — only instruct-tuned. For
   a logit-readout task you want the base model; decider and Nimble both used `-Base`.
   An instruct tune has already been shaped toward generating text, which is precisely
   the behaviour we are bypassing.
2. **The real 3.8 option is the 27B**, and on one H100 it is LoRA-only at 54 GB of
   weights, leaving 25.7 GB of headroom, at **~108 h per 3-epoch run**. That is 4.5 days
   for a single run — it eliminates the ablation budget entirely.
3. **Qwen3.5-4B is architecturally the same thing one tier down.** Verified from its
   `config.json`, not assumed:

```
32 layers = 24 linear_attention + 8 full_attention   (full_attention_interval 4)
hidden 2560 · head_dim 256 · 16 heads / 4 KV (GQA) · vocab 248320 · tied embeddings
max_position_embeddings 262144 · image_token_id 248056 (natively multimodal)
```

### What that config buys, for free

| Property | Consequence |
|---|---|
| Only 8 of 32 layers hold K/V | decider's persistent prefix cache **without pretraining a hybrid** — the other 24 carry small conv/recurrent state |
| 262 k context | The 512-token trap that caps laya and open-jev-deberta cannot occur |
| Native image/video tokens | Multimodal states at no extra cost |
| GQA 16/4 | Small per-state K/V footprint — and the state cache is the thing we fork N ways |

`Qwen3.5-2B-Base` is **24 layers = 18 linear + 6 full**, exactly the "6 full-attention
layers, 18 delta-net layers" decider describes. We are one tier up on a validated choice.

### Why 4B and not 2B or 9B

Measured, on S1Bench's completed runs: reflex-4b (Qwen3.5-4B) **0.7189**; decider-2b
(full fine-tune) **0.7033**. ~1.6 pp for double the parameters — and 4B still fits
comfortably under LoRA. 9B doubles memory and time for accuracy we do not need, because
**our target is calibration, not the accuracy crown** (ADR-006).

### Feasibility table, one H100 80 GB, bf16

| backbone | weights | full-FT state | LoRA state | headroom | h / 3 epochs | verdict |
|---|---|---|---|---|---|---|
| Qwen3.5-2B-Base | 4 GB | 32 GB | 4.3 GB | 75.7 GB | 8 | fallback |
| **Qwen3.5-4B-Base** | **8 GB** | 64 GB | **8.3 GB** | **71.7 GB** | **16** | **chosen** |
| Qwen3.5-9B-Base | 18 GB | 144 GB | 18.3 GB | 61.7 GB | 36 | if 4B underfits |
| Qwen3.8-27B | 54 GB | 432 GB | 54.3 GB | 25.7 GB | 108 | no Base ckpt, no ablations |

**Reopen this if:** a Qwen3.8 `-Base` appears at ≤9B, or you get more than one H100.

---

## ADR-004 — LoRA, not a full fine-tune

**Accepted.**

4B full fine-tune needs ~64 GB of weights + gradients + AdamW state, leaving ~16 GB for
activations — too tight at long context. LoRA r32 needs ~8.3 GB, leaving ~72 GB.

**The non-obvious part that made this safe:** Mode B's matching head is *new* parameters
and trains at full precision regardless of LoRA freezing the backbone. Freezing the
backbone does not prevent training a new head, which is what initially made this look
like a trade-off and turns out not to be one.

**Consequence:** a full run is ~2 h once measured (see ADR-016), so a dozen is cheap. Budget the H100
for **ablations, not one heroic run**.

---

## ADR-005 — Dual-mode readout (the differentiator)

**Accepted.** This is the one genuinely novel piece.

Every implementation surveyed picks exactly one readout family:

- **A — label-token:** map options to single tokens, read logits at `Answer:`. Zero
  parameters, works untrained. Hard option ceiling; decider measured **−5 to −24 points**
  on 50–219 option sets.
- **B — candidate-path:** encode each candidate's *text*, score with a trained matching
  head. No ceiling. Costs a head; NanoJev shows the candidates batch into one forward
  ("44 candidate paths, 1 backbone forward").

**Nobody does both.** We route per question, and critically: **the boundary is
tokenizer-verified single-token-ness, not an option count.** Where LitJev *rejects* a
tokenizer that cannot express the codes, we fall through to Mode B. Their hard failure
becomes our second mode.

**Two obligations this creates**, both treated as first-class rather than assumed away:

1. A and B produce distributions by different mechanisms, so they get **separate fitted
   temperatures** (ADR-006).
2. Where both are valid we **measure agreement**. Material disagreement makes the router
   a correctness hazard, not merely a capacity switch.

---

## ADR-006 — Calibration is the product

**Accepted.** The most important decision after the backbone.

S1Bench, completed runs only:

| | macro | ECE |
|---|---|---|
| jev | 0.7751 | **0.0764** |
| simplejev-qwen38-27b | 0.7582 | 0.1214 |
| djev-full | 0.7485 | 0.1661 |
| **reflex-4b** | 0.7189 | **0.0849** |
| qwen3-8b-full (untuned) | 0.5346 | 0.4252 |

Two readings settle the project's direction:

1. **The accuracy gap is ~1 point.** Open reproductions have essentially caught Jev on
   accuracy. Competing there is a losing, expensive fight.
2. **reflex-4b is the only open model with both accuracy and calibration** — and it gets
   there *not by architecture* but by fitting **one scalar** post-hoc. Untuned: 0.4252.
   With a temperature: 0.0849. Jev: 0.0764.

**Calibration is a cheap bolt-on that almost none of the clones bothered with.** That is
the winnable fight, and it is why `lev.calibrate` fits **per (question type, readout
mode)** — one global scalar under-serves at least one bucket — and why
`calibrate.fit()` **raises** on a split named test/eval/holdout.

### First measured head-to-head

Jev `jev-1.13.0` against lev on `Qwen3.5-4B-Base` (untuned, Mode A, no calibration),
same 24 items, same questions, `levbench eval`:

| question | type | Jev acc | lev acc | Jev ECE | lev ECE |
|---|---|---|---|---|---|
| department | Choice | 0.958 | **0.958** | **0.0250** | 0.2320 |
| frustration | Score | 0.750 | 0.500 | 0.1350 | 0.4635 |
| is_urgent | Noul | 0.917 | 0.292 | 0.0804 | 0.5261 |

**Choice is a dead tie on accuracy and a 9× gap on calibration.** That is ADR-006's
whole argument, reproduced on our own data at the first attempt: an untuned 4B
already matches a frontier decision model at picking the right option, and loses
entirely on knowing how sure it is.

Three further readings:

- **lev's Choice is *under*confident, not over.** Every reliability bin scores
  accuracy 1.000 while reporting 0.15–0.93 confidence — all gaps negative. Fitting a
  temperature gives **T = 0.409** (sharpening, not flattening). That is the easy
  direction to fix and further evidence for ADR-006.
- **Score is Jev's weak primitive too.** Its `frustration` log loss is **1.9484**
  against `ln(3) = 1.099` for a uniform guess — worse than chance despite 75%
  accuracy, meaning it is confidently wrong on the quarter it misses. Consistent
  with S1Bench, where the Score-shaped `helpsteer2` subset sat at 0.348 for everyone
  including Jev.
- **Measured ECE depends on which statistic the server calls `confidence`.** lev
  reports normalised Gini (LitJev's choice); scoring the *same* distributions by
  max-probability instead gives department ECE 0.0878 rather than 0.2320. So part of
  the gap above is a statistic mismatch rather than a worse distribution — which is
  exactly why open question Q1 has to be settled before this table is read too hard.

Jev also reported **1,920 output tokens** across the 24 calls (~80 per call), billed
at $0. lev reported 0: it genuinely generates nothing.

---

## ADR-007 — Noul from nine rating tokens

**Accepted.**

Jev's `NoulAnswer` is a bare float: no `probabilities`, no `confidence` (verified at
runtime against the real SDK, [`FINDINGS.md`](FINDINGS.md) §2). It is therefore the one
question type whose calibration cannot be measured from a response.

Following simple-jev, we read Noul from a **9-level rating scale** and report
`p(yes) = Σ (i/8)·p_i` alongside the full distribution. Finer resolution, and Noul
becomes calibratable like Choice and Score. The `noul` field itself is unchanged.

### Measured caveat: the 9-point scale does not survive an untuned base model

First real run, `Qwen3.5-4B-Base`, Mode A, no fine-tune, no calibration, on the
24-item triage set:

| question | type | accuracy | ECE |
|---|---|---|---|
| department | Choice | **0.958** | 0.232 |
| frustration | Score | 0.500 | 0.464 |
| is_urgent | Noul | **0.292** | 0.526 |

0.292 is exactly 7/24 — the positive base rate. The model answered *yes to every
item*. Probing it directly shows why: the rating distribution is pinned at the
endpoints with P(8) ≈ 0.8 regardless of content (0.839 on a clearly non-urgent
question, 0.797 on a clearly urgent one — flat, and slightly inverted).

**Choice works zero-shot on a base checkpoint; a 9-point rating scale does not.**
A base model has no instruction-following prior for "Rate 0-8", so the digits
after `Answer:` reflect token priors rather than judgment. This does not
invalidate ADR-007 — the scale is still the right *trained* target, and it is the
only way to make Noul calibratable — but it means the untrained baseline in the
build order cannot use it. Options, in preference order:

1. Run the zero-shot baseline on an **instruct** checkpoint (`Qwen3.5-4B`, not
   `-Base`), keeping `-Base` for the fine-tune where it belongs.
2. Fall back to a 2-token yes/no Noul until the scale is trained.

This is the clearest empirical support so far for ADR-011's "shipping untuned is
not an option" — and it identifies *which* primitive fails first.

---

## ADR-008 — Both prompt layouts, trained 50/50

**Accepted.**

- **state-first** — cache the state, fork across questions. The shared-state win.
- **schema-first** — cache the question catalogue across many states.

Random layout per example buys the choice at inference. decider's measured cost of
schema-first is inherited as a **routing rule**, not a preference:

| workload | cost |
|---|---|
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

Three S1Bench entries self-declare contamination and their numbers are compromised.
Our single differentiating claim is calibration measured on exactly these subsets, so
contamination would not merely weaken the result — **it would silently improve it**,
which is worse.

The block list covers all **13** subsets in the snapshot's `published_jev`, not the 6
that happened to execute in the `s1-fast` suite. The other 7 are evaluation data that
simply did not run; training on them would contaminate any later full-suite comparison
and we would not find out until the number was already published. Blocking only what
ran is how you get a result that looks good and means nothing. The cost of the wider
list is four otherwise-usable sources (squad2, multinli, civil_comments, pubmedqa) —
cheap next to an uninterpretable headline.

`lev.data.contamination` resolves aliases (`tals/vitaminc`, `paws-x`, `google/boolq`,
`nvidia/HelpSteer2`, `rajpurkar/squad_v2`, `nyu-mll/multi_nli`, …) and **raises**. It
runs before any data loads. decider's ~95-dataset registry contains several of the
thirteen and must be filtered.

---

## ADR-010 — Two packages, one workspace

**Accepted.**

`packages/levbench` (measurement) and `packages/lev` (the model) are separate
distributions. The benchmark predates the model, is useful on its own against Jev and
any compatible server, and **must not depend on our model** — a measuring instrument
that imports the thing it measures is not an instrument.

The lev **core** (schema, prompt, router, calibration, contamination) imports
without torch, so it is fully testable on a laptop; everything needing a GPU sits behind
the `[train]` extra.

---

## ADR-011 — Rejected trade-offs

**Accepted.** Each refusal has evidence; see [`ARCHITECTURE.md`](ARCHITECTURE.md) §4.

| Rejected | Evidence |
|---|---|
| 512-token context | Structurally destroys the shared-state premise. kotoba measured DeBERTa-v3-large as not fitting 100 questions in 512 |
| Diffusion backbone | djev: accuracy 0.7485 at ECE 0.166–0.178 (2.2× Jev). kotoba clocked a dLLM at 846 ms vs 19–34 ms for ModernBERT-base |
| Pure linear/recurrent | simplejev-rwkv holds the bottom three slots (0.313–0.380). Hybrid yes, pure RWKV no |
| Shipping untuned | ECE 0.4252 completed, 0.366–0.493 stopped. Uncalibrated confidence is worthless, and confidence is the product |
| Optimising for speed | `verdict`: 101 dec/s at ECE 0.4622. Fast and confidently wrong is not a decision model |

---

## ADR-012 — Abstain means taking the state away

**Accepted.**

10% of training examples are unanswerable. The first implementation set a flag on
an otherwise ordinary example and changed nothing else, which is worse than no
augmentation at all: the state still determines the answer, so the only thing the
model learns is to be unsure when it should not be. That is a calibration
*regression* dressed as a calibration aid.

An abstain example now pairs a question with a state drawn from a **different
source**, and supervises a **uniform** distribution over the candidate set. With
no evidence, every candidate is equally supported, so the uniform vector is the
calibrated answer rather than a hedge — and teaching it is the point.

This is why `decision_loss` takes soft targets. The hard rows are unaffected:
with a one-hot target the soft cross-entropy is exactly `F.cross_entropy`, which
is asserted in `test_loss.py`.

Abstain rows are training-only. They are excluded from the exported eval set,
because scoring them would measure our abstention rather than our accuracy, and
the two are different numbers.

---

## ADR-013 — Mode B is weighted by what must be learned, not by corpus size

**Accepted.**

Only two of the nine sources — `banking77` (77 intents) and `clinc_oos` (151) —
have option sets that overflow single-token label codes, so they are the *only*
Mode B training signal. Every other source trains Mode A and nothing else.

Weighting the mixture by corpus size would give the pair a few percent of the
examples, and the head that is supposed to be this project's differentiator
([ADR-005](#adr-005--dual-mode-readout-the-differentiator)) would ship untrained. `default_weights()` gives them **25%**. The
remaining 75% splits evenly across the three primitives, so no readout starves.

The cost is accepted knowingly: those two sources are over-represented relative
to any natural distribution, which will bias Mode A's option-set prior toward
short intent phrases. `test_data_pipeline.py` asserts the 25% floor so a later
"tidy-up" of the weights cannot quietly undo it.

---

## ADR-014 — Split before mixing, into three splits

**Accepted.**

Three splits, not two: a temperature fitted on the test set is not a measurement.
`calibrate.fit()` already refuses a split named test/eval/holdout; the
`calibration` split is the other half of that guarantee — the thing it can
legitimately accept.

Assignment is a **blake2b hash of a stable per-row key**, not an RNG draw. The
same row lands in the same split on any machine, in any dataset order, on any
re-run, which is what makes a resumed or repeated experiment comparable to the
original. Python's `hash()` is salted per process and would not reproduce
tomorrow.

Splitting happens **before** the mixture is drawn. Mixing first would let one
underlying row appear in train and in test wearing two different layouts — a
contamination leak with our own data rather than S1Bench's. Subtler than
ADR-009's, and flattering in exactly the same way.

Two coverage checks, because one is not enough. Every label *observed* must
appear in train, or its error rate measures nothing. And every label the
*question offers* must be observed at all — the check that catches an
undersampled banking77, where 3 intents out of 77 satisfy the first check and
the mixture is still junk. Both raise.

---

## ADR-015 — The 24-item task set is a fixture, not a benchmark

**Accepted.**

At n=24 the 95% interval on an accuracy estimate is about ±16 points. A training
run that moved accuracy by 5 points is indistinguishable from one that moved it
by nothing, so the hand-labelled support set cannot answer the only question
training raises.

It stays, as a smoke fixture: does the server answer, are the types right, does a
distribution come back. The real eval is exported from the mixture's held-out
test split — 2,760 items across nine sources, ±4 points per source — and
`levbench eval --tasks` reads it.

One file per source, because each source carries exactly one question. A single
combined file would make the harness ask every question of every item, i.e. ask
"how positive is this review?" of a banking ticket, and score the answer.

The exporter lives in `lev`, not in `levbench`. levbench must not import the
thing it measures (ADR-010), so the handoff is a JSON file and the writer sits on
the model side of the fence.

---

## ADR-016 — Sequence length is measured, and the budget was wrong by 10x

**Accepted.**

`avg_tokens_per_example` was `1200`. It was a guess, made before any data
existed. The measured mean over 1,500 rendered prompts from the real mixture is
**121 tokens** — median 73, p95 390, longest 1,104.

| Source | Mean tokens |
|---|---|
| clinc_oos | 38 |
| banking77 | 39 |
| rotten_tomatoes | 65 |
| emotion | 71 |
| sst5 | 77 |
| ag_news | 98 |
| dbpedia_14 | 164 |
| yelp_review_full | 223 |
| imdb | 305 |

The 4B budget therefore read **16.0 hours** when it is closer to **1.7**. That
error is not academic: it is the number the H100 is booked against, and it was
wrong in the direction that makes you plan for one run when you can afford ten.

Three things changed as a result:

- The default is `128`, with the measurement and its provenance in the
  docstring, and `lev plan --data <dir>` re-measures against a real mixture
  rather than trusting it.
- `tokens_per_step` was `max_seq_len × batch`, which bills the truncation cap
  rather than the work. The collator pads to the longest row in the batch, so
  at a 4,096 cap over 121-token data that over-counted by ~16x. It is now
  `avg_tokens_per_example × batch`, and `steps_per_epoch` counts **examples**,
  because an epoch is a pass over the data and how many optimiser steps that
  takes is a function of the batch, not of a token budget.
- `per_device_batch` went 8 → 32. At a 121-token mean, a batch of 8 makes
  75,000 optimiser steps and the run is bound by step overhead long before it
  is bound by FLOPs. `max_seq_len` dropped 4,096 → 2,048, which still clears
  the longest prompt seen with room to spare.

The general lesson, and the reason this is an ADR rather than a commit message:
**every number in the budget that was not measured was wrong.** The FLOPs
constant and the H100 throughput figure are still assumptions, so Q5 stays open.

---

## ADR-017 — Batches are length-bucketed, and the budget was wrong again

**Accepted.** Measured on the first real H100 run, not predicted.

The first `train --preset 4b` on Modal reported:

```
step 25/18750  0.1%  A=5.7621  B=5.1222  lr=4.45e-06  0.22 it/s  826 tok/s  eta 23:13:22  mem 29.3G
```

**23 hours against a 24-hour timeout**, and 826 tok/s where ADR-016's arithmetic
implied ~12,000. Two causes, both measured afterwards:

**Padding, 4.4x.** A batch is padded to its longest row, so the model computes
on the rectangle rather than on the real tokens. The mixture is bimodal — a
banking intent is ~39 tokens and an imdb review reaches 1,300 — so one long row
in a batch of 32 drags the whole batch up to it. Through the real collator on
the real mixture:

| batching | real tokens | padded | waste |
|---|---|---|---|
| random | 380,794 | 1,686,163 | **4.43x** |
| length-bucketed | 380,794 | 542,779 | **1.43x** |

`ModeBatcher` now sorts by length inside a shuffled window of `bucket_window`
(64) batches, then shuffles the batch order. Sorting globally would feed every
short example before every long one and correlate length with step number;
sorting within a window keeps the order effectively random. The proxy is
character count, because tokenising twice would cost more than the padding it
saves — the resulting buckets measure 1.43x against a 1.03x theoretical floor.

**Kernels, the rest.** 24 of Qwen3.5-4B's 32 layers are linear-attention, and
without `flash-linear-attention` transformers runs `chunk_gated_delta_rule` on
its reference PyTorch path — three quarters of the model on the slow route.
That layer is now in the image. It had been left out on the grounds that a
failed build is worse than a slow run; a run that does not fit its timeout
changes that trade.

`causal-conv1d` is **not** included, and the first attempt to add it alongside
proved why the two are not interchangeable. It is a CUDA *source* build that
needs `nvcc`, which `debian_slim` does not ship, so the image build dies at
`Getting requirements to build wheel` with `NameError: name
'bare_metal_version' is not defined` — a confusing way to say "no compiler".
Including it would mean an `nvidia/cuda:*-devel` base and a far heavier image,
to accelerate only the short depthwise convolution.
`flash-linear-attention` carries the linear-attention core and is pure Python
and Triton, so it needs no compiler. The lesson is narrower than "install the
kernels": **a pure-Python wheel and a CUDA source build are different
decisions**, and bundling them into one turned a correct call into a failed
build.

**The trade accepted:** length-bucketed batches are more homogeneous in source,
because length correlates with source here. Over an epoch the shuffled batch
order distributes them evenly, and the alternative is paying 3x. Worth
revisiting if the loss curves look source-periodic.

**A caveat on the 23 h figure itself.** It was a *cumulative* rate read at step
25, so it was dominated by startup — weight loading, allocator warmup, and
Triton JIT. The 0.8B smoke makes the size of that effect concrete: 4.68 s/step
over the first 25 steps, 0.40 s/step after, reported cumulatively as 0.33 it/s
against a true 2.50. `ProgressLog` now measures rate, throughput and ETA over
the window since the last report. The padding finding stands on its own —
4.43x was measured offline through the collator, not inferred from the ETA —
but the 23 h number was inflated by an unknown amount and should not be quoted
as a baseline.

**The general lesson, restated from ADR-016 because it recurred:** the estimate
was built from a token count that was correct and a padding factor that was
assumed to be 1. Both ADR-016 and this one are the same failure — an unmeasured
term in the budget — and the throughput readout that exposed it only existed
because the loop had been made to print (see `ProgressLog`). Q5 stays open until
a full run lands.

---

## ADR-018 — What the first trained checkpoint actually shows

**Accepted.** Measured on 1,800 held-out items, `step-18750`, `modal run
modal/app.py::evaluate`.

Training: 18,750 steps, ~4h50 on one H100, every loss finite. Mode A 1.754 →
0.233, Mode B 4.843 → 0.813 from an `ln(151) = 5.02` chance start.

| | uncalibrated | calibrated |
|---|---|---|
| accuracy | 0.856 ±0.016 | 0.856 ±0.016 |
| ECE | 0.1162 | **0.0529** |

Four findings, in descending order of how much they change the design.

**Q6 is closed: the Noul collapse was a base-checkpoint artefact.** Untrained,
the model answered "urgent" for everything — 0.292, `P(8) ≈ 0.8` regardless of
content (ADR-007). Trained: imdb 0.975, rotten_tomatoes 0.915. Mapping the
binary class onto the *ends* of the 0-8 rating scale rather than passing it
through as a class index is what fixed it.

**Mode B works, and it is the differentiator.** banking77 0.870 over 77 options,
clinc_oos 0.865 over **151** — within 6 points of Mode A's average, with no
option ceiling. Every Family-A implementation surveyed in ARCHITECTURE.md §2
either caps the option count or rejects the request at this point.

**Per-bucket calibration was the right call, and now there is evidence.**
`choice:B` fitted to T=1.033 — the candidate-path head came out essentially
calibrated on its own — while Mode A needed 2.084 (choice), 2.981 (noul) and
4.096 (score). A single global temperature of ~2.5 would have actively damaged
Mode B. That rule was argued from first principles in `calibrate.py`; it is now
measured.

Two sources got *worse* ECE under calibration: dbpedia_14 0.0049 → 0.0149 and
imdb 0.0244 → 0.0427. Both were already near-perfectly calibrated and were
over-softened, because the bucket key is `(type, mode)` and is shared across
every source in the bucket. Log loss improved on both, so the temperature is
net-positive even there, but per-source calibration is the obvious refinement.

**Score is the weak spot.** sst5 0.545, yelp_review_full 0.680; everything else
is ≥0.865. Both are 5-level ordered scales where adjacent levels are genuinely
ambiguous. sst5 also had the worst uncalibrated ECE at 0.4336. A next run should
target Score specifically.

**These numbers are not comparable to Jev's 0.7751 / 0.0764.** This is a
held-out split of the same nine corpora the model trained on — in-distribution.
S1Bench is thirteen different subsets, deliberately blocked from training
(ADR-009) and still untouched. Reporting 0.856 against Jev's 0.7751 would be the
exact error the contamination guard exists to prevent.

---

## ADR-019 — The evaluation is not bit-reproducible, so the report states its interval

**Accepted.**

Two runs of `evaluate` over the same checkpoint and the same data disagreed:
yelp_review_full accuracy 0.685 → 0.680 (one item in 200), clinc_oos ECE 0.0515
→ 0.0421. Accuracy on the other eight sources was identical.

The cause is GPU kernel non-determinism: bf16 reductions and the Triton
linear-attention kernels do not pin their reduction order, so logits differ in
the last bits and items near a decision boundary flip. clinc_oos moved most
because a 151-way softmax has many near-ties sitting on bin edges and ECE is a
*binned* statistic; log loss barely moved (0.4871 → 0.4873), which is the tell
that nothing real changed.

`torch.use_deterministic_algorithms(True)` was rejected: it would likely refuse
the linear-attention kernels and give back most of the speedup ADR-017 bought,
to remove a difference that is far inside the sampling interval anyway.

Instead the report prints its own resolution — a 95% half-width on every
accuracy, ±0.048 at n=200 and ±0.016 at n=1,800, with a standing note that ECE
moves below ~0.01 are noise. A four-decimal table that is reproducible to two is
worse than no table, because a reader takes the fourth decimal for a result.

---

## Open questions

| # | Question | How it gets settled |
|---|---|---|
| ~~Q1~~ | ~~Is Jev's `confidence` normalised Gini?~~ | **Closed: no.** It is chance-corrected *max probability*, `(K·max − 1)/(K − 1)`, rounded to 2dp — mean abs error 0.0026 over 48 live answers. Gini shares the wrapper and has the wrong inner statistic. See FINDINGS.md §confidence |
| **Q2** | Do Mode A and Mode B agree where both are valid? | Explicit eval ([ADR-005](#adr-005--dual-mode-readout-the-differentiator)). A correctness gate, not a nice-to-have |
| **Q3** | Can a *state* cache persist across requests? | decider persists a **schema** cache; persisting state is unclaimed and is the genuinely novel direction |
| **Q4** | Does Mode B cost accuracy under the ceiling? | Ablation: Mode B forced on small option sets vs Mode A |
| **Q5** | How long does a 4B run actually take? | Measured once at 23 h before bucketing (ADR-017). Sequence length and padding are now both measured; kernel throughput is not. `modal run modal/app.py::smoke`, then extrapolate from measured steps/s. Note that 24 of the 32 layers run on a reference PyTorch path unless `flash-linear-attention` is installed, so the first measurement may not be the ceiling |
| ~~Q6~~ | ~~Does an instruct checkpoint fix zero-shot Noul?~~ | **Closed by ADR-018.** Training fixed it: 0.292 untrained → 0.975/0.915. The instruct checkpoint was never needed |
| **Q7** | Do nine public classification corpora transfer to support-triage states? | Train, then eval on both the generated set *and* the 24-item fixture. Agreement between them is the signal; the fixture alone cannot resolve it |
| **Q9** | How does lev compare to Jev on S1Bench? | No harness exists. The thirteen subsets are blocked from training and untouched, so the comparison is available but unbuilt — see ADR-018 |
| **Q8** | Is 25% the right Mode B share? | Ablation at 10% / 25% / 40%, read on banking77 and clinc_oos accuracy against Mode A sources' regression |
