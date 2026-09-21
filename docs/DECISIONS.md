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

**Consequence:** a full run is ~16 h, so you can afford roughly a dozen. Budget the H100
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

---

## ADR-007 — Noul from nine rating tokens

**Accepted.**

Jev's `NoulAnswer` is a bare float: no `probabilities`, no `confidence` (verified at
runtime against the real SDK, [`FINDINGS.md`](FINDINGS.md) §2). It is therefore the one
question type whose calibration cannot be measured from a response.

Following simple-jev, we read Noul from a **9-level rating scale** and report
`p(yes) = Σ (i/8)·p_i` alongside the full distribution. Finer resolution, and Noul
becomes calibratable like Choice and Score. The `noul` field itself is unchanged.

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

## ADR-009 — The six evaluation subsets are banned from training

**Accepted.** Enforced in code, not documentation.

```
vitaminc-dev   massive-en-US   boolq   helpsteer2   aegis2   paws
```

Three S1Bench entries self-declare contamination and their numbers are compromised.
Our single differentiating claim is calibration measured on exactly these subsets, so
contamination would not merely weaken the result — **it would silently improve it**,
which is worse.

`lev.data.contamination` resolves aliases (`tals/vitaminc`, `paws-x`,
`google/boolq`, `nvidia/HelpSteer2`, …) and **raises**. It runs before any data loads.
decider's ~95-dataset registry contains several of the six and must be filtered.

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

## Open questions

| # | Question | How it gets settled |
|---|---|---|
| **Q1** | Is Jev's `confidence` normalised Gini? | `levbench confidence` against the live API. Identifier is validated by a known-answer test |
| **Q2** | Do Mode A and Mode B agree where both are valid? | Explicit eval (ADR-005). A correctness gate, not a nice-to-have |
| **Q3** | Can a *state* cache persist across requests? | decider persists a **schema** cache; persisting state is unclaimed and is the genuinely novel direction |
| **Q4** | Does Mode B cost accuracy under the ceiling? | Ablation: Mode B forced on small option sets vs Mode A |
| **Q5** | Is the 16 h estimate right? | `modal run modal/app.py::smoke`, then extrapolate from measured tokens/s |
