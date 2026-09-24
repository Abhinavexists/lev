# Designing lev: what to take from each implementation, and what to refuse

This is the build document. [`FINDINGS.md`](FINDINGS.md) establishes what Jev *is*; this establishes what **lev** should be. It was written before the first training run: §§1–4 are the survey it rests on, §5 the design as built. Where the build later departed from §5 the change is noted inline, with the ADR that records it.

Every model below is on the S1Bench leaderboard, so each architectural claim is paired with measured accuracy, calibration and speed rather than a README's self-assessment.

**Reading rule:** S1Bench has two groups. **Completed** = 6 subsets, 1,999 decisions. **Stopped** = `vitaminc-dev` only, 599 rows. Stopped numbers are *not* comparable and are never used for conclusions here. This matters — `jeff-gpu` looks superb at 0.6644/ECE 0.0502 but is a stopped run; its completed sibling `jeff-gpu-full` is 0.5595/0.0738.

---

## 1. The architectural fork

Every implementation picks one of two ways to turn a forward pass into a typed distribution. They are not a ladder — **they compose**, and that is the opening.

### Family A — label-token readout

*simple-jev, litjev, decider, reflex, djev*

Map each option to a single token (`A`, `B`, … then `AA`…), prefill the state once, fork per question, read next-token logits at the `Answer:` boundary restricted to those label tokens, softmax.

- **Gain:** zero added parameters. Works on any pretrained decoder, untrained.
- **Cost:** a hard option ceiling (50–255), tokenizer verification, and accuracy decay as the label set grows. decider documents this honestly: *"on full label sets of 50–219 options and on states of several thousand tokens it loses 5–24"* points.
- The option *text* is only seen in the prompt; the scored token is a bare letter.

### Family B — candidate-path scoring

*NanoJev, jeff (GLiFormer), kotoba-lang's encoder*

Encode each candidate's actual text, score it against the question with a trained matching head.

- **Gain:** no option ceiling, arbitrary option strings, the option's semantics reach the scorer directly.
- **Cost:** a trained head, and candidates must be encoded. Not N forward passes though — NanoJev reports **"6 states · 18 questions · 44 candidate paths · 1 backbone forward"** using set attention.

**Nobody on the leaderboard does both.** A dual-mode readout — Family A under the token cap, Family B above it — is the clearest unclaimed design, and it removes the single hardest constraint in the category.

---

## 2. Model-by-model dissection

Completed runs only. `Mac?` = plausibly runnable on Apple Silicon.

| Model | Architecture | macro | ECE | dec/s | Mac? | The good part | The trade-off it carries |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **jev** (anchor) | proprietary, RLCD | 0.7751 | **0.0764** | 2.39 | api | Best calibration at top accuracy; 255 options | Slow vs small models; closed |
| **simplejev-qwen38-27b** | Qwen3.8-27B, label-token readout | 0.7582 | 0.1214 | 1.63 | no | Purest Family A; 1.0pp off Jev untrained | No calibration step; 50 options, 2k tokens |
| **djev-full** | DiffusionGemma 26B-A4B, mask-slot readout | 0.7485 | 0.1661 | 3.70 | no | Parallel slots; 128 options, 4k tokens | ECE 2.2× Jev; needs a DGX |
| **simplejev-qwen36-35b-a3b** | Qwen3.6-35B-A3B, label-token | 0.7442 | 0.1365 | 1.67 | no | MoE keeps cost down at 35B | Same: uncalibrated, 50 options |
| **reflex-4b** | Qwen3.5-4B, label-token **+ post-hoc temperature** | 0.7189 | **0.0849** | 7.25 | **yes (CPU)** | **The only open model with both accuracy and calibration.** Multimodal; WebGPU demo | 26-option cap |
| **decider-2b** | Qwen3.5-2B-Base hybrid (6 full-attn + 18 delta-net), schema cache | 0.7033 | 0.1142 | **27.17** | 0.8b variant | **255 options, 32k tokens, 27 dec/s** — best capability envelope | Self-declared contamination; GPU for the 2B |
| **laya-gpu** | ModernBERT-large 421M + 2 layers | 0.6254 | 0.1304 | 17.86 | yes | Tiny and quick | **512-token state cap** — fatal |
| **jeff-gpu-full** | GLiFormer-large 400M (GLiNER family) | 0.5595 | 0.0738 | 6.21 | yes | Family B at 400M; serves `/v1/systemone` | Accuracy 22pp below Jev |
| **open-jev-deberta** | DeBERTa-v3-large 435M, matching head | 0.5235 | **0.0668** | 37.44 | yes | Best ECE on the board; 37 dec/s | **512 ctx**; contamination; 25pp down |
| **reflex-08b** | Qwen3.5-0.8B, label-token | 0.5158 | 0.2426 | 24.39 | yes | Same code path as reflex-4b | Calibration collapses at 0.8B |
| **kev-05b** | Qwen2.5-0.5B + LoRA + readout head | 0.4926 | 0.1492 | 0.55 | yes | Runs on a laptop | Slowest on the board |
| **gliner-\*** | GLiNER 2.5, 74–287M | 0.40–0.43 | 0.18–0.32 | 2.9–7.2 | yes | Tiny | Repurposed NER; weak at judgment |
| **qwen3-8b-full** (raw) | untuned, label-token | 0.5346 | **0.4252** | 2.90 | no | Free baseline | **Catastrophically overconfident untuned** |
| **simplejev-rwkv-\*** | RWKV linear attention | 0.31–0.38 | 0.28–0.30 | 1.9–2.5 | — | — | **Bottom three slots.** See §4 |

Not scored above but architecturally important: **NanoJev** (Qwen3-0.6B + decision heads, full training pipeline) and **kotoba-lang/typed-decisions** (two backbones benchmarked head-to-head).

Two *stopped* runs are worth naming despite being excluded from the table, because they mark the extremes: **verdict** (151M) hit **101 dec/s at ECE 0.4622** — the fastest and least trustworthy point on the board — and the other four raw `qwen3-*` checkpoints all landed at ECE 0.366–0.493, agreeing with the completed `qwen3-8b-full` above.

---

## 3. The parts worth taking

### 3.1 Post-hoc temperature calibration — from `reflex` — highest value per unit effort

The single most important finding in the data. **reflex-4b is the only non-Jev model with both high accuracy and good calibration** (0.7189 / ECE 0.0849), and it gets there *not by architecture* but by fitting one temperature on held-out data after training:

```bash
reflex-eval-mmlu --n 1200 --fit-temperature runs/calibration.json
reflex-serve --calibration runs/calibration.json
```

Compare the untuned Qwen backbones at ECE 0.4252 (completed) and 0.366–0.493 (stopped). **Calibration is a cheap bolt-on that almost none of the clones bothered with.** Our plan's §15 was right to make it first-class — and it is far easier than the plan assumed. Fit one scalar, never on test labels.

### 3.2 Dual-layout training — from `decider`

decider trains **two prompt layouts 50/50**:

- **state-first** (`Context … Question … Options … Answer:`) → cache the *state*, fork across questions. This is the shared-state win.
- **schema-first** (questions/options before the state) → cache the *schema*, reuse it across many states.

Training both buys the choice at inference. Their measured cost of schema-first is specific and worth inheriting as a decision rule: **−1.5 points** on fixed-label tasks, **−5** when options change per example, **−5 to −24** on 50–219 options or multi-thousand-token states. So: schema-first for high-volume fixed-schema batch work, state-first everywhere else.

### 3.3 Hybrid attention to make the cache affordable — from `decider`

decider is 6 full-attention layers + 18 delta-net layers. Only the 6 full layers hold a K/V cache; the rest carry recurrent state. That is what makes a **read-only, cross-request** prefix cache cheap enough to persist (`Decider.schema`, `DECIDER_SCHEMA_CACHE=1`).

**This settles a question left open in FINDINGS §4(b):** a pinned cross-request cache *does* exist in the wild — but it caches the **schema**, not the **state**. Persisting a *state* encoding across requests remains unclaimed, and is still the genuinely novel direction.

Note the tension with §4: hybrid ≠ pure linear. RWKV (fully recurrent) is bottom of the board. decider keeps real attention layers and uses delta-net only for the rest.

### 3.4 Noul as nine rating tokens — from `simple-jev`

simple-jev derives Noul from the model's distribution over **nine rating tokens**, not a two-way yes/no softmax. Finer resolution, and it gives Noul an actual distribution.

This directly fixes the asymmetry FINDINGS §2 flagged in the real API, where `NoulAnswer` is a bare float with no `probabilities` and no `confidence`. **Our Noul should return a distribution.** It costs nothing and makes Noul calibratable like the other two types.

### 3.5 Set attention over candidate paths — from `NanoJev`

`Choice` uses a shared scalar head plus **set attention** over candidates; `Score` evaluates ordered level descriptions and returns the probability-weighted expectation; `Boolean` is a single-path sigmoid. One shared head serves all three types.

This is our plan's §14 ("encode the question, don't build per-question heads") done properly — and because candidates are scored as *text paths*, it carries no single-token ceiling. 44 candidate paths resolve in one backbone forward.

### 3.6 Training objectives — from `NanoJev` and `decider`

- NanoJev: CE / Brier / paired-proper-reward variants, with exact gradient checks.
- decider: cross-entropy fine-tune with **random layout per example** and **abstain augmentation**; evaluation reports accuracy / NLL / Brier / ECE / AURC / selective accuracy per task; v10 adds **RL with a proper-score belief reward** — the open analogue of RLCD.

Supervised proper-scoring first, RL only after. That vindicates the plan's §17 ordering.

---

## 4. Trade-offs to refuse, each with its evidence

| Refuse | Evidence |
| --- | --- |
| **512-token context** | open-jev-deberta and laya are capped at 512. kotoba measured DeBERTa-v3-large as *"512 ctx に入らない"* — it does not fit 100 questions. A 512-token state destroys the shared-state premise, which is the whole point |
| **Diffusion backbone** | djev buys accuracy (0.7485) at ECE 0.166–0.178 — 2.2× Jev's. kotoba clocked a dLLM at **846 ms vs 19–34 ms** for ModernBERT-base. Accuracy at 25–44× the latency and twice the miscalibration |
| **Pure linear/recurrent** | simplejev-rwkv holds the **bottom three slots** (0.313–0.380). Recurrent state cannot support prefix-fork cheaply. Hybrid (decider) yes; pure RWKV no |
| **Shipping untuned** | Completed `qwen3-8b-full` scores ECE **0.4252**; the four stopped raw checkpoints land at 0.366–0.493. An untuned backbone's confidence is nearly worthless, and confidence is the product |
| **Speed as the headline** | `verdict` runs at 101 dec/s with ECE 0.4622 (stopped run, but the point stands). Fast and confidently wrong is not a decision model |
| **Untracked contamination** | decider-2b, open-jev-deberta and kev-05b all self-declare it. Holdout hygiene from day one or the eval means nothing |

---

## 5. The final design — one H100, buildable

Constraint: **a single H100 80 GB for training**, Apple Silicon for prototyping. Every choice below is resolved against that.

### 5.1 Backbone — `Qwen/Qwen3.5-4B-Base`, then its instruct checkpoint

The released model fine-tunes the instruct checkpoint, `Qwen/Qwen3.5-4B`, in its own chat format (ADR-020, ADR-027); the architecture below is shared by both.

Verified from its `config.json`, not assumed:

```
num_hidden_layers   32
layer_types         24 × linear_attention  +  8 × full_attention   (full_attention_interval 4)
hidden_size         2560     head_dim 256    heads 16 / kv 4 (GQA)
max_position_emb    262144
vocab_size          248320   tie_word_embeddings true
image/video tokens  248056 / 248057         ← natively multimodal
```

Four properties fall out of that config, and each one buys a part §3 said we wanted:

| Property | What it gives us free |
| --- | --- |
| **Hybrid: only 8 of 32 layers hold K/V** | decider's persistent prefix cache (§3.3) **without pretraining a hybrid**. The other 24 layers carry conv/recurrent state, which is small and forkable |
| **262 k context** | The 512-token trap (§4) cannot happen. 32 k states are unremarkable |
| **Native image/video tokens** | reflex's multimodal states, at no extra cost |
| **GQA 16/4, tied embeddings** | Small K/V footprint per cached state — the thing we fork per question |

This is the same family decider used. Their 2B is 24 layers = **18 linear + 6 full** — exactly the "6 full-attention layers, 18 delta-net layers" their README describes. We are inheriting a validated choice, one size up.

**Why 4B and not another size.** Evidence, not preference: reflex-4b (Qwen3.5-4B) scores **0.7189** with temperature calibration; decider-2b scores **0.7033** with a full fine-tune. ~1.6 pp for 2× the parameters, and 4B still fits comfortably under LoRA. 9B doubles memory for accuracy we do not need, since **our target is calibration, not the accuracy crown**.

**Named fallback:** `Qwen3.5-2B-Base` with a full fine-tune — decider's exact proven recipe. Use it if LoRA underfits the Family-B heads.

### 5.2 Why LoRA rather than a full fine-tune

On one 80 GB card, in bf16:

| | weights | +grads | +AdamW (fp32 m, v, master) | total | left for activations |
| --- | --- | --- | --- | --- | --- |
| 4B full-FT | 8 GB | 16 GB | 48 GB | **~64 GB** | ~16 GB — too tight at 32 k |
| **4B LoRA** | 8 GB | ~0.1 GB | ~0.1 GB | **~8.2 GB** | **~70 GB** |
| 2B full-FT | 4 GB | 8 GB | 24 GB | ~32 GB | ~48 GB (the fallback) |

LoRA r32 on `q,k,v,o,gate,up,down` ≈ 25–40 M trainable parameters. **The Family-B heads are new parameters and are trained at full precision regardless** — LoRA freezing the backbone does not prevent training a new head, which is the only thing that made this choice non-obvious.

### 5.3 The readout — dual mode, and the part nobody else has

```
                       questions + option sets
                                 │
                    ┌────────────┴────────────┐
                    │      MODE ROUTER        │
                    │ every option maps to a  │
                    │ verified single token?  │
                    └────────┬───────┬────────┘
                       yes   │       │   no / long option text
                             ▼       ▼
              A: LABEL-TOKEN READOUT   B: CANDIDATE-PATH SCORING
              logits at "Answer:"      shared matching head
              over A,B,…,AA,…          + set attention over candidates
              0 added params           3.67M params, no option ceiling
                             │       │
                             └───┬───┘
                                 ▼
     raw scores → temperature per type, mode and option band → softmax
```

**The router's boundary is tokenizer-verified single-token-ness, not a fixed count.** LitJev rejects unsupported tokenizers rather than truncating, and that is the correct behaviour — but where LitJev *rejects*, we **fall through to Mode B**. That is the whole differentiator: the failure case of every Family-A implementation becomes our second mode.

**A and B must be measured against each other.** They compute distributions by different mechanisms, so:

- they get **separate fitted temperatures** — one global scalar would be wrong for at least one of them;
- on option sets where **both** are valid we measure agreement. Material disagreement means the router is a correctness hazard, not merely a capacity switch. This is an explicit eval, not an assumption.

### 5.4 Output types

| Type | Mode A | Mode B |
| --- | --- | --- |
| **Choice** | softmax over label-token logits | softmax over candidate scores |
| **Score** | `Σ(i × p_i)` over ordered levels | same, **plus an ordinality constraint** |
| **Noul** | distribution over **9 rating tokens**, `p(yes) = Σ (i/8)·p_i` | the same nine ratings as candidates (only if they are not single tokens) |

**Score under Mode B needs care that Mode A does not.** Under A the levels are unordered symbols and ordering lives in the prompt. Under B, a shared matching head has nothing forcing level *i+1* to score above level *i*, so the loss adds an explicit ordinal term (applied to Noul rows as well, whose rating scale is equally ordered).

**Noul returns a real distribution.** Nine rating tokens rather than a two-way yes/no (simple-jev, §3.4). This fixes the asymmetry FINDINGS §2 found in the real API, where `NoulAnswer` is a bare float with no `probabilities` and no `confidence`. Ours carries both, and is therefore calibratable like the other two types.

### 5.5 Caching — two layouts, trained 50/50

- **State-first** (default) — `Context … Question … Options … Answer:`. The state is the shared prefix. Serving runs one batched forward over prefix + suffix per question, which measured faster than prefilling once and forking the 8-layer K/V + conv state, because the forward is launch-bound at this size (ADR-023); the fork remains as `prefix_mode="fork"`.
- **Schema-first** (opt-in) — questions/options before the state, so the schema is a state-independent prefix cached read-only **across requests**.

Random layout per example at training time buys the choice at inference. decider's measured cost of schema-first:

| workload | schema-first cost |
| --- | --- |
| fixed label set (classification, routing, scales) | −1.5 pts (median −0.7, calibration equal) |
| options change per example | −5 pts |
| 50–219 options, or multi-thousand-token states | −5 to −24 pts |

So: **schema-first only for high-volume fixed-schema batch work; state-first everywhere else.**

### 5.6 Calibration — the differentiator, and it is cheap

Three splits, not two. Temperature is fitted on a split disjoint from **both** train and test, and **fitted per question type and per readout mode**, plus an option-count band for Choice — the three types have different distribution shapes and one global scalar under-serves at least one. Each bucket keeps whichever of a row-weighted and a family-weighted fit transfers better to held-out task families (ADR-028).

1. Train with a proper scoring rule (cross-entropy + Brier).
2. **Abstain augmentation** (decider): examples where the answer is not determinable from the state, teaching the model to spread mass rather than guess confidently.
3. Post-hoc temperature fit on the calibration split.
4. Report accuracy, NLL, Brier and **ECE** per source, with and without the temperature.

Evidence this is worth the effort: untuned Qwen3.5 backbones sit at **ECE 0.4252**; reflex, the same family plus one fitted scalar, reaches **0.0849**. Jev is **0.0764**. **A single scalar is most of the gap.**

RL with a proper-score belief reward (decider v10) comes after a supervised baseline exists, never before.

### 5.7 Training data, and the contamination rule

**This is a build requirement, not hygiene.** Every S1Bench number is only meaningful if no evaluation subset reached training; three leaderboard entries self-declare contamination and their results are compromised by it.

**All 13 S1Bench subsets are excluded from the training mixture**, not only the six that ran (ADR-009), and the guard raises at import time and again before training. decider's task registry (~95 public datasets) contains several of them. The sources actually used are listed in [TRAINING.md](TRAINING.md#the-sources).

### 5.8 The training budget, with the arithmetic visible

Assumptions stated so you can change them:

```
trainable model      4e9 params (backward cost is full-model even under LoRA)
cost per token       ~8 × N FLOPs   (6 × N fwd+bwd, +33% for grad checkpointing)
                     = 8 × 4e9      = 3.2e10 FLOP/token
dataset              200,000 question-instances × 128 tokens (measured) = 2.56e7 tokens/epoch
epochs               3                                          = 7.68e7 tokens
total compute        3.2e10 × 7.68e7                            = 2.46e18 FLOPs
H100 effective       ~400 TFLOP/s bf16 (realistic with checkpointing, not peak 990)
```

**2.46e18 / 4e14 ≈ 6.1e3 s ≈ 1.7 GPU-hours** for the full 3-epoch run.

Cross-check on step count: batch 32 gives 6,250 steps/epoch, 18,750 total.

Two corrections this arithmetic has already needed, both from [ADR-016](DECISIONS.md#adr-016--sequence-length-is-measured-and-the-budget-was-wrong-by-10x) and [ADR-017](DECISIONS.md#adr-017--batches-are-length-bucketed-and-the-budget-was-wrong-again): the 128-token mean is measured, not assumed — an earlier guess of 1,200 put this at 16 hours — and the model computes on the padded batch rectangle, not on real tokens, which cost a further 4.4× until batches were length-bucketed. The first measured run sustained ~4,200 tok/s against a ~4-hour wall clock, so treat 1.7 h as a floor.

**A full run is hours, not days.** That is the real consequence of choosing LoRA on a 4B: you can afford many full runs, which means ablations — 2B vs 4B, Mode B on vs off, temperature per-type vs global — are affordable rather than aspirational. Budget the H100 for **ablations, not for one heroic run**.

### 5.9 Build order

1. **Mode A readout on stock `Qwen3.5-4B-Base`, serving `/v1/systemone`. No training.** simplejev's evidence says this alone reaches ~0.75 macro. *Done.*
2. **Fit a temperature.** Evidence says this is ECE 0.43 → ~0.08. The cheapest point on the curve, so it comes before anything else. *Done.*
3. **Add Mode B** and the router. This is the differentiator — measure A/B agreement where both are valid. *Done; agreement unmeasured (Q2).*
4. **LoRA fine-tune** with random layout, abstain augmentation, proper scoring. *Done.*
5. **RL with a belief reward** — only once 1–4 are measured.

Steps 1–2 need no training and run on a laptop, before the H100 is touched.

### 5.10 What is evidenced and what is projected

Stated plainly, because the distinction matters:

- **Verified by me:** the Qwen3.5 config facts (hybrid layer split, context, multimodal tokens), the memory arithmetic, every S1Bench number in §2, and the architecture of each implementation surveyed.
- **Taken from others' published numbers:** the 0.7189 target (reflex), 0.7033 (decider-2b), the schema-first cost table, ECE 0.0849 vs 0.4252.
- **Measured since:** the 128-token mean and the padding factor (§5.8); that both readouts serve and train; that `levbench` on S1Bench's pinned items reproduces Jev's published per-subset numbers within 0.8 pp on all 13 subsets (FINDINGS §17).
- **Measured after training:** three 4B checkpoints trained, calibrated and scored on S1Bench against Jev through identical task files; results and what each run changed are in [FINDINGS §12–17](FINDINGS.md), including the lev-vs-Jev calibration comparison on all 13 subsets.
- **Still open:** whether Mode A and Mode B agree where both are valid (Q2).

## 6. How the harness validates this

`levbench` is instrumentation for this build, not a side project:

- Every design above serves `/v1/systemone`, so `levbench eval --base-url …` measures our model against Jev on identical questions with no code change.
- `levbench eval` reports ECE, Brier, log loss and selective accuracy — the exact axis §3.1 identifies as the one that matters and that most clones failed.
- `levbench sweep` verifies our state cache actually amortises, rather than assuming it.
- `levbench confidence` settles which statistic our confidence should be, and whether it matches Jev's.

**The target to beat is not Jev's accuracy — it is the open clones' calibration.** They sit ~1 point behind Jev on accuracy (§FINDINGS 9) and 1.6–2.2× worse on ECE. Closing the ECE gap with a fitted temperature, on a backbone we can run, is the achievable win.
