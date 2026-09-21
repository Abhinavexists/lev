# Training

Read [`ARCHITECTURE.md`](ARCHITECTURE.md) §5 for *why* the pipeline looks like this and
[`DECISIONS.md`](DECISIONS.md) for the alternatives that were rejected. This document is
the *how*.

---

## The order matters

Steps 1 and 2 need **no training at all** and deliver most of the value. Do them first.

```
1  Mode A readout on a stock checkpoint, serving /v1/systemone     no GPU training
2  Fit a temperature                                               ECE 0.43 -> ~0.08
3  Add Mode B + the router                                         the differentiator
4  LoRA fine-tune                                                  ~16 h on one H100
5  RL with a belief reward                                         only after 1-4
```

The evidence for that ordering: on S1Bench, an *untuned* 27B with a label-token readout
scores 0.7582 — one point behind Jev. The gap that matters is calibration (0.1214 vs
0.0764), and step 2 closes most of it for the cost of fitting one scalar.

---

## Step 1 — serve an untrained baseline

```bash
uv sync --extra serve
uv run lev serve --model Qwen/Qwen3.5-4B-Base --port 8000
```

Measure it with the same harness that measures Jev:

```bash
uv run levbench eval  --backend jev --base-url http://localhost:8000
uv run levbench sweep --backend jev --base-url http://localhost:8000
```

`sweep` is the one to watch: it verifies the state cache actually amortises across
questions rather than assuming it. Cost per question should fall roughly linearly with
the number of questions in a request.

---

## Step 2 — fit a temperature

Three splits, not two. The calibration split must be disjoint from **both** train and
test — `calibrate.fit()` raises on a split named `test`/`eval`/`holdout`, because
fitting on test labels produces a profile that looks excellent and means nothing.

```bash
modal run modal/app.py::calibrate --preset 4b --split calibration
```

One temperature is fitted **per `(question_type, readout_mode)` bucket**. A global
scalar under-serves at least one bucket: the three types produce differently shaped
distributions, and Mode A and Mode B produce them by different mechanisms.

A bucket with fewer than 50 samples is left unfitted at `T=1.0` (the identity) rather
than fitted on noise, and never borrows another bucket's scalar.

---

## Step 3 — the data mixture

**The contamination guard runs before anything loads.** These six are banned:

```
vitaminc-dev   massive-en-US   boolq   helpsteer2   aegis2   paws
```

```bash
uv run lev check-data sources.txt
```

It resolves aliases, so `tals/vitaminc`, `paws-x`, `google/boolq` and
`nvidia/HelpSteer2` are all caught. Contamination would *improve* our headline number
while invalidating it, which is why this raises rather than warns.

Candidate sources and their risk:

| Source | Risk |
|---|---|
| decider's registry (~95 public datasets) | **contains several of the six** — filter |
| Nimble (2,676 train / 324 test) | synthetic contrastive pairs |
| NanoJev observed-event data | good fit for the calibration objective |
| Teacher-generated states from a local 27B | safe if the teacher never sees the six |

The mixture controls three things the architecture depends on:

| Knob | Default | Why |
|---|---|---|
| `schema_first_fraction` | 0.5 | Both cache layouts must work at inference ([ADR-008](DECISIONS.md#adr-008--both-prompt-layouts-trained-5050)) |
| `abstain_fraction` | 0.1 | Teaches spreading mass instead of confident guessing — a calibration aid |
| large-option questions | some | Mode B is untrained otherwise |

**This is the hacking-phase task.** `lev.data.mixture.build_mixture` defines the
contract and enforces the guard; the per-source loaders are what you write.

---

## Step 4 — the fine-tune

```bash
make smoke                 # ALWAYS first: 0.8B, ~5 min of H100
make plan PRESET=4b        # confirm the budget
make train PRESET=4b       # ~16 h
```

### The budget, with the arithmetic visible

```
cost/token   8 x N FLOPs      (6 x N fwd+bwd, +33% for gradient checkpointing)
             8 x 4e9        = 3.2e10 FLOP/token
data         200,000 examples x 1,200 tokens x 3 epochs = 7.2e8 tokens
compute      3.2e10 x 7.2e8 = 2.30e19 FLOPs
H100         ~400 TFLOP/s sustained bf16 (not the 990 peak)
             2.30e19 / 4e14 = 5.8e4 s = 16.0 hours
cross-check  21,972 steps @ 32,768 tok/step = 2.6 s/step = 12.5k tok/s
```

`make plan` recomputes this from the config, so changing any knob shows the new cost
*before* you rent the GPU.

### Objective

Cross-entropy alone optimises the argmax and tolerates overconfidence, which is exactly
the failure we are trying to beat. So the loss is a **proper scoring rule**:

```
loss = CE  +  0.50 * Brier  +  0.25 * ordinal   (ordinal: Score under Mode B only)
```

The ordinal term exists because Mode B loses something Mode A gets free: under Mode A a
Score's levels are unordered symbols and ordering lives in the prompt text; under a
shared matching head, nothing forces level *i+1* to score above level *i*. `ordinal_mae`
is tracked as its own metric because accuracy hides this entirely.

### Presets

| Preset | Backbone | Adaptation | State | Headroom | Hours |
|---|---|---|---|---|---|
| `smoke` | Qwen3.5-0.8B-Base | LoRA r32 | 1.9 GB | 78.1 GB | minutes |
| `2b` | Qwen3.5-2B-Base | full FT | 32.0 GB | 48.0 GB | 8 |
| **`4b`** | **Qwen3.5-4B-Base** | **LoRA r32** | **8.3 GB** | **71.7 GB** | **16** |
| `9b` | Qwen3.5-9B-Base | LoRA r32 | 18.3 GB | 61.7 GB | 36 |

A 16-hour run means you can afford about a dozen. **Budget the H100 for ablations, not
one heroic run.**

---

## Step 5 — ablations worth the GPU time

Ranked by what they would actually change:

1. **Temperature per-bucket vs global** — validates ADR-006, the core claim. Cheapest.
2. **Mode A vs Mode B where both are valid** — open question Q2. If they disagree
   materially the router is a correctness bug, so this is a gate, not a nice-to-have.
3. **2B vs 4B** — is the 1.6 pp worth 2× the time?
4. **Layout 50/50 vs state-first only** — does dual-layout cost accuracy?
5. **Abstain augmentation on/off** — measured on ECE, not accuracy.

---

## What is and is not implemented

| | Status |
|---|---|
| Schema, prompt layouts, router, label codes | **done, tested** (68 tests) |
| Calibration fitting, ECE, profile I/O | **done, tested** |
| Contamination guard | **done, tested** |
| Training config + budget arithmetic | **done, verified** |
| Modal app, volumes, smoke path | written, **not yet run** |
| Decision engine (prefill, fork, readout) | **runs on Qwen3.5-4B-Base**, Mode A verified |
| Mode B head | written, **untrained** |
| Data loaders | **the hacking-phase task** |

Everything marked "not yet run" says so in its module docstring too.
