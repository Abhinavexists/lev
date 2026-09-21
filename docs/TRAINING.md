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
4  LoRA fine-tune                                                   ~2 h on one H100
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

> **Use an instruct checkpoint for the *zero-shot* baseline.** Measured on
> `Qwen3.5-4B-Base`: Choice reaches 0.958 accuracy, but Noul collapses to 0.292 —
> exactly the positive base rate, because a base model answers "yes" to every
> 9-point rating prompt. `-Base` is the right choice for the fine-tune; it is the
> wrong one for an untrained server.
> [ADR-007.](DECISIONS.md#adr-007--noul-from-nine-rating-tokens)

Measure it with the same harness that measures Jev:

```bash
uv run levbench eval  --backend lev
uv run levbench sweep --backend lev
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

```bash
make data                  # ~10 min, downloads and writes three splits
make eval-set              # export the held-out split for levbench
```

or with the knobs visible:

```bash
uv run lev data build --out data/mixture --limit-per-source 20000 --n-examples 200000
uv run lev data eval  --data data/mixture --out data/eval
```

**The contamination guard runs before anything loads**, and it covers all
**thirteen** S1Bench evaluation subsets, not just the six that executed in the
`s1-fast` run ([ADR-009](DECISIONS.md#adr-009--all-thirteen-evaluation-subsets-are-banned-from-training)):

```
ran      vitaminc-dev  massive-en-US  boolq  helpsteer2  aegis2  paws
unrun    massive-de-DE  squad2  multinli  civil_comments
         summeval-relevance  summeval-consistency  pubmedqa
```

```bash
uv run lev check-data sources.txt
```

It resolves aliases, so `tals/vitaminc`, `paws-x`, `google/boolq`,
`nvidia/HelpSteer2` and `rajpurkar/squad_v2` are all caught. Contamination would
*improve* our headline number while invalidating it, which is why this raises
rather than warns.

### The sources

Nine public classification corpora. A Jev-like model is not learning world
knowledge, it is learning to put calibrated mass on a candidate set, so the
supervision it needs is a gold label plus an explicit option set — which is
exactly what a classification corpus is.

| Source | Primitive | Options | Why it is in the mixture |
|---|---|---|---|
| `fancyzhx/ag_news` | Choice | 4 | small, well-separated options |
| `dair-ai/emotion` | Choice | 6 | overlapping options — harder to be confident |
| `fancyzhx/dbpedia_14` | Choice | 14 | mid-size option set |
| `legacy-datasets/banking77` | Choice | **77** | **Mode B** — overflows single-token codes |
| `clinc/clinc_oos` | Choice | **151** | **Mode B** at the extreme |
| `SetFit/sst5` | Score | 5 | genuinely ordered levels |
| `Yelp/yelp_review_full` | Score | 5 | ordered, and a different domain |
| `stanfordnlp/imdb` | Noul | 2 | long states |
| `cornell-movie-review-data/rotten_tomatoes` | Noul | 2 | short states |

banking77 and clinc_oos are the **only** sources whose option sets cannot fit
single-token label codes, so they are the only Mode B training signal. Weighting
them by corpus size would give them a few percent and Mode B would not learn, so
`default_weights()` gives the pair **25%** of the mixture deliberately. The
remaining 75% splits evenly across the three primitives so no readout starves.

Three candidates were dropped: `CogComp/trec` and `takala/financial_phrasebank`
are script-backed with no parquet mirror carrying their label names, and
`datasets>=5` no longer runs dataset scripts; `PolyAI/banking77` is the same
corpus as the `legacy-datasets` mirror we use, minus a loadable `ClassLabel`.

### Two failure modes the pipeline is built to prevent

**A head slice is not a sample.** Most of these corpora ship grouped by label, so
`imdb[:400]` is 400 negative reviews and `dbpedia_14[:400]` is one class out of
fourteen. `load_source` shuffles with a fixed seed before it cuts. This one is
nasty because it is invisible downstream: every split drawn from a skewed sample
is skewed identically, so the splits still agree with each other.

**A Noul's label is a rating, not a class.** A Noul is read out as nine rating
tokens and collapsed by `noul_probability`, so supervising "yes" with the raw
class `1` teaches rating 1 — which reads back as P(yes) = 0.125. The model learns
to answer *no* on every positive example while the loss looks healthy. Binary
labels are mapped to the ends of the scale, 0 and 8.

### Splits: three, and split before mixing

| Split | Share | Purpose |
|---|---|---|
| `train` | 80% | the fine-tune |
| `calibration` | 10% | fitting the temperature — never train, never test |
| `test` | 10% | the held-out eval, exported for levbench |

Assignment is a blake2b hash of a stable per-row key, not an RNG draw: the same
row lands in the same split on any machine, in any dataset order, on any re-run.
Python's `hash()` is salted per process and would not reproduce tomorrow.

Splitting happens **before** mixing. Mixing first would let one underlying row
appear in train and in test wearing two different layouts — a contamination leak
with our own data rather than S1Bench's, subtler, and flattering in the same way.

Two coverage checks run, and both are needed:

- every label *observed* must appear in train — otherwise its error rate
  measures nothing;
- every label the *question offers* must be observed at all — this is the one
  that catches an undersampled banking77, where 3 intents out of 77 would
  satisfy the first check and still be junk.

Noul is exempt from the second: it offers nine rating levels and its data
supplies two.

### The knobs

| Knob | Default | Why |
|---|---|---|
| `schema_first_fraction` | 0.5 | Both cache layouts must work at inference ([ADR-008](DECISIONS.md#adr-008--both-prompt-layouts-trained-5050)) |
| `abstain_fraction` | 0.1 | Teaches spreading mass instead of confident guessing ([ADR-012](DECISIONS.md#adr-012--abstain-means-taking-the-state-away)) |
| `limit_per_source` | 20,000 | rows sampled per source; raise it if coverage fails |

Abstain examples are built by pairing a question with a state from a *different*
source and supervising a uniform distribution. Flagging an otherwise-answerable
row `abstain=True` is worse than no augmentation: the state still determines the
answer, so the only thing learned is doubt where there should be none.

---

## Step 4 — the fine-tune

```bash
make smoke                 # ALWAYS first: 0.8B, ~5 min of H100
make plan PRESET=4b        # confirm the budget
make train PRESET=4b       # ~2 h
```

### The budget, with the arithmetic visible

```
cost/token   8 x N FLOPs      (6 x N fwd+bwd, +33% for gradient checkpointing)
             8 x 4e9        = 3.2e10 FLOP/token
data         200,000 examples x 128 tokens x 3 epochs = 7.7e7 tokens
compute      3.2e10 x 7.7e7 = 2.46e18 FLOPs
H100         ~400 TFLOP/s sustained bf16 (not the 990 peak)
             2.46e18 / 4e14 = 6.1e3 s = 1.7 hours
cross-check  18,750 steps @ 32 examples/step = 0.33 s/step

The 128 tokens is **measured**, not assumed: the mean over 1,500 rendered
prompts from the real mixture, tokenised with the Qwen3.5 tokenizer. Per source
it runs 38 (clinc_oos) to 305 (imdb); p95 is 390 and the longest seen is 1,104.
An earlier version of this file guessed 1,200 and therefore quoted 16 hours --
a 10x error, on the number you use to decide whether a run is affordable. Run
`lev plan --data data/mixture` to re-measure after changing the mixture.

At a 128-token mean the batch size matters more than the sequence cap: a batch
of 8 would make 75,000 optimiser steps and the run would be bound by step
overhead long before it was bound by FLOPs. Hence `per_device_batch = 32`.

**This arithmetic still under-predicts, and the first real run proved it.**
It assumes the model computes on the real tokens; it computes on the padded
rectangle. Batches are length-bucketed, which takes padding from 4.43x to
1.43x, and the linear-attention kernels are installed. Treat ~2 h as a floor
rather than an estimate until a full run lands — the measured figure before
those two fixes was 23 h. [ADR-017.](DECISIONS.md#adr-017--batches-are-length-bucketed-and-the-budget-was-wrong-again)
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

### Watching a run

The loop prints a flushed progress line every `log_every` steps (25 by default)
and a line per checkpoint, so `modal app logs` shows a live run rather than
nothing until it finishes:

```
training 200,000 examples x 3 epochs = 18,750 steps at batch 32 | 32.8M trainable params on cuda:0
step     25/18750    0.1%  A=2.7413  B=4.9902  lr=1.71e-05  3.14 it/s  12,861 tok/s  elapsed 0:00:08  eta 1:39:28  mem 22.4G
...
  checkpoint -> /checkpoints/4b/step-2000  (137 MB)
```

Losses are windowed **per readout mode**, not blended. The two sit at different
scales — Mode B starts near `ln(K)`, so ~5.0 for a 151-option question — and a
single average hides which one is moving.

Rate, throughput and ETA are measured over the window since the last report,
not cumulatively. Startup — weight load, allocator warmup, and a Triton JIT
compile that can run for minutes — is a one-off, and a cumulative average never
stops paying for it. On the 0.8B smoke that was 0.33 it/s reported against a
2.50 it/s reality, and an ETA wrong by the same factor. **Read the second
progress line, not the first.**

Throughput counts the tokens actually fed to the model, from the attention
mask, not `avg_tokens_per_example × batch`. That constant was wrong by 10×
once (ADR-016); a throughput readout derived from it would have agreed with the
mistake instead of exposing it.

Flushing matters more than it sounds: Python block-buffers stdout when it is
not a tty, so an unflushed `print` shows nothing for two hours and then
everything at once. Every write in `ProgressLog` is flushed.

`history.json` is rewritten at every checkpoint, not only at the end, so a run
that dies at step 17,000 still leaves its loss curve behind.

### Resuming

```bash
modal run modal/app.py::train --preset 4b --resume /checkpoints/4b
```

`--resume` takes the preset's output directory and resolves the newest
`step-N` inside it. It restores **weights only** — the LoRA adapter and the
Mode B head — so a resumed run repeats the warmup and restarts its cosine
schedule. The optimiser moments are not carried over. At ~2 h per run that
trade is fine; it would not be at 16.

Resuming without the `mode_b_head.pt` beside the adapter is a hard error rather
than a silent reinitialisation, because a freshly initialised head looks exactly
like a trained one until you read the Mode B numbers.

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

## Step 4b — serve what you trained

```bash
uv run lev serve --checkpoint checkpoints/lev-4b --model Qwen/Qwen3.5-4B-Base
uv run levbench eval --backend lev --tasks data/eval
```

`--checkpoint` takes the output directory and resolves the newest `step-N`
inside it. It loads the base model, applies the LoRA adapter on top, picks up
`mode_b_head.pt` if it is there, and uses the `calibration.json` sitting beside
the weights unless `--calibration` overrides it. `GET /health` reports which
checkpoint was resolved, whether a Mode B head loaded, and whether the profile
is calibrated — check it before reading any number off an eval.

A checkpoint directory is an *adapter*, not a model. Pointing
`AutoModelForCausalLM` at one does not work, which is why `--model` still names
the base.

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

## What has actually been run

A full pass on `Qwen/Qwen3.5-0.8B-Base`, on CPU, against a mixture built from
all nine live sources:

```
train      30 steps, 666 s          mode A: 20 steps   mode B: 10 steps
                                    every loss finite, both modes exercised
checkpoint step-30/                 adapter_model.safetensors  mode_b_head.pt
                                    tokenizer.json  adapter_config.json
resume     load_checkpoint()        one "default" adapter, 192 LoRA tensors,
                                    head weights restored
serve      base + adapter + head    Choice / Score / Noul / 40-option Mode B
                                    all answered, output_tokens = 0
```

**The loss did not decrease**, and that is the honest reading: 30 steps at
batch 2 is noise, not learning. Mode A went 11.72 → 9.29 and Mode B 5.23 → 5.07
between the first and last thirds, which is within the step-to-step spread.
What the run demonstrates is that the path executes and produces a usable
checkpoint — not that training works. Mode B sitting at ~5.0 is a useful sanity
check on its own: `ln(151) = 5.02`, so the untrained head is exactly at chance.

Two bugs came out of running it that no test had caught:

- **every loss was `nan`.** Padded candidate slots carry a `-inf` logit and a
  zero target probability, so the plain product is `0 * -inf`. One padded slot
  poisons the batch mean, and backward still runs, so the symptom is a nan loss
  rather than a crash.
- **SIGSEGV part-way through loading weights.** `device_map="auto"` had
  accelerate dispatching to MPS. It reads as a corrupt download rather than a
  placement bug.

---

## What is and is not implemented

| | Status |
|---|---|
| Schema, prompt layouts, router, label codes | **done, tested** |
| Calibration fitting, ECE, profile I/O | **done, tested** |
| Contamination guard (13 subsets) | **done, tested** |
| Training config + budget arithmetic | **done, verified** |
| Data loaders, splits, mixture | **done — 9 sources load, splits verified** |
| Held-out eval export | **done — 2,760 items, ±4pts** |
| Collator, both readouts, loss | **done, exercised on a real backbone** |
| Training loop, checkpointing | **done — runs, writes adapter + head** |
| Modal app, image, volumes | **image builds on Modal** |
| Modal `download` / `build_data` / `smoke` | **run green on Modal** |
| Modal `train` | **runs on an H100**; no full run completed |
| Modal `serve` / `calibrate` | written, **not yet run remotely** |
| Decision engine (prefill, fork, readout) | **runs on Qwen3.5-4B-Base**, Mode A verified |
| Mode B head | **trains**, untrained at scale |

Everything marked "not yet run" says so in its module docstring too. No trained
checkpoint has been evaluated, so no accuracy or calibration number here comes from
a model this repository produced.
