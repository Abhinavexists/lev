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

Twenty-three public corpora, chosen so that the *question* carries information
the state does not. The first mixture had nine sources with one fixed
instruction each, and the model learned to ignore the instruction entirely --
the state identified the answer set on its own. See ADR-020.

| Source | Primitive | Options | What it adds |
|---|---|---|---|
| `fancyzhx/ag_news` | Choice | 4 | small, well-separated options |
| `dair-ai/emotion` | Choice | 6 | overlapping options |
| `fancyzhx/dbpedia_14` | Choice | 14 | mid-size option set |
| `ehovy/race` | Choice | 4 per row | passage + question; options differ every row |
| `tau/commonsense_qa` | Choice | 5 per row | commonsense QA |
| `allenai/sciq` | Choice | 4 per row | science QA with supporting context |
| `allenai/openbookqa` | Choice | 4 per row | fact + question |
| `allenai/ai2_arc` (Easy) | Choice | 3–5 per row | grade-school science |
| `stanfordnlp/snli`, `facebook/anli` | Choice | 3 | NLI -- the answer is a relation between two fields |
| `benayas/snips` | Choice | 7 | a second, small intent taxonomy |
| `legacy-datasets/banking77` | Choice | **77** | **Mode B** |
| `clinc/clinc_oos` | Choice | **151** | **Mode B** at the extreme |
| `SetFit/sst5`, `Yelp/yelp_review_full` | Score | 5 | ordered sentiment levels |
| `openbmb/UltraFeedback` | Score | 5 | helpfulness rubric over instruction + response |
| `stanfordnlp/imdb`, `cornell-movie-review-data/rotten_tomatoes` | Noul | 2 | sentiment, long and short states |
| `SetFit/mrpc`, `SetFit/qqp` | Noul | 2 | paraphrase over sentence pairs |
| `lmsys/toxic-chat`, `toxigen/toxigen-data` | Noul | 2 | **yes = toxic**: the bad outcome is the yes |
| `PKU-Alignment/BeaverTails` | Noul | 2 | safety of a response; yes = safe, negated half the time |

Every source carries paraphrased instructions and every Noul source a negation
that flips the target, so no polarity is constant. Choice option sets are
subsampled, shuffled and sometimes stripped of descriptions in the train split.
The held-out splits keep each source's canonical question, so the exported eval
set and the fitted temperatures describe what is actually served.

banking77 and clinc_oos are the sources over the Mode A cap of 26 options
(`labels.LABEL_OPTION_CAP`), so they are the Mode B training signal;
`default_weights()` gives the pair **25%** of the mixture deliberately, and
subsampling never takes them below 27 options. The remaining 75% splits evenly
across the three primitives.

Dropped, with reasons: `CogComp/trec`, `takala/financial_phrasebank`,
`allenai/cosmos_qa`, `allenai/social_i_qa`, `wics/strategy-qa` and
`mteb/mtop_intent` are script-backed and `datasets>=5` refuses them.
`DeepPavlov/hwu64` loads, and is the 64-intent schema MASSIVE inherited via
SLURP -- training on it would make massive-en-US a seen taxonomy, so the guard
now blocks it.

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

A run picks up where it stopped. `make train` resumes from the newest `step-N`
in the preset's checkpoint directory; `FRESH=1` starts over, `RESUME=path`
names a checkpoint explicitly. Each checkpoint carries the optimiser moments,
the schedule position, the step and epoch, and the RNG state that reproduces
the epoch's data order, so a resumed run continues at step N through the
batches it had not yet seen, on the learning rate it had reached, and
`history.json` extends rather than restarts. A preemption costs at most
`checkpoint_every` steps -- 2,000, about half an hour on the 4B preset.

`FRESH=1` also moves the previous run's `step-*`, `history.json` and
`calibration.json` into a `superseded-<utc>` directory beside them: left in
place, a preemption's auto-resume would take the stale highest step, and
`serve` would pick up the old temperatures for the new weights.

A checkpoint written before this existed carries weights only; resuming one
restores the adapter and head and starts the optimiser and schedule fresh,
which is what every resume did before ([ADR-021](DECISIONS.md#adr-021--a-checkpoint-carries-the-training-state-and-a-run-resumes-by-default)).

`smoke` always starts fresh: a smoke test that resumed the previous smoke would
skip the very steps it exists to exercise.

### Presets

| Preset | Backbone | Adaptation | State | Headroom | Hours |
|---|---|---|---|---|---|
| `smoke` | Qwen3.5-0.8B-Base | LoRA r32 | 1.9 GB | 78.1 GB | minutes |
| `2b` | Qwen3.5-2B-Base | full FT | 32.0 GB | 48.0 GB | 8 |
| `4b` | Qwen3.5-4B-Base | LoRA r32 | 8.3 GB | 71.7 GB | 16 |
| **`4b-instruct`** | **Qwen3.5-4B** (instruct) | **LoRA r32, lr 5e-5** | **8.3 GB** | **71.7 GB** | **16** |
| `9b` | Qwen3.5-9B-Base | LoRA r32 | 18.3 GB | 61.7 GB | 36 |

`4b-instruct` is the recommended start after ADR-020: frozen, the instruct
checkpoint scores 0.719 on S1Bench (reflex-4b); the `4b` fine-tune scored 0.489.

A 16-hour run means you can afford about a dozen. **Budget the H100 for ablations, not
one heroic run.**

---

## Step 4b — serve what you trained

```bash
uv run lev serve --checkpoint checkpoints/lev-4b --model Qwen/Qwen3.5-4B-Base
uv run levbench eval --backend lev --tasks data/eval
```

`--checkpoint` takes the output directory and resolves the newest `step-N`
inside it, a flat release directory, or a Hub id. It loads the base model,
applies the LoRA adapter on top, picks up `mode_b_head.pt` if it is there, and
uses the `calibration.json` beside the weights unless `--calibration`
overrides it. `GET /health` reports which checkpoint was resolved, whether a
Mode B head loaded, whether the profile is calibrated, the Noul readout and the
option cap — check it before reading any number off an eval.

A checkpoint directory is an *adapter*, not a model. Pointing
`AutoModelForCausalLM` at one does not work, which is why `--model` names the
base — except for a packaged release, whose `lev_release.json` names it.

### Releasing the weights

```bash
make release PRESET=4b-instruct                 # -> /checkpoints/releases/4b-instruct on the volume
make weights RELEASE=4b-instruct                # -> weights/4b-instruct locally
make publish RELEASE=4b-instruct REPO=org/name  # -> huggingface.co/org/name (HF_TOKEN)
```

`lev release build` copies adapter, head, tokenizer and calibration into one
directory with a manifest and a model card; the optimiser state is left
behind. `lev serve --checkpoint org/name` downloads and serves it.

### On Modal: which checkpoint, and `serve` vs `deploy`

`make deploy PRESET=4b-instruct` serves that preset's newest checkpoint; the
preset travels to the container as `LEV_SERVE_PRESET`. `LEV_SERVE_MODEL=Qwen/Qwen3.5-4B make deploy`
serves that model frozen — no adapter, binary Noul — the zero-shot baseline.

`modal serve` is an ephemeral dev server. So is every `modal run` of this app
file, and each one registers the `serve` web function under the same `-dev`
label — so running an export or a diagnostic while the dev server is up steals
its URL, and the URL returns 404 when the run exits. `make deploy` gives the
server a stable URL that runs cannot take.

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
| Data loaders, splits, mixture | **done — 23 sources, augmentation on the train split** |
| Held-out eval export | **done — 2,760 items, ±4pts** |
| Collator, both readouts, loss | **done, exercised on a real backbone** |
| Training loop, checkpointing | **done — runs, writes adapter + head** |
| Modal app, image, volumes | **image builds on Modal** |
| Modal `download` / `build_data` / `smoke` | **run green on Modal** |
| Modal `train` / `calibrate` / `evaluate` | **complete 4B run, calibrated and scored** ([ADR-018](DECISIONS.md#adr-018--what-the-first-trained-checkpoint-actually-shows)) |
| Modal `serve` | **runs**; scored on S1Bench over HTTP ([FINDINGS §12](FINDINGS.md)) |
| Decision engine (prefill, fork, readout) | **runs**; two-order averaging, binary Noul, option cap (ADR-020) |
| Mode B head | **trains and serves**; transfer to an unseen taxonomy untested (Q10) |
| S1Bench harness (`lev s1bench export`) | **done, validated against Jev's own numbers** |

Everything marked "not yet run" says so in its module docstring too. No trained
checkpoint has been evaluated, so no accuracy or calibration number here comes from
a model this repository produced.
