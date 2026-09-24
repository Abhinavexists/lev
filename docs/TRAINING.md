# Training

Read [`ARCHITECTURE.md`](ARCHITECTURE.md) §5 for _why_ the pipeline looks like this and [`DECISIONS.md`](DECISIONS.md) for the alternatives that were rejected. This document is the _how_.

---

## The order matters

Steps 1 and 2 need **no training at all** and deliver most of the value. Do them first.

```text
1  Mode A readout on a stock checkpoint, serving /v1/systemone     no GPU training
2  Fit a temperature                                               ECE 0.43 -> ~0.08
3  Add Mode B + the router                                         the differentiator
4  LoRA fine-tune                                                   ~2 h on one H100
5  RL with a belief reward                                         only after 1-4
```

The evidence for that ordering: on S1Bench, an _untuned_ 27B with a label-token readout scores 0.7582 — one point behind Jev. The gap that matters is calibration (0.1214 vs 0.0764), and step 2 closes most of it for the cost of fitting one scalar.

---

## Step 1 — serve an untrained baseline

```bash
uv sync --extra serve
uv run lev serve --model Qwen/Qwen3.5-4B --prompt-style chat --port 8000
```

> **Use the instruct checkpoint.** Measured on `Qwen3.5-4B-Base`: Choice reaches 0.958 accuracy, but Noul collapses to 0.292 — exactly the positive base rate, because a base model answers "yes" to every 9-point rating prompt ([ADR-007](DECISIONS.md#adr-007--noul-from-nine-rating-tokens)). The instruct checkpoint is also the better starting point for the fine-tune (ADR-020).

Measure it with the same harness that measures Jev:

```bash
uv run levbench eval  --backend lev
uv run levbench sweep --backend lev
```

`sweep` is the one to watch: it verifies the state cache actually amortises across questions rather than assuming it. Cost per question should fall roughly linearly with the number of questions in a request.

---

## Step 2 — fit a temperature

Three splits, not two. The calibration split must be disjoint from **both** train and test — `calibrate.fit()` raises on a split named `test`/`eval`/`holdout`, because fitting on test labels produces a profile that looks excellent and means nothing.

```bash
modal run modal/app.py::calibrate --preset 4b-instruct --split calibration
```

One temperature is fitted **per `(question_type, readout_mode)` bucket**, and per option-count band for Choice. A global scalar under-serves at least one bucket: the three types produce differently shaped distributions, and Mode A and Mode B produce them by different mechanisms. Each bucket is fitted two ways (every row weighted equally, or every task family weighted equally) and keeps whichever transfers better to a held-out family ([ADR-028](DECISIONS.md#adr-028--skip-split-label-codes-when-serving-calibrate-for-families-the-model-has-not-seen)).

A bucket with fewer than 50 samples is left unfitted at `T=1.0` (the identity) rather than fitted on noise, and never borrows another bucket's scalar.

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

**The contamination guard runs before anything loads**, and it covers all **thirteen** S1Bench evaluation subsets, not just the six that executed in the `s1-fast` run ([ADR-009](DECISIONS.md#adr-009--all-thirteen-evaluation-subsets-are-banned-from-training)):

```
ran      vitaminc-dev  massive-en-US  boolq  helpsteer2  aegis2  paws
unrun    massive-de-DE  squad2  multinli  civil_comments
         summeval-relevance  summeval-consistency  pubmedqa
```

```bash
uv run lev check-data sources.txt
```

It resolves aliases, so `tals/vitaminc`, `paws-x`, `google/boolq`, `nvidia/HelpSteer2` and `rajpurkar/squad_v2` are all caught. Contamination would _improve_ the headline number while invalidating it, which is why this raises rather than warns.

### The sources

Twenty-nine sources over 26 public datasets, chosen so that the _question_ carries information the state does not. The first mixture had nine sources with one fixed instruction each, and the model learned to ignore the instruction entirely -- the state identified the answer set on its own. See ADR-020.

| Source                                                          | Primitive | Options     | What it adds                                                         |
| --------------------------------------------------------------- | --------- | ----------- | -------------------------------------------------------------------- |
| `fancyzhx/ag_news`                                              | Choice    | 4           | small, well-separated options                                        |
| `dair-ai/emotion`                                               | Choice    | 6           | overlapping options                                                  |
| `fancyzhx/dbpedia_14`                                           | Choice    | 14          | mid-size option set                                                  |
| `ehovy/race`                                                    | Choice    | 4 per row   | passage + question; options differ every row                         |
| `tau/commonsense_qa`                                            | Choice    | 5 per row   | commonsense QA                                                       |
| `allenai/sciq`                                                  | Choice    | 4 per row   | science QA with supporting context                                   |
| `allenai/openbookqa`                                            | Choice    | 4 per row   | fact + question                                                      |
| `allenai/ai2_arc` (Easy)                                        | Choice    | 3–5 per row | grade-school science                                                 |
| `stanfordnlp/snli`, `facebook/anli`                             | Choice    | 3           | NLI -- the answer is a relation between two fields                   |
| `pietrolesci/nli_fever`                                         | Choice    | 3           | claim + evidence → SUPPORTS / REFUTES / NOT ENOUGH INFO              |
| `benayas/snips`                                                 | Choice    | 7           | a second, small intent taxonomy                                      |
| `legacy-datasets/banking77`                                     | Choice    | **77**      | **Mode B**                                                           |
| `clinc/clinc_oos`                                               | Choice    | **151**     | **Mode B** at the extreme                                            |
| `SetFit/sst5`, `Yelp/yelp_review_full`                          | Score     | 5           | ordered sentiment levels                                             |
| `openbmb/UltraFeedback`                                         | Score     | 5           | helpfulness rubric over instruction + response                       |
| `stanfordnlp/imdb`, `cornell-movie-review-data/rotten_tomatoes` | Noul      | 2           | sentiment, long and short states                                     |
| `SetFit/mrpc`, `SetFit/qqp`                                     | Noul      | 2           | paraphrase, plus word-swapped negatives so overlap is not the answer |
| `tasksource/parade`                                             | Noul      | 2           | paraphrase between high-overlap definitions                          |
| `ChilleD/StrategyQA`                                            | Noul      | 2           | yes/no needing the given facts                                       |
| race / sciq / openbookqa as yes/no                              | Noul      | 2           | "is the proposed answer correct?" -- yes on a factual axis           |
| `lmsys/toxic-chat`, `toxigen/toxigen-data`                      | Noul      | 2           | **yes = toxic**: the bad outcome is the yes                          |
| `PKU-Alignment/BeaverTails`                                     | Noul      | 2           | safety of a response; yes = safe, negated half the time              |

Every source carries paraphrased instructions and every Noul source a negation that flips the target, so no polarity is constant. Choice option sets are subsampled, shuffled and sometimes stripped of descriptions in the train split. The test split keeps each source's canonical question, so the exported eval set has one question per source; the calibration split varies option sets (not wording) so every option-count band has rows to fit on.

banking77 and clinc_oos are the large-taxonomy sources (over 26 options, `labels.LABEL_OPTION_CAP`); `default_weights()` gives the pair **25%** of the mixture. Half their rows keep the full option set -- that is the Mode B head's training data -- and the other half are cut to anywhere from 15 options up, so the label-token readout also trains on large lettered sets. Routing follows the tokenizer in training as in serving (ADR-025/026). The remaining 75% splits evenly across the three primitives.

Dropped, with reasons: `CogComp/trec`, `takala/financial_phrasebank`, `allenai/cosmos_qa`, `allenai/social_i_qa`, `wics/strategy-qa` and `mteb/mtop_intent` are script-backed and `datasets>=5` refuses them. `DeepPavlov/hwu64` loads, and is the 64-intent schema MASSIVE inherited via SLURP -- training on it would make massive-en-US a seen taxonomy, so the guard blocks it.

### Two failure modes the pipeline is built to prevent

**A head slice is not a sample.** Most of these corpora ship grouped by label, so `imdb[:400]` is 400 negative reviews and `dbpedia_14[:400]` is one class out of fourteen. `load_source` shuffles with a fixed seed before it cuts; otherwise every split drawn from the skewed sample is skewed identically and nothing downstream notices.

**A Noul's label is a rating, not a class.** A Noul is read out as nine rating tokens and collapsed by `noul_probability`, so supervising "yes" with the raw class `1` teaches rating 1, which reads back as P(yes) = 0.125. Binary labels are mapped to the ends of the scale, 0 and 8.

### Splits: three, and split before mixing

| Split         | Share | Purpose                                           |
| ------------- | ----- | ------------------------------------------------- |
| `train`       | 80%   | the fine-tune                                     |
| `calibration` | 10%   | fitting the temperature — never train, never test |
| `test`        | 10%   | the held-out eval, exported for levbench          |

Assignment is a blake2b hash of a per-row key (source, position within the source, text), not an RNG draw, so it reproduces on any machine and re-run for the same row order; a re-ordered corpus re-splits. Python's `hash()` is salted per process.

Splitting happens **before** mixing. Mixing first would let one underlying row appear in train and in test under two different layouts, a leak that flatters the result.

Two coverage checks run, and both are needed:

- every label _observed_ must appear in train — otherwise its error rate measures nothing;
- every label the _question offers_ must be observed at all — this is the one that catches an undersampled banking77, where 3 intents out of 77 would satisfy the first check and still be junk.

Noul is exempt from the second: it offers nine rating levels and its data supplies two.

### The knobs

| Knob                    | Default | Why                                                                                                                         |
| ----------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------- |
| `schema_first_fraction` | 0.5     | Both cache layouts must work at inference ([ADR-008](DECISIONS.md#adr-008--both-prompt-layouts-trained-5050))               |
| `abstain_fraction`      | 0.1     | Teaches spreading mass instead of confident guessing ([ADR-012](DECISIONS.md#adr-012--abstain-means-taking-the-state-away)) |
| `limit_per_source`      | 20,000  | rows sampled per source; raise it if coverage fails                                                                         |

Abstain examples are built by pairing a question with a state from a _different_ source and supervising a uniform distribution. Flagging an otherwise-answerable row `abstain=True` is worse than no augmentation: the state still determines the answer, so the only thing learned is doubt where there should be none.

---

## Step 4 — the fine-tune

```bash
make smoke                          # always first: 0.8B, ~5 min of H100
make plan PRESET=4b-instruct        # confirm the budget
make train PRESET=4b-instruct       # ~2 h
```

### The budget, with the arithmetic visible

```text
cost/token   8 x N FLOPs      (6 x N fwd+bwd, +33% for gradient checkpointing)
             8 x 4e9        = 3.2e10 FLOP/token
data         200,000 examples x 128 tokens x 3 epochs = 7.7e7 tokens
compute      3.2e10 x 7.7e7 = 2.46e18 FLOPs
H100         ~400 TFLOP/s sustained bf16 (not the 990 peak)
             2.46e18 / 4e14 = 6.1e3 s = 1.7 hours
cross-check  18,750 steps @ 32 examples/step = 0.33 s/step
```

The 128 tokens is **measured** (a 121-token mean over 1,500 rendered prompts from the real mixture under the Qwen3.5 tokenizer, rounded up). Per source it runs 38 (clinc_oos) to 305 (imdb); p95 is 390 and the longest seen is 1,104. A guessed 1,200 once put the estimate at 16 hours (ADR-016). Run `lev plan --data data/mixture` to re-measure after changing the mixture.

At a 128-token mean the batch size matters more than the sequence cap: a batch of 8 would make 75,000 optimiser steps and the run would be bound by step overhead long before it was bound by FLOPs. Hence `per_device_batch = 32`.

This arithmetic assumes the model computes on the real tokens; it computes on the padded rectangle. Batches are length-bucketed, which takes padding from 4.43x to 1.43x, and the linear-attention kernels are installed, so treat ~2 h as a floor. [ADR-017.](DECISIONS.md#adr-017--batches-are-length-bucketed-and-the-budget-was-wrong-again)

`make plan` recomputes this from the config, so changing any knob shows the new cost _before_ you rent the GPU.

### Objective

Cross-entropy alone optimises the argmax and tolerates overconfidence. So the loss is a **proper scoring rule**:

```text
loss = CE  +  0.50 * Brier  +  0.25 * ordinal   (ordinal: Score and Noul rows)
```

The ordinal term weights probability mass by its squared distance from the true level, so an ordered scale does not treat every wrong level alike: without it, Noul rating 4 is as wrong as rating 0 when the truth is 8, and under Mode B nothing forces Score level _i+1_ to score above level _i_.

### Watching a run

The loop prints a flushed progress line every `log_every` steps (25 by default) and a line per checkpoint, so `modal app logs` shows a live run rather than nothing until it finishes:

```text
training 200,000 examples x 3 epochs = 18,750 steps at batch 32 | 32.8M trainable params on cuda:0
step     25/18750    0.1%  A=2.7413  B=4.9902  lr=1.71e-05  3.14 it/s  12,861 tok/s  elapsed 0:00:08  eta 1:39:28  mem 22.4G
...
  checkpoint -> /checkpoints/4b/step-2000  (137 MB)
```

Losses are windowed **per readout mode**, not blended. The two sit at different scales — Mode B starts near `ln(K)`, so ~5.0 for a 151-option question — and a single average hides which one is moving.

Rate, throughput and ETA are measured over the window since the last report, not cumulatively: startup (weight load, a Triton JIT compile of minutes) made a cumulative rate read 0.33 it/s against 2.50 on the 0.8B smoke. **Read the second progress line, not the first.**

Throughput counts real tokens from the attention mask, not `avg_tokens_per_example × batch`, so it can contradict a wrong plan (ADR-016). Every write is flushed: Python block-buffers stdout when it is not a tty.

`history.json` is rewritten at every checkpoint, not only at the end, so a run that dies at step 17,000 still leaves its loss curve behind.

### Resuming

A run picks up where it stopped. `make train` resumes from the newest `step-N` in the preset's checkpoint directory; `FRESH=1` starts over, `RESUME=path` names a checkpoint explicitly. Each checkpoint carries the optimiser moments, the schedule position, the step and epoch, and the RNG state that reproduces the epoch's data order, so a resumed run continues at step N through the batches it had not yet seen, on the learning rate it had reached, and `history.json` extends rather than restarts. A preemption costs at most `checkpoint_every` steps -- 2,000, about half an hour on the 4B preset.

`FRESH=1` also moves the previous run's `step-*`, `history.json` and `calibration.json` into a `superseded-<utc>` directory beside them: left in place, a preemption's auto-resume would take the stale highest step, and `serve` would pick up the old temperatures for the new weights.

A weights-only checkpoint (from before [ADR-021](DECISIONS.md#adr-021--a-checkpoint-carries-the-training-state-and-a-run-resumes-by-default)) restores the adapter and head and starts the optimiser and schedule fresh. `smoke` always starts fresh, so it never skips the steps it tests.

### Presets

| Preset            | Backbone                                    | Adaptation            | State      | Headroom    | Est. hours |
| ----------------- | ------------------------------------------- | --------------------- | ---------- | ----------- | ---------- |
| `smoke`           | Qwen3.5-0.8B-Base                           | LoRA r32              | 1.9 GB     | 78.1 GB     | minutes    |
| `2b`              | Qwen3.5-2B-Base                             | full FT               | 32.0 GB    | 48.0 GB     | 0.9        |
| `4b`              | Qwen3.5-4B-Base                             | LoRA r32              | 8.3 GB     | 71.7 GB     | 1.7        |
| **`4b-instruct`** | **Qwen3.5-4B** (instruct), **chat prompts** | **LoRA r32, lr 5e-5** | **8.3 GB** | **71.7 GB** | **1.7**    |
| `9b`              | Qwen3.5-9B-Base                             | LoRA r32              | 18.3 GB    | 61.7 GB     | 3.8        |

Hours are `make plan`'s floor estimates. `4b-instruct` is the released preset (ADR-020): reflex-4b (these instruct weights plus one temperature) scores 0.719 on S1Bench, where the `4b` Base fine-tune scored 0.489. It trains in the `chat` prompt style, worth 5.7 points frozen over `plain` (ADR-027); the style is recorded in the release manifest and applied when serving. The Base presets stay `plain`.

**Budget the H100 for ablations, not one heroic run.**

---

## Step 4b — serve what you trained

```bash
uv run lev serve --checkpoint checkpoints/lev-instruct --model Qwen/Qwen3.5-4B --prompt-style chat
uv run levbench eval --backend lev --tasks data/eval
```

Or in-process: `lev.load("checkpoints/lev-instruct", model_id="Qwen/Qwen3.5-4B", prompt_style="chat")`.

`--checkpoint` takes the output directory and resolves the newest `step-N` inside it, a flat release directory, or a Hub id. It loads the base model, applies the LoRA adapter on top, picks up `mode_b_head.pt` if it is there, and uses the `calibration.json` beside the weights unless `--calibration` overrides it. `GET /health` reports which checkpoint was resolved, whether a Mode B head loaded, whether the profile is calibrated, the Noul readout and the option cap — check it before reading any number off an eval.

A checkpoint directory is an _adapter_, not a model, so `--model` and `--prompt-style` name the base and format it was trained with — except for a packaged release, whose `lev_release.json` records both.

### Releasing the weights

```bash
make release PRESET=4b-instruct                 # -> /checkpoints/releases/4b-instruct on the volume
make weights RELEASE=4b-instruct                # -> weights/4b-instruct locally
make publish RELEASE=4b-instruct REPO=org/name  # -> huggingface.co/org/name (HF_TOKEN)
```

`lev release build` copies adapter, head, tokenizer and calibration into one directory with a manifest and a model card; the optimiser state is left behind. `lev serve --checkpoint org/name` downloads and serves it.

### On Modal: which checkpoint, and `serve` vs `deploy`

`make deploy PRESET=4b-instruct` serves that preset's newest checkpoint; the preset travels to the container as `LEV_SERVE_PRESET`. `LEV_SERVE_MODEL=Qwen/Qwen3.5-4B make deploy` serves that model frozen — no adapter, binary Noul — the zero-shot baseline.

`modal serve` is an ephemeral dev server. So is every `modal run` of this app file, and each one registers the `serve` web function under the same `-dev` label — so running an export or a diagnostic while the dev server is up steals its URL, and the URL returns 404 when the run exits. `make deploy` gives the server a stable URL that runs cannot take.

---

## Step 5 — ablations worth the GPU time

Ranked by what they would actually change:

1. **Temperature per-bucket vs global** — validates ADR-006, the core claim. Cheapest.
2. **Mode A vs Mode B where both are valid** — open question Q2. If they disagree materially the router is a correctness bug, so this is a gate, not a nice-to-have.
3. **2B vs 4B** — is the 1.6 pp worth 2× the time?
4. **Layout 50/50 vs state-first only** — does dual-layout cost accuracy?
5. **Abstain augmentation on/off** — measured on ECE, not accuracy.

---

## What has been run

Three 4B runs have been trained, calibrated and scored on S1Bench; what each run changed and measured is in [FINDINGS §12–17](FINDINGS.md), and the component status table is in [STATUS.md](STATUS.md).
