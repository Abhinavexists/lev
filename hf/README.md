---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
base_model_relation: adapter
library_name: peft
pipeline_tag: zero-shot-classification
inference: false
language:
  - en
tags:
  - lev
  - system-one
  - decision-model
  - calibrated-decisions
  - classification
  - routing
  - moderation
  - lora
---

# lev

**A calibrated System One decision model.** You give it a **state** (text, a ticket, an email, or JSON) and a set of **typed questions**. It returns typed answers with calibrated probabilities, all in one forward pass (about 69 ms on an H100). It never generates text, so there is nothing to parse and no label outside your option set.

| Question | You give | You get |
|---|---|---|
| `noul` | a yes/no question | `noul` = p(yes) |
| `choice` | instructions + options (name → description or `null`) | `choice`, `probabilities`, `confidence` |
| `score` | instructions + 2–10 ordered levels | `score` (expected level), `probabilities`, `confidence` |

The request and response contract matches TypeSafe's `POST /v1/systemone`.
Code written for the TypeSafe SDK works against lev-4b once you change the
base URL. It is built for routing, moderation, intent detection, triage,
grading, and verifying LLM output.

## Installation

```bash
pip install "lev[serve] @ git+https://github.com/Abhinavexists/lev#subdirectory=packages/lev"
```

This needs Python 3.12 or newer and, for real-time use, a CUDA GPU. The `serve` extra installs torch, transformers, peft, and the HTTP server. The first load downloads the base model (Qwen/Qwen3.5-4B, about 8 GB) and this adapter (about 200 MB).

## Quickstart

```python
import lev

model = lev.load("interfaze-ai/lev")

state = "Hi, I was charged twice for my order #4471 and I want a refund."
questions = {
    "intent": {
        "type": "choice",
        "instructions": "What does the customer want?",
        "criteria": {
            "refund": "wants money back",
            "cancel": "wants to cancel an order",
            "track": "wants to know where an order is",
            "other": "anything else",
        },
    },
    "urgent": {"type": "noul", "instructions": "Does this need a human within the hour?"},
    "frustration": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["calm", "mildly annoyed", "annoyed", "angry"],
    },
}

result = model.system_one(state, questions)
print(result.answers["intent"].choice)  # refund
print(result.answers["intent"].probabilities)
# {'refund': 0.84, 'cancel': 0.094, 'track': 0.012, 'other': 0.054}
print(result.answers["urgent"].noul)  # 0.43
print(result.answers["frustration"].score)  # 1.57, between "mildly annoyed" and "annoyed"
print(result.usage.output_tokens)  # 0
```

These are real outputs from this checkpoint. The answers are objects, and `result.model_dump()` gives the same JSON the HTTP server returns.

All the questions share one forward pass, so asking three questions costs about the same as asking one. `lev.load` reads `lev_release.json` from this repository to find the base model and the prompt format the adapter was trained with. It also applies the shipped calibration and loads the matching head, so there is nothing to configure.

Because the probabilities are calibrated, you can gate on them. For example, act automatically above 0.9 and send anything lower to a person.

## Self-hosting: Jev-compatible HTTP server

```bash
lev serve --checkpoint interfaze-ai/lev-4b --host 0.0.0.0 --port 8000
```

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "The package arrived crushed and the screen is cracked.",
  "questions": {"damaged": {"type": "noul", "instructions": "Was the item damaged?"}}
}'
```

Existing TypeSafe clients work once you point them at the server:

```python
from typesafe_sdk import Choice, Noul, TypeSafeClient

# The first request after startup compiles kernels; allow more than the default 10 s.
client = TypeSafeClient(base_url="http://localhost:8000", api_key="local", timeout=60)
response = client.system_one(
    state={"ticket": "The app crashes every time I open settings."},
    questions={
        "team": Choice(
            instructions="Which team owns this?",
            criteria={"billing": "payments, refunds", "technical": "bugs, crashes", "other": None},
        ),
        "bug": Noul(instructions="Is this a bug report?"),
    },
)
print(response.answers["team"].choice, response.answers["bug"].noul)  # technical 0.92
```

`GET /health` reports the loaded checkpoint, whether calibration is active, and the routing settings. The server batches every question in a request into one forward pass, accepts concurrent requests, and returns 422 with the reason for a malformed question.

## How it works

- **Backbone:** Qwen/Qwen3.5-4B, adapted with LoRA (r=32, α=64) on the q/k/v/o attention and MLP projections.
- **Label-token readout.** Each option gets a short code, and the answer isread from the next-token logits over those codes. Codes that would split into two tokens are skipped, so every option stays one token. This handles yes/no, scores, and choice lists up to several hundred options. Choices are read in two option orders and averaged, which cancels position bias.
- **Candidate-path head.** Past that point, a small learned head matches the state against each option's text, so the number of options has no fixed ceiling.
- **Calibration.** Temperatures fitted after training are applied at load time.
There is one per question type, readout mode, and choice option-count band.
  They were selected by how well they carry over to task families left out of
  the fit, not only by how well they fit held-out rows.

## Benchmarks

### S1Bench

Six S1Bench subsets. Every row uses the same items and harness. Accuracy:

| subset | task | **lev** | Qwen3.5-4B, untuned |
|---|---|---|---|
| aegis2 | safety moderation | **0.864** | 0.776 |
| boolq | yes/no reading comprehension | **0.880** | 0.860 |
| massive-en-US | intent, 60 classes | **0.791** | 0.734 |
| vitaminc-dev | claim verification | **0.738** | 0.733 |
| paws | adversarial paraphrase | 0.716 | **0.756** |
| helpsteer2 | helpfulness rating | 0.360 | **0.400** |
| **macro** | | **0.725** | 0.710 |

Against the 20 other models on the S1Bench board over these subsets, only a commercial decision API and three open models of 26B parameters or more score higher.

### Held-out split

A held-out split of the 29 training sources, with no row shared with training:

| metric | value |
|---|---|
| weighted accuracy | 0.807\* |
| expected calibration error | 0.061\* (0.180 before calibration) |
| banking77 (77 intents) | 0.980 |
| clinc_oos (151 intents) | 0.968 |
| FEVER claim verification | 0.872\* |

\* Measured on this checkpoint before the last serving update. That update routes choice sets of more than 68 options to label-token readout and re-selects the temperatures. The banking77 and clinc_oos rows come from after the update, which raised banking77 from 0.818. Smaller option sets are routed the same way as before.

### Speed

| | |
|---|---|
| compute per call, one H100 | **69 ms** |
| from 1 question to a 60-option choice | flat: one forward pass either way |
| output tokens | 0 |
| end to end from a laptop to a hosted endpoint, median | 414–463 ms (mostly network) |

## Training

- **Data:** 200,000 examples from 29 sources built on 26 public Hugging Face datasets. The tasks cover topic, sentiment, and emotion classification; intent detection (banking77, clinc_oos, snips); NLI (SNLI, ANLI, FEVER); paraphrase (MRPC, QQP, PARADE, plus word-swapped hard negatives); multiple-choice QA (RACE, ARC, SciQ, OpenBookQA, CommonsenseQA, StrategyQA);toxicity and safety (ToxiGen, ToxicChat, BeaverTails); and helpfulness (UltraFeedback). Questions are paraphrased, negated, and recast between types, and option sets are shuffled and resized, so the model learns the question format and not one wording.
- **Contamination guard:** the data build refuses any source that resolves to one of the 13 S1Bench subsets.
- **Recipe:** 3 epochs, 18,750 steps, batch size 32, learning rate 5e-5, in the base model's chat format, on one H100.

## Limitations

- **Minimal-edit pairs.** On paws and vitaminc, two inputs can differ by one swapped word or one changed number. Here the model can be confidently wrong. It scores below the untuned backbone on paws and only matches it on vitaminc.
- **Fine-grained quality ratings are weak.** Helpfulness scoring (helpsteer2) sits at 0.36. Treat such scores as a rough signal.
- **Calibration is fitted on the training distribution.** Temperatures are chosen to transfer across task families. Even so, a task very unlike the training mix may be less well calibrated. Check on your own data before you gate on the probabilities.
- **Partial benchmark coverage.** S1Bench results cover 6 of its 13 subsets.
- **English only.**
- **Needs a GPU for real-time use.** It runs on CPU, but a 4B backbone there takes seconds per call, not milliseconds.

## Files

| file | role |
|---|---|
| `adapter_model.safetensors`, `adapter_config.json` | LoRA adapter |
| `mode_b_head.pt` | candidate-path head (tensor state dict, loaded with `weights_only=True`) |
| `tokenizer*`, `chat_template.jinja` | the tokenizer that the label codes were verified against |
| `calibration.json` | fitted temperatures |
| `lev_release.json` | manifest: base model, prompt format, readout, training step |

## License

The adapter is released under Apache-2.0, the same license as the base model. Some of the training datasets have their own terms, including non-commercial licenses. Review them before commercial use.

Apache 2.0 · Interfaze
