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

<p align="center">
  <img src="assets/hero.png" alt="lev: one state, many typed decisions, one forward pass" width="100%">
</p>

lev answers typed questions about a piece of context in a single forward pass. You give it a **state** (text, a ticket, an email, or JSON) and a set of yes/no, choice, and score questions. It reads each answer from the logits it already computed and returns calibrated probabilities over exactly the options you supplied. It is a LoRA adapter on Qwen3.5-4B, and it speaks TypeSafe's `/v1/systemone` protocol, so code written for the TypeSafe SDK works against it once you change the base URL.

<div align="center" style="line-height: 1;"><img src="https://img.shields.io/badge/license-Apache--2.0-2a78d6?style=flat-square" alt="Apache-2.0" style="display: inline-block; vertical-align: middle; margin: 2px;"> <img src="https://img.shields.io/badge/base-Qwen3.5--4B-2a78d6?style=flat-square" alt="Qwen3.5-4B" style="display: inline-block; vertical-align: middle; margin: 2px;"> <img src="https://img.shields.io/badge/output%20tokens-0-2a78d6?style=flat-square" alt="Zero output tokens" style="display: inline-block; vertical-align: middle; margin: 2px;"> <img src="https://img.shields.io/badge/API-%2Fv1%2Fsystemone-2a78d6?style=flat-square" alt="/v1/systemone compatible" style="display: inline-block; vertical-align: middle; margin: 2px;"> <a href="https://github.com/Abhinavexists/lev"><img src="https://img.shields.io/badge/code-GitHub-14181f?style=flat-square&logo=github" alt="GitHub" style="display: inline-block; vertical-align: middle; margin: 2px;"></a></div>

<h2 align="center">68.9% on all 13 S1Bench subsets. 69 ms of compute. 4B parameters.</h2>

<p align="center"><strong>Qwen3.5-4B + LoRA · one H100 · 13 S1Bench subsets, 3,880 items.</strong><br>On the six subsets the public board completed, level with reflex-4b and behind only Jev and three open models of 26B–35B.<br>69 ms is engine compute for a short request, inside the container; S1Bench's longer states take more. End to end from a laptop, lev answered in 414–654 ms across two runs.<br><a href="#benchmarks">Benchmarks</a> · <a href="#speed">Speed</a> · <a href="#boundaries-worth-understanding">Boundaries</a></p>

<p align="center"><a href="#quickstart"><strong>Quickstart</strong></a> · <a href="#self-hosting-a-jev-compatible-http-server">Self-hosting</a> · <a href="#benchmarks">Benchmarks</a> · <a href="#why-it-works">Why it works</a> · <a href="#training">Training</a> · <a href="https://github.com/Abhinavexists/lev">GitHub</a></p>

**No generated tokens. No JSON to parse. No retries until it validates.**

**lev cannot return a label outside your options.** The answer space is the option set you send, so every response is well-formed by construction. lev can still pick the wrong option: this is a structural guarantee, not a guarantee of correctness.

| Question | You give | You get |
| --- | --- | --- |
| `noul` | a yes/no question | `noul` = p(yes) |
| `choice` | instructions + options (name → description or `null`) | `choice`, `probabilities`, `confidence` |
| `score` | instructions + 2–10 ordered levels | `score` (expected level), `probabilities`, `confidence` |

It is built for the high-volume judgement calls inside a product: routing, moderation, intent detection, triage, grading, and checking LLM output.

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

## Self-hosting: a Jev-compatible HTTP server

```bash
lev serve --checkpoint interfaze-ai/lev --host 0.0.0.0 --port 8000
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

## Benchmarks

### S1Bench

All 13 S1Bench subsets, 3,880 items: every item S1Bench scores, pinned by [Nimble](https://github.com/bespokelabsai/nimble)'s manifests. lev and TypeSafe Jev were run through the same harness on the same task files, and the Jev run lands within 0.8 points of TypeSafe's published figure on every subset. Accuracy, best of lev and Jev in bold:

<p align="center"><img src="assets/accuracy-by-subset.png" alt="lev and Jev accuracy on each S1Bench subset" width="100%"></p>

| subset | task | **lev** | Jev | always the most common label |
| --- | --- | --: | --: | --: |
| vitaminc-dev | claim verification | 0.668 | **0.801** | 0.503 |
| massive-en-US | intent routing, 18 scenarios | 0.857 | **0.874** | 0.163 |
| massive-de-DE | intent routing, German | 0.823 | **0.871** | 0.163 |
| boolq | yes/no reading comprehension | 0.827 | **0.893** | 0.580 |
| squad2 | answerability | 0.813 | **0.836** | 0.502 |
| paws | adversarial paraphrase | 0.776 | **0.900** | 0.516 |
| multinli | natural language inference | **0.890** | 0.836 | 0.361 |
| civil_comments | toxicity | 0.760 | **0.803** | 0.893 |
| aegis2 | safety moderation | 0.800 | **0.804** | 0.568 |
| helpsteer2 | helpfulness, 5 levels | **0.386** | 0.341 | 0.422 |
| summeval-relevance | summary relevance, 5 levels | 0.358 | 0.358 | 0.458 |
| summeval-consistency | summary faithfulness, 5 levels | 0.271 | **0.812** | 0.840 |
| pubmedqa | biomedical yes/no/maybe | 0.732 | **0.764** | 0.532 |
| **macro** | | 0.689 | **0.761** | |

> **What the headline means:** at these sizes a per-subset difference needs roughly 5–9 points to clear sampling noise. Jev leads on 10 of 13 subsets and on macro accuracy by 7.2 points. Mean calibration error (ECE) is 0.115 for lev against Jev's 0.091; lev is better calibrated on 5 of 13.

<p align="center"><img src="assets/leaderboard.png" alt="S1Bench leaderboard over the six subsets every listed model completed" width="100%"></p>

The public S1Bench board has complete results on six of the subsets. There lev scores 0.719, level with reflex-4b (0.719, the same backbone) and behind only Jev and three open models of 26B–35B parameters.

<p align="center"><img src="assets/accuracy-per-parameter.png" alt="Macro accuracy against parameter count" width="100%"></p>

#### Where Jev leads

- **Minimal-edit pairs:** paws (−12.4 points) and vitaminc (−13.3), where two inputs differ by one swapped word or one changed number.
- **Summary faithfulness:** summeval-consistency (−54.1). 121 of 144 summaries are rated fully faithful, and lev mostly answers one level lower. Fine-tuning introduced this: the untuned backbone scores 0.826.
- **End-to-end latency from a laptop:** Jev's hosted API answered in 335–346 ms median, lev on one Modal H100 in 600–654 ms.

lev leads on natural language inference (multinli, +5.4) and helpfulness rating (helpsteer2, +4.5), both at the edge of sampling noise. On civil_comments and the three 5-level rating subsets, always answering the most common label beats both models.

### Held-out split

A held-out split of the 29 training sources, with no row shared with training:

| metric | value |
| --- | --- |
| weighted accuracy | 0.807\* |
| expected calibration error | 0.061\* (0.180 before calibration) |
| banking77 (77 intents) | 0.980 |
| clinc_oos (151 intents) | 0.968 |
| FEVER claim verification | 0.872\* |

\* Measured on this checkpoint before the last serving update. That update routes choice sets of more than 68 options to label-token readout and re-selects the temperatures. The banking77 and clinc_oos rows come from after the update, which raised banking77 from 0.818. Smaller option sets are routed the same way as before.

### Speed

<p align="center"><img src="assets/compute.png" alt="lev compute per call on one H100: 169 ms to 69 ms" width="100%"></p>

A call is one batched forward pass over every question, so compute stays flat from one question to eight, and a 60-option choice costs the same as a yes/no.

The 69 ms is for a short request (a three-sentence state), measured inside the container. On S1Bench's aegis2 states, about 470 tokens each, Modal logged a median of 287 ms of execution per call and the laptop saw 589 ms.

## Why it works

<p align="center"><img src="assets/how-it-works.png" alt="State and questions go through one forward pass, then label-token readout or the candidate-path head, then per-bucket temperatures, then typed answers" width="460"></p>

- **Label-token readout.** Each option gets a short code, and the answer is read from the next-token logits over those codes. Codes that would split into two tokens are skipped, so every option stays one token. This handles yes/no, scores, and choice lists up to several hundred options. Choices are read in two option orders and averaged, which cancels position bias.
- **Candidate-path head.** Past that point, a small learned head matches the state against each option's text, so the number of options has no fixed ceiling.
- **Calibration.** Temperatures fitted after training are applied at load time. There is one per question type, readout mode, and choice option-count band. They were selected by how well they carry over to task families left out of the fit, not only by how well they fit held-out rows.

### The optimizations that mattered

- **One batched forward instead of prefill-and-fork: 169 → 69 ms.** At this size the forward pass is bound by kernel launches, not arithmetic, so forking the cache saved FLOPs and cost time. One batched forward plus the depthwise-conv kernel cut compute by 59%. [ADR-023](https://github.com/Abhinavexists/lev/blob/main/docs/DECISIONS.md#adr-023--one-batched-forward-not-prefill-and-fork)
- **Label-token readout up to the tokenizer's limit: +51 points on 60-class intent.** Serving 60 options through label codes instead of the learned head took 60-class intent from 0.231 to 0.746. [ADR-025](https://github.com/Abhinavexists/lev/blob/main/docs/DECISIONS.md#adr-025--serving-routes-mode-a-up-to-the-tokenizer-limit-training-keeps-its-cap)
- **Skipping codes that split: banking77 0.818 → 0.980.** Passing over codes that tokenize to two tokens lifts label-token readout from 68 options to several hundred. [ADR-028](https://github.com/Abhinavexists/lev/blob/main/docs/DECISIONS.md#adr-028--skip-split-label-codes-when-serving-calibrate-for-families-the-model-has-not-seen)

## Training

- **Data:** 200,000 examples from 29 sources built on 26 public Hugging Face datasets. The tasks cover topic, sentiment, and emotion classification; intent detection (banking77, clinc_oos, snips); NLI (SNLI, ANLI, FEVER); paraphrase (MRPC, QQP, PARADE, plus word-swapped hard negatives); multiple-choice QA (RACE, ARC, SciQ, OpenBookQA, CommonsenseQA, StrategyQA); toxicity and safety (ToxiGen, ToxicChat, BeaverTails); and helpfulness (UltraFeedback). Questions are paraphrased, negated, and recast between types, and option sets are shuffled and resized, so the model learns the question format and not one wording.
- **Contamination guard:** the data build refuses any source that resolves to one of the 13 S1Bench subsets.
- **Recipe:** LoRA r=32, α=64 on the q/k/v/o attention and MLP projections; 3 epochs, 18,750 steps, batch size 32, learning rate 5e-5, in the base model's chat format, on one H100.

## Boundaries worth understanding

- **Minimal-edit pairs.** On paws and vitaminc, two inputs can differ by one swapped word or one changed number. Here the model can be confidently wrong, and Jev leads by 12–13 points.
- **Fine-grained quality ratings are weak.** On helpsteer2 and both summeval subsets, always answering the most common level beats lev, and lev under-rates faithful summaries. Treat such scores as a rough signal.
- **Calibration is fitted on the training distribution.** Temperatures are chosen to transfer across task families. Even so, a task very unlike the training mix may be less well calibrated. Check on your own data before you gate on the probabilities.
- **Questions are answered independently.** Answers in one request do not condition on each other. Encode a joint decision as one choice, or ask in stages.
- **English only.**
- **Needs a GPU for real-time use.** It runs on CPU, but a 4B backbone there takes seconds per call, not milliseconds.

## Files

| file | role |
| --- | --- |
| `adapter_model.safetensors`, `adapter_config.json` | LoRA adapter |
| `mode_b_head.pt` | candidate-path head (tensor state dict, loaded with `weights_only=True`) |
| `tokenizer*`, `chat_template.jinja` | the tokenizer that the label codes were verified against |
| `calibration.json` | fitted temperatures |
| `lev_release.json` | manifest: base model, prompt format, readout, training step |

## License

The adapter is released under Apache-2.0, the same license as the base model. Some of the training datasets have their own terms, including non-commercial licenses. Review them before commercial use.

Not affiliated with or endorsed by TypeSafe AI.

Apache 2.0 · Interfaze
