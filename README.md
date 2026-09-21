<div align="center">

# lev

**A typed, calibrated decision model — and the harness that proves it.**

State in, typed probabilistic decisions out. One prefill, many questions,
zero generated tokens.

[Architecture](docs/ARCHITECTURE.md) · [Decisions](docs/DECISIONS.md) · [Setup](docs/SETUP.md) · [Training](docs/TRAINING.md) · [Findings](docs/FINDINGS.md)

</div>

---

## What this is

An open reproduction of the *abstraction* behind [TypeSafe's Jev](https://typesafe.ai) —
a "System One" model that answers typed questions about a state and returns calibrated
probabilities instead of prose.

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient(base_url="http://localhost:8000", api_key="local")

result = client.system_one(
    state="My payouts failed three times this week and nobody replied to my emails.",
    questions={
        "team": Choice(
            instructions="Which team?",
            criteria={"payments": "Money movement", "account": "Login and profile"},
        ),
        "urgency": Score(instructions="How urgent?", criteria=["low", "medium", "high"]),
        "escalate": Noul(instructions="Should a human take over?"),
    },
)
result.answers["team"].choice  # "payments"
result.answers["team"].probabilities  # {"payments": 0.97, "account": 0.03}
result.answers["urgency"].score  # 1.86
result.answers["escalate"].noul  # 0.91
```

No JSON to parse, no retry-until-it-validates, no generated tokens. Answers are read
out of a logit vector the forward pass already produced.

## The one-paragraph thesis

Open reproductions have already caught Jev on **accuracy** — the best is 1.0 points
behind. Where they have not caught it is **calibration**: 1.6–2.2× worse Expected
Calibration Error at comparable accuracy. And the one open model that *is* well
calibrated gets there not through architecture but by fitting **a single scalar**
after training. So this project competes on calibration, not accuracy, and treats it
as a first-class objective rather than a post-processing step.
[The evidence.](docs/DECISIONS.md#adr-006--calibration-is-the-product)

---

## Quickstart

```bash
make setup     # uv sync — no torch, no GPU
make test      # 68 tests: no GPU, no network, no API keys
make plan      # the H100 training budget, before you spend it
```

```
model            Qwen/Qwen3.5-4B-Base  (4.0B, bfloat16)
adaptation       LoRA r32
data             200,000 examples x 1200 tok x 3 epochs  = 0.72B tokens
steps            21,972 (32,768 tok/step)
compute          2.30e+19 FLOPs
H100 estimate    16.0 hours (0.7 days)
memory           8.3 GB state, 71.7 GB headroom of 80 GB
```

Then [SETUP.md](docs/SETUP.md) for Modal, or [TRAINING.md](docs/TRAINING.md) to train.

---

## Layout

```
packages/
  lev/      the model     — schema, prompt layouts, router, readouts, calibration, server
  levbench/     the harness   — accuracy, ECE, cost, latency, batching economics
modal/          app.py        — train / calibrate / serve on one H100
docs/           ARCHITECTURE  — the design, and every model it was derived from
                DECISIONS     — one ADR per irreversible choice, with the evidence
                FINDINGS      — what Jev actually is, verified against the shipping SDK
                SETUP         — local and Modal
                TRAINING      — the pipeline, step by step
data/           sample state + the S1Bench snapshot the analysis rests on
```

`levbench` deliberately does **not** depend on `lev`. A measuring instrument that
imports the thing it measures is not an instrument.
[ADR-010.](docs/DECISIONS.md#adr-010--two-packages-one-workspace)

---

## The architecture in one diagram

```
                    ┌──────────── STATE ────────────┐
                    │  32k tokens, text or JSON      │
                    └───────────────┬────────────────┘
                                    ▼
          Qwen3.5-4B-Base: 24 linear + 8 full-attention layers
          only 8 layers hold K/V, so forking the cache is cheap
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
          STATE-FIRST cache                SCHEMA-FIRST cache
          fork across questions            reuse across states
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
         A: label-token readout          B: candidate-path scoring
         options that fit one token      everything else — no ceiling
         zero added params               shared head + set attention
                    └───────────────┬───────────────┘
                                    ▼
                    raw logits over candidates
                                    ▼
             TEMPERATURE, fitted per (type, mode)
                                    ▼
    Choice: softmax  │  Score: Σ(i·pᵢ)  │  Noul: 9-rating-token distribution
```

**The unclaimed part is the router.** Every other implementation picks one readout
family and hits a wall: when options will not fit into single tokens, they cap the
option count or reject the request. We route those questions to Mode B instead. Their
hard failure is our second mode.
[ADR-005.](docs/DECISIONS.md#adr-005--dual-mode-readout-the-differentiator)

---

## Benchmarking

`levbench` measures us, Jev, and any `/v1/systemone`-compatible server through the
same code path — one changed flag.

```bash
levbench eval  --backend jev                                    # the hosted API
levbench eval  --backend jev --base-url http://localhost:8000   # us, or any clone
levbench compare                                                # vs an LLM baseline
levbench sweep                                                  # batching economics
levbench confidence                                             # which statistic is `confidence`?
```

Reports accuracy, log loss, Brier, **ECE with reliability bins**, selective accuracy,
p50 latency, tokens, cost, and schema-retry counts. Runs offline against a fake
transport with no key at all.

---

## Status

| | |
|---|---|
| Schema, prompt layouts, router, label codes | **done, tested** |
| Calibration fitting, ECE, profile I/O | **done, tested** |
| Contamination guard | **done, tested** |
| Training config + budget arithmetic | **done, verified** |
| Benchmark harness | **done, tested** (offline) |
| Modal app, volumes, smoke path | written, **not yet run** |
| Decision engine (prefill, fork, readout) | written, **not yet run on hardware** |
| Mode B head | written, **untrained** |
| Data loaders | **the next task** |

Every module that has not been executed says so in its own docstring. Nothing in this
repository has been trained, and the harness has not yet been pointed at the live Jev
API — so the design is evidenced, and its outcome is a hypothesis.

---

## Prior work

This design is assembled from measured trade-offs in other people's implementations.
[ARCHITECTURE.md](docs/ARCHITECTURE.md) dissects each one; the parts taken:

| From | What |
|---|---|
| [reflex](https://github.com/kshetrajna12/reflex) | post-hoc temperature calibration — the cheapest large win |
| [decider](https://github.com/Mapika/decider) | dual prompt layouts, hybrid-attention schema cache, abstain augmentation |
| [simple-jev](https://github.com/featherless-ai/simple-jev) | nine-rating-token Noul |
| [NanoJev](https://github.com/TianyuCodings/NanoJev) | set attention over candidate paths, proper-scoring objectives |
| [litjev](https://github.com/zhengxuyu/litjev) | the reference prefill + logit-readout write-up |
| [jeff](https://github.com/logan-markewich/jeff), [Nimble](https://github.com/bespokelabsai/nimble), [typed-decisions](https://github.com/kotoba-lang/typed-decisions) | encoder and LoRA baselines, and the latency numbers that ruled out diffusion |

Benchmark data is a snapshot of a third-party S1Bench dashboard
(`data/s1bench-snapshot.json`) that I did not produce; its caveats are in
[FINDINGS.md §9](docs/FINDINGS.md).

Not affiliated with or endorsed by TypeSafe AI.

## License

[Apache-2.0](LICENSE).
