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
make test      # the full suite: no GPU, no network, no API keys
make plan      # the H100 training budget, before you spend it
```

```
model            Qwen/Qwen3.5-4B-Base  (4.0B, bfloat16)
adaptation       LoRA r32
data             200,000 examples x 128 tok x 3 epochs  = 0.08B tokens
steps            18,750 (32 ex/step, ~4,096 tok/step)
compute          2.46e+18 FLOPs
H100 estimate    1.7 hours (0.1 days)
memory           8.3 GB state, 71.7 GB headroom of 80 GB
```

The 128-token figure is **measured** over 1,500 real rendered prompts, not assumed.
`lev plan --data data/mixture` re-measures it against your actual mixture.

Then, to train:

```bash
make setup-train           # torch, transformers, peft
make data                  # 9 public corpora -> train / calibration / test
make eval-set              # export the held-out split for levbench
make smoke-local STEPS=20  # prove the path on 0.8B, on CPU, before spending a GPU
make train PRESET=4b       # Modal, one H100, ~2 h
```

[SETUP.md](docs/SETUP.md) for Modal, [TRAINING.md](docs/TRAINING.md) for the pipeline.

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
levbench eval  --backend jev                       # the hosted API
levbench eval  --backend lev  --tasks data/eval    # us, on localhost:8000
levbench compare                                   # vs an LLM baseline
levbench sweep                                     # batching economics
levbench confidence                                # which statistic is `confidence`?
```

Omit `--tasks` and you get a built-in 24-item fixture. It is a smoke test: at n=24 the
95% interval is **±16 accuracy points**, so it cannot tell a 5-point improvement from
none. `make eval-set` writes the real one.
[ADR-015.](docs/DECISIONS.md#adr-015--the-24-item-task-set-is-a-fixture-not-a-benchmark)

Reports accuracy, log loss, Brier, **ECE with reliability bins**, selective accuracy,
p50 latency, tokens, cost, and schema-retry counts. Runs offline against a fake
transport with no key at all.

---

## Watch it decide

The model plays Snake, one `/v1/systemone` call per move -- laya-mlx's demo,
ported so it runs against any System One server through the same client the
benchmark uses:

```bash
uv sync --extra demo
uv run levbench snake --backend planner                        # no server: the planner plays, see the display
make snake URL=https://<your-serve-url>                        # lev
uv run levbench snake --backend jev --steps 200 --record artifacts/snake/jev.jsonl
uv run levbench replay artifacts/snake/jev.jsonl              # play it back at original speed
```

The display is laya's: the board, big-digit score/length/best, the four
direction probabilities with the model's pick marked and unsafe moves flagged,
`EXECUTING` with `SHIELD` when the planner had to overrule an unsafe first
choice, the model's dead-end-risk and food-reachability estimates beside the
planner's ground truth, inference time, decisions/second, and a running score
of both Noul questions against that truth. SPACE pauses, +/- change the pace,
R resets, Q quits. `--unassisted` removes the shield so deaths end the run;
with it on, a finished board rolls into the next round. `--record` writes a
JSONL that `levbench replay` plays back. In the compact prompt the state text
*states* the Noul answers, so the running Noul score is the cheapest test there
is of whether a model reads its question.

## Speed

Measured inside the container on an H100 (`modal run modal/app.py::profile_engine`):
the original prefill-and-fork path took ~160 ms per call regardless of how
many questions were asked, because at these sizes the forward is bound by
kernel launches, not arithmetic. The engine now runs one batched forward
(65–95 ms depending on the host) and the image builds the `causal_conv1d`
kernel (another ~10%). `torch.compile` was tried and measured slower on this
hybrid model, so it is off. What you see from a laptop is mostly the route to
the container: a 280 ms round trip, TLS on top, then Modal's ingress.

```bash
LEV_SERVE_COMPILE=1 make deploy PRESET=4b          # torch.compile path: measured slower here, off by default
LEV_SERVE_CONCURRENCY=8 make deploy                # requests in flight per container (default 4)
LEV_SERVE_WARM=1 make deploy                       # keep one container up: no cold start, idle cost
LEV_SERVE_REGION=<modal region> make deploy        # put the container near the client
uv run levbench eval --backend lev --tasks data/s1bench --base-url $URL --concurrency 4
```

Measured on paws (250 items, same checkpoint, identical accuracy and ECE to
four decimals): the old path took ~232 s; the new one 112 s sequentially and
39 s at `--concurrency 4`. Per-call latency from a laptop halved (p50 928 →
417 ms); the full 1,999-item S1Bench pass takes about five minutes. ADR-023
and FINDINGS §13 have the breakdown.

## Weights

A release is one flat directory -- adapter, Mode B head, tokenizer, fitted
temperatures, and a `lev_release.json` naming the base model and the option
cap the weights were trained under -- that `lev serve` loads as-is from disk
or from the Hub.

```bash
make release PRESET=4b-instruct          # package the newest checkpoint on the Modal volume
make weights RELEASE=4b-instruct         # pull it to weights/4b-instruct
make publish RELEASE=4b-instruct REPO=org/lev-4b-instruct   # needs HF_TOKEN

uv run lev serve --checkpoint weights/4b-instruct            # from disk
uv run lev serve --checkpoint org/lev-4b-instruct            # from the Hub
```

The server reads the manifest for the base model, so `--model` is not needed
for a packaged release, and `GET /health` reports what it loaded. On Modal,
`make deploy PRESET=4b-instruct` serves that preset's newest checkpoint.

## Status

| | |
|---|---|
| Schema, prompt layouts, router, label codes | **done, tested** |
| Calibration fitting, ECE, profile I/O | **done, tested** |
| Contamination guard (all 13 eval subsets) | **done, tested** |
| Training config + budget arithmetic | **done, verified** |
| Benchmark harness | **done, tested** (offline) + run against live Jev |
| Data pipeline — 29 sources, 3 splits, augmented mixture | **done**, loads and splits verified |
| Held-out eval export (2,760 items, ±4 pts) | **done**, round-trips into levbench |
| Collator, both readouts, objective | **done, tested** |
| Training loop + checkpointing | **done** — 30 steps on Qwen3.5-0.8B-Base, both modes, losses finite |
| Modal app, image, volumes | **image builds on Modal** |
| Modal `download` / `build_data` / `smoke` | **run green on Modal** |
| Modal `train` / `calibrate` / `evaluate` | **complete 4B run, calibrated and scored** ([ADR-018](docs/DECISIONS.md#adr-018--what-the-first-trained-checkpoint-actually-shows)) |
| Modal `serve` | **runs**; scored on S1Bench over HTTP |
| S1Bench harness | **done**; validated against Jev's own per-subset numbers |
| Decision engine (prefill, fork, readout) | **runs**; option cap, two-order averaging, binary Noul ([ADR-020](docs/DECISIONS.md#adr-020--the-trained-model-lost-to-the-frozen-one-what-changes-and-what-does-not)) |
| Mode B head | **trains and serves**; unseen-taxonomy transfer untested |

Two checkpoints have been trained, calibrated and scored on **S1Bench against
Jev through identical task files**. The first, on nine classification corpora,
scored 0.489 macro against Jev's 0.754 and the frozen backbone's 0.719
([FINDINGS §12](docs/FINDINGS.md), [ADR-020](docs/DECISIONS.md#adr-020--the-trained-model-lost-to-the-frozen-one-what-changes-and-what-does-not)).
The second, from the instruct checkpoint on 23 sources with paraphrased and
negated questions and varied option sets, scores **0.697** -- equal to Jev on
aegis2, above it on helpsteer2, three points under the frozen backbone -- with
held-out 0.836 / ECE 0.046 ([FINDINGS §15](docs/FINDINGS.md)). Getting there
included a serving decision worth 51 points on one subset: route Mode A up to
the tokenizer's single-token limit rather than the training cap
([ADR-025](docs/DECISIONS.md#adr-025--serving-routes-mode-a-up-to-the-tokenizer-limit-training-keeps-its-cap)).
The remaining gap is paws, where paraphrase training taught the lexical-overlap
shortcut PAWS exists to punish, and calibration on option sets larger than the
mixture contains. A reader bug that corrupted the labels of one
intermediate run is recorded in [ADR-024](docs/DECISIONS.md#adr-024--the-mixture-reader-merged-shuffled-questions-and-the-instruct-run-trained-on-it).

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
