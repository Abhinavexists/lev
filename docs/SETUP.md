# Setup

Two environments, and you only need the first to do useful work.

| | What it runs | Needs |
|---|---|---|
| **Local (any machine)** | the full test suite, the benchmark, the router, calibration fitting, budget planning | Python 3.12, `uv` |
| **H100 via Modal** | training, calibration over a real model, GPU serving | a Modal account |

The core package imports without `torch` on purpose — the schema, prompt layouts,
router, calibration and contamination guard are pure Python, so you can develop and
test the parts most likely to contain bugs on a laptop.

---

## 1. Local

```bash
git clone <this repo> && cd lev
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv
make setup
make test          # 68 tests, no GPU, no network, no API keys
```

Expected: `68 passed`. If that works, everything below is optional until you train.

```bash
make plan                      # the H100 budget for the 4B preset
make plan PRESET=9b            # ...or any other preset
uv run lev --help
uv run levbench --help
```

### Keys (optional)

Only needed to benchmark against the hosted Jev API or an LLM baseline.

```bash
cp .env.example .env
```

| Variable | Needed for |
|---|---|
| `TYPESAFE_API_KEY` | `levbench eval --backend jev`. Get one at <https://console.typesafe.ai/settings/keys> |
| `ANTHROPIC_API_KEY` | `levbench compare`, the LLM baseline |
| `LEVBENCH_LOCAL_API_KEY` | Only if you front a local server with auth |

> **Your real `TYPESAFE_API_KEY` is never sent to a `--base-url` host.** The SDK always
> sends `Authorization: Bearer <key>`, so reusing it against a third-party server would
> leak it. A placeholder is substituted instead, and there is a regression test for it.

---

## 2. Modal (the H100)

```bash
uv sync --extra train
pip install modal && modal setup
```

If the backbone is gated, add a token — a public checkpoint needs no secret:

```bash
modal secret create huggingface HF_TOKEN=hf_...
```

Warm the model cache once (~8 GB into a Volume, so later runs skip the download):

```bash
modal run modal/app.py::download --model-id Qwen/Qwen3.5-4B-Base
```

Then **always run the smoke test before the real thing**:

```bash
make smoke      # 0.8B, a few hundred steps, ~5 min of H100
```

It exercises the whole path — image, volumes, data, loss, checkpoint write. Finding a
bug here costs minutes; finding the same bug at hour 15 of a 16-hour run does not.

### Volumes

| Volume | Holds | Why not in the image |
|---|---|---|
| `lev-models` | downloaded backbones | ~8 GB per checkpoint; baking it in makes every rebuild slow |
| `lev-checkpoints` | adapters, calibration profiles | written *during* training so a preemption is survivable |
| `lev-data` | prepared mixtures | reused across runs and ablations |

---

## 3. Local GPU instead of Modal

Nothing is Modal-specific except `modal/app.py`. With a local H100:

```bash
uv sync --extra train
uv run python -c "
from lev.train.config import PRESETS
from lev.train.loop import run_training
run_training(PRESETS['4b'], data_dir='data', model_cache='~/.cache/huggingface')
"
```

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `FileNotFoundError: data/` | Run from the repo root, or check `repo_data_dir()` can walk up to it |
| `ContaminationError` | Working as designed. A source collides with an evaluation subset — remove it ([ADR-009](DECISIONS.md#adr-009--the-six-evaluation-subsets-are-banned-from-training)) |
| `refusing to fit calibration on split 'test'` | Working as designed. Use a third split, disjoint from train and test |
| OOM at 4 k context | Check `make plan` headroom. Below ~20 GB, `validate()` should already have refused |
| LoRA loss flat, nothing learns | `enable_input_require_grads()` missing alongside gradient checkpointing — activations arrive with no `grad_fn` and adapters get no gradient |
| `no cache-fork API` | Your `transformers` predates hybrid-cache batch expansion. Upgrade, or run one forward per question (correct, slower) |
