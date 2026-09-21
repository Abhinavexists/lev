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
uv sync --extra modal
uv run modal setup
```

The default backbone is public, so **no credential is needed**. If you point at a
gated one, export a token and the app passes it through:

```bash
export HF_TOKEN=hf_...
```

For a shared or long-lived deployment, use a real Modal Secret instead:

```bash
modal secret create huggingface HF_TOKEN=hf_...
export LEV_HF_SECRET=huggingface
```

### The order

```bash
modal run modal/app.py::download --model-id Qwen/Qwen3.5-4B-Base   # once, ~8 GB
modal run modal/app.py::build_data --limit-per-source 20000        # CPU, no GPU
make smoke                                                         # ~5 min H100
make train PRESET=4b                                               # ~2 h H100
make calibrate PRESET=4b                                           # the temperatures
modal serve modal/app.py                                           # /v1/systemone
```

Warm the model cache once (~8 GB into a Volume, so later runs skip the download):

```bash
modal run modal/app.py::download --model-id Qwen/Qwen3.5-4B-Base
```

Build the data on **CPU**, not on the H100 — this is downloads, and paying GPU rates
to wait on a CDN is the most avoidable line on the bill:

```bash
modal run modal/app.py::build_data --limit-per-source 20000
```

Then **always run the smoke test before the real thing**:

```bash
make smoke      # 0.8B, 40 steps, ~5 min of H100
```

It exercises the whole path — image, volumes, **both readout modes**, the loss, and a
checkpoint write — and fails if only one mode was covered, so Mode B cannot ship
untested. If the data volume is empty it builds a small mixture first, so a fresh
workspace needs exactly this one command — though it builds that mixture *on the
GPU*, so for a real run do `build_data` first. Finding a bug here costs minutes;
finding it most of the way through the real run does not.

You can prove the same path with no GPU and no Modal account at all:

```bash
make smoke-local STEPS=20     # 0.8B on CPU; slow, but it is the real loop
```

### Volumes

| Volume | Holds | Why not in the image |
|---|---|---|
| `lev-models` | downloaded backbones | ~8 GB per checkpoint; baking it in makes every rebuild slow |
| `lev-checkpoints` | adapters, calibration profiles | written *during* training so a preemption is survivable |
| `lev-data` | prepared mixtures | reused across runs and ablations |

`modal serve` exposes whichever preset `SERVE_PRESET` names in `modal/app.py`
(`4b` by default) and falls back to the untrained backbone, loudly, if that
preset has no checkpoint yet — serving uncalibrated is a legitimate first step,
so it warns rather than refusing to start. `GET /health` reports which
checkpoint was resolved, whether a Mode B head loaded, and whether a calibration
profile is in effect. Check it before reading a number off any eval.

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
| `ContaminationError` | Working as designed. A source collides with an evaluation subset — remove it ([ADR-009](DECISIONS.md#adr-009--all-thirteen-evaluation-subsets-are-banned-from-training)) |
| `refusing to fit calibration on split 'test'` | Working as designed. Use a third split, disjoint from train and test |
| OOM at 4 k context | Check `make plan` headroom. Below ~20 GB, `validate()` should already have refused |
| LoRA loss flat, nothing learns | `enable_input_require_grads()` missing alongside gradient checkpointing — activations arrive with no `grad_fn` and adapters get no gradient |
| `chunk_gated_delta_rule is falling back to its reference PyTorch implementation` | Speed, not correctness — but 24 of the 32 layers are linear-attention, so it matters. Uncomment the `flash-linear-attention` / `causal-conv1d` layer in `modal/app.py` and re-measure |
| `ModuleNotMountable` — "lev has no spec - might not be installed?" | A stale `add_local_python_source`, which resolves the package through the local interpreter and so needs it installed in whichever Python runs `modal`. The app mounts the source directory instead; if you see this, your `modal/app.py` predates that fix |
| `no cache-fork API` | Your `transformers` predates hybrid-cache batch expansion. Upgrade, or run one forward per question (correct, slower) |
| `No training mixture is defined` | The data volume is empty. `make data` locally, or `modal run modal/app.py::build_data` |
| `labels appear outside train but never in it` | `--limit-per-source` is too small for a 77- or 151-option source. Raise it |
| `never show some of the options their question offers` | Same cause, caught earlier: the sample never contains some options at all |
| `Dataset scripts are no longer supported` | A source without a parquet mirror. `datasets>=5` dropped script execution; see the source table in [TRAINING.md](TRAINING.md) |
| `loss is nan at step N` | Deliberate stop. Training through a non-finite loss corrupts the adapter silently |
| SIGSEGV while loading weights on a Mac | `device_map="auto"` dispatching to MPS. `build_model` only uses it for multi-GPU; if you set it by hand, do not |
