# Setup

Two environments, and you only need the first to do useful work.

| | What it runs | Needs |
| --- | --- | --- |
| **Local (any machine)** | tests, the benchmark, the router, calibration fitting, budget planning | Python 3.12, `uv`; the `train` extra for CPU tensor tests |
| **H100 via Modal** | training, calibration over a real model, GPU serving | a Modal account |

The core package imports without `torch` on purpose — the schema, prompt layouts, router, calibration and contamination guard are pure Python, so you can develop and test the parts most likely to contain bugs on a laptop.

---

## 1. Local

```bash
git clone <this repo> && cd lev
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv
make setup
make test          # offline tests; optional model tests need the train extra
```

Tests that need torch skip on a bare `uv sync`; the rest must pass. If they do, everything below is optional until you train.

```bash
make plan PRESET=4b-instruct   # the H100 budget for the released preset
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
| --- | --- |
| `TYPESAFE_API_KEY` | `levbench eval --backend jev`. Get one at <https://console.typesafe.ai/settings/keys> |
| `ANTHROPIC_API_KEY` | `levbench compare`, the LLM baseline |
| `LEVBENCH_LOCAL_API_KEY` | Only if you front a local server with auth |

`.env` also reaches `modal deploy` — see [the deploy knobs](#deploy-knobs) below. Anything you `export` wins over it.

> **Your real `TYPESAFE_API_KEY` is never sent to a `--base-url` host.** The SDK always sends `Authorization: Bearer <key>`, so reusing it against a third-party server would leak it. A placeholder is substituted instead, and there is a regression test for it.

---

## 2. Modal (the H100)

```bash
uv sync --extra modal
uv run modal setup
```

The backbones are public, so **no credential is needed**. For a gated one, export a token and the app passes it through:

```bash
export HF_TOKEN=hf_...
```

For a shared or long-lived deployment, use a real Modal Secret instead:

```bash
modal secret create huggingface HF_TOKEN=hf_...
export LEV_HF_SECRET=huggingface
```

<a id="deploy-knobs"></a>
### Deploy knobs

Read on the machine where `modal deploy` runs — from the environment or from `.env` — never inside the container. The first group travels to the container as a Secret; the second sets decorator arguments, which are fixed at import.

| Variable | Default | Effect |
| --- | --- | --- |
| `LEV_SERVE_PRESET` | `4b` | Which preset's newest checkpoint to serve. `make deploy PRESET=...` sets it |
| `LEV_SERVE_MODEL` | unset | Serve this model frozen instead — no adapter, binary Noul, the zero-shot baseline |
| `LEV_SERVE_COMPILE` | `0` | `torch.compile`. Off unless asked: measured slower than eager here (ADR-023) |
| `LEV_SERVE_MAX_LABEL_OPTIONS` | unset | Cap on label options. Unset means Mode A up to the tokenizer limit (ADR-025) |
| `LEV_SERVE_PROMPT` | the preset's | Override the prompt style, mainly for a frozen model |
| `LEV_SERVE_SKIP_CODES` | `1` | Skip label codes that tokenize to more than one token (ADR-028) |
| `LEV_SERVE_CONCURRENCY` | `4` | Requests one container handles at once; GPU forwards serialise, parsing and network overlap |
| `LEV_SERVE_WARM` | `0` | Containers kept running. One removes the 20-55 s cold start, at the cost of an idle GPU |
| `LEV_SERVE_REGION` | unset | A Modal region near the client; the measured 280 ms round trip is a continent, not a server |
| `LEV_SERVE_SCALEDOWN` | `300` | Idle seconds before a container stops |

> Leave these **commented** in `.env` rather than writing `VAR=`. The empty string is not the same as unset: `int("")` raises for the four numeric knobs, and an empty `LEV_SERVE_PRESET` fails the preset check instead of falling back to the default.

### The order

```bash
modal run modal/app.py::download --model-id Qwen/Qwen3.5-4B  # once, ~8 GB
modal run modal/app.py::build_data --limit-per-source 20000   # CPU, no GPU
make smoke                                                    # ~5 min H100
make train PRESET=4b-instruct                                 # ~2 h H100
make calibrate PRESET=4b-instruct                             # the temperatures
make deploy PRESET=4b-instruct                                # /v1/systemone
```

`download` warms the model cache (~8 GB into a Volume). `build_data` runs on CPU: it is downloads, and there is no reason to pay GPU rates to wait on a CDN.

Then **always run the smoke test before the real thing**:

```bash
make smoke      # 0.8B, 40 steps, ~5 min of H100
```

It exercises the whole path — image, volumes, **both readout modes**, the loss, and a checkpoint write — and fails if only one mode was covered. If the data volume is empty it builds a small mixture first, on the GPU, so for a real run do `build_data` first.

You can prove the same path with no GPU and no Modal account at all:

```bash
make smoke-local STEPS=20     # 0.8B on CPU; slow, but it is the real loop
```

### Volumes

| Volume | Holds | Why not in the image |
| --- | --- | --- |
| `lev-models` | downloaded backbones | ~8 GB per checkpoint; baking it in makes every rebuild slow |
| `lev-checkpoints` | adapters, calibration profiles | written *during* training so a preemption is survivable |
| `lev-data` | prepared mixtures | reused across runs and ablations |

The server exposes the preset named by `LEV_SERVE_PRESET` (`make deploy PRESET=...` sets it), else `SERVE_PRESET` in `modal/app.py` (`4b`). With no checkpoint for that preset it serves the untrained backbone with a warning. `GET /health` reports which checkpoint resolved, whether a Mode B head loaded, and whether a calibration profile is in effect; check it before reading a number off any eval. `make serve` gives an ephemeral dev URL that any other `modal run` on the app takes over, so use `make deploy` for anything you benchmark.

---

## 3. Local GPU instead of Modal

Nothing is Modal-specific except `modal/app.py`. With a local H100:

```bash
uv sync --extra train
uv run python -c "
from lev.train.config import PRESETS
from lev.train.loop import run_training
run_training(PRESETS['4b-instruct'], data_dir='data/mixture', model_cache='~/.cache/huggingface')
"
```

---

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `FileNotFoundError: data/` | Run from the repo root, or check `repo_data_dir()` can walk up to it |
| `ContaminationError` | Working as designed. A source collides with an evaluation subset — remove it ([ADR-009](DECISIONS.md#adr-009--all-thirteen-evaluation-subsets-are-banned-from-training)) |
| `refusing to fit calibration on split 'test'` | Working as designed. Use a third split, disjoint from train and test |
| OOM at 4 k context | Check `make plan` headroom. Below ~20 GB, `validate()` should already have refused |
| LoRA loss flat, nothing learns | `enable_input_require_grads()` missing alongside gradient checkpointing — activations arrive with no `grad_fn` and adapters get no gradient |
| `chunk_gated_delta_rule is falling back to its reference PyTorch implementation` | Speed, not correctness — but 24 of the 32 layers are linear-attention, so it matters. The image installs `flash-linear-attention` for this ([ADR-017](DECISIONS.md#adr-017--batches-are-length-bucketed-and-the-budget-was-wrong-again)) |
| `causal_conv1d was requested, but nvcc was not found` / `NameError: bare_metal_version` | `causal-conv1d` is a CUDA source build; the image uses an `nvidia/cuda:*-devel` base for its compiler. If the build breaks on a version bump, delete that `pip_install` line: runs get slower, not wrong |
| `ModuleNotMountable` — "lev has no spec - might not be installed?" | A stale `add_local_python_source`, which resolves the package through the local interpreter and so needs it installed in whichever Python runs `modal`. The app mounts the source directory instead; if you see this, your `modal/app.py` predates that fix |
| The cache fork fails under `prefix_mode="fork"` | It needs `transformers` 5.x (`reorder_cache` on a hybrid cache). Upgrade, or use the default `prefix_mode="single"` |
| `data directory ... does not exist` | The data volume is empty. `make data` locally, or `modal run modal/app.py::build_data` |
| `labels appear outside train but never in it` | `--limit-per-source` is too small for a 77- or 151-option source. Raise it |
| `never show some of the options their question offers` | Same cause, caught earlier: the sample never contains some options at all |
| `Dataset scripts are no longer supported` | A source without a parquet mirror. `datasets>=5` dropped script execution; see the source table in [TRAINING.md](TRAINING.md) |
| `loss is nan at step N` | Deliberate stop. Training through a non-finite loss corrupts the adapter silently |
| SIGSEGV while loading weights on a Mac | `device_map="auto"` dispatching to MPS. `build_model` only uses it for multi-GPU; if you set it by hand, do not |
