# Contributing

## Before you push

```bash
make check     # ruff + the full test suite
```

Both must be clean. Tests run without a GPU, network or API keys. A basic `uv sync` skips tests that need optional model dependencies; use `uv sync --extra train` to include the CPU tensor, cache and training-loop tests.

## Finding the code

| Work on | Start here | Focused tests |
| --- | --- | --- |
| Request schema, prompts or routing | `packages/lev/src/lev/{types,prompt,router,labels}.py` | `test_core.py` |
| Inference and HTTP serving | `packages/lev/src/lev/{model,server}.py`, `readout/` | `test_core.py`, `test_cache_fork.py` |
| Training data and contamination | `packages/lev/src/lev/data/` | `test_data_pipeline.py`, `test_contamination.py`, `test_eval_export.py` |
| Training and resume | `packages/lev/src/lev/train/{loop,collate,checkpoints}.py` | `test_training_loop.py`, `test_collate.py`, `test_loss.py` |
| Calibration and evaluation | `packages/lev/src/lev/calibrate.py`, `train/{calibration_run,evaluate}.py` | `test_calibration.py`, `test_evaluate.py` |
| CLI and release packaging | `packages/lev/src/lev/{cli,release}.py` | `test_cli.py`, `test_release.py`, `test_packaging.py` |
| Benchmark harness and demo | `packages/levbench/src/levbench/` | `test_offline.py`, `test_snake.py` |

Tests live under each package's `tests/` directory. For example:

```bash
uv run pytest packages/lev/tests/test_core.py
```

`modal/app.py` connects these components to remote volumes and GPUs. Import model dependencies inside the functions that need them so the core and CLI help remain usable without the training extras. Copy a `PRESETS` entry with `dataclasses.replace` before applying per-run overrides.

## Comments and structure

Use names and direct control flow to explain ordinary operations. Keep comments for constraints, tensor shapes, units, compatibility requirements and reasons a seemingly simpler implementation would be wrong. Docstrings should describe a contract or non-obvious behavior; omit ones that only repeat the function name.

Keep benchmark histories and design comparisons in `docs/FINDINGS.md` and `docs/DECISIONS.md`, with a short reference beside code when needed. Avoid copying those narratives into module headers or adding generic helpers for a single call site. Preserve package boundaries and public interfaces during cleanup.

## Where things go

| Change | Also update |
| --- | --- |
| An irreversible design choice | a new ADR in [`docs/DECISIONS.md`](docs/DECISIONS.md) |
| Anything about the readout, cache or objective | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §5 |
| A benchmark number | the snapshot it came from, and say which run it is |
| A new training knob | `TrainConfig`, so `make plan` reprices it |

## House rules

**Claims carry provenance.** Every number in the docs is tagged: verified here, taken from someone's published result, or projected. Keep the distinction, most of this repo's value is that a reader can tell which is which.

**Never quote a stopped benchmark run.** S1Bench has completed runs (6 subsets, 1,999 rows) and stopped ones (1 subset, 599 rows). Mixing them silently breaks every conclusion. `jeff-gpu` at 0.6644 is stopped; `jeff-gpu-full` at 0.5595 is not.

**Verify, don't reason.** "It should work" is not a check, and neither is a passing test that doesn't exercise the change. If you cannot run it, say so.

**The contamination guard is not negotiable.** All 13 S1Bench subsets are banned from training ([ADR-009](docs/DECISIONS.md#adr-009--all-thirteen-evaluation-subsets-are-banned-from-training)). Contamination makes the headline number *better* while invalidating it, which is why the guard raises instead of warning. Do not add a bypass flag.

**`levbench` must not import `lev`.** A measuring instrument that depends on the thing it measures is not an instrument.

## Tests

Write the test that would have caught the bug. Examples already in the tree:

- the fake tokenizer returns *more* than one token for unknown strings, because an earlier version returned one for single characters and made the router test vacuous;
- `test_padded_slots_do_not_produce_nan` exists because every loss in the first smoke run was `nan`: padded candidates carry `-inf` and a zero target, and `0 * -inf` poisons the batch mean while backward still runs;
- `test_limit_samples_rather_than_truncates` exists because `imdb[:400]` is 400 negative reviews — a head slice of a label-sorted corpus, invisible downstream because every split drawn from it is skewed identically;
- the calibration profile test asserts an unfitted bucket falls back to `T=1.0` rather than borrowing another bucket's scalar.

## Commits

One-line subject describing what the change ends up doing, not the route taken to it.
