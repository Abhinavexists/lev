# lev

lev is an open System One decision model: `f(state, questions) → typed, calibrated distributions`, served over the same `/v1/systemone` wire format as TypeSafe's Jev, with no token generation. It is `Qwen/Qwen3.5-4B` with a rank-32 LoRA adapter, a set-attention candidate head, and per-bucket temperature calibration, released on the Hub as [`interfaze-ai/lev`](https://huggingface.co/interfaze-ai/lev). `levbench` is the harness that measures it, and Jev, on S1Bench.

@CONTRIBUTING.md

## Where the facts live

Read the relevant doc before changing anything it covers; most "obvious" improvements here were already tried and are recorded with their evidence.

| Doc | Holds |
| --- | --- |
| [`docs/STATUS.md`](docs/STATUS.md) | What is built, how far each part is exercised, the current headline numbers |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | ADR-001 to ADR-030; the next decision is ADR-031 |
| [`docs/FINDINGS.md`](docs/FINDINGS.md) | Every measurement and head-to-head, in order, with provenance |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The design, and the trade-offs refused |
| [`docs/TRAINING.md`](docs/TRAINING.md) | The training order, presets, budgets, resume |
| [`docs/SETUP.md`](docs/SETUP.md) | Local and Modal setup, volumes, deploy knobs |
| [`hf/README.md`](hf/README.md) | The Hub model card; keep it consistent with `README.md` |

## The model

- **Readout.** Mode A reads label-token logits at the answer boundary whenever every label code is a single token — 68 options for this tokenizer, since the 69th code, `BQ`, is two. Above that, Mode B scores candidate text with the trained head. Training and serving route the same way (ADR-025, ADR-026); serving also skips split codes (ADR-028). Routing is in `router.py`.
- **Calibration** is per bucket, keyed `{type}:{mode}[:{band}]`, selected by leave-one-family-out transfer ECE. `calibrate.fit` refuses `test`, `eval` and `holdout` splits — fit only on the dedicated `calibration` split.
- **The release** (`lev-4b`) is preset `4b-instruct`, step 18750, chat prompt style. Its `lev_release.json` manifest sets the base model, prompt style and optional `adapter_scale`, and `lev.load` honours all three.
- **The released weights were trained at commit `c6ccad7`.** Changes to data, sources, prompts or the training loop after it are not in the weights until a retrain; do not attribute a benchmark move to them.

## Running things

- **Anything that needs a GPU runs on Modal**, through `modal/app.py` and the `make` targets. Do not download checkpoints or run models locally to check something — probe the deployed endpoint or add a `modal run` function.
- **Pass `--preset 4b-instruct` to every Modal function you call directly.** Their defaults are preset `4b`, the plain base-model run, not the release; the `make` targets pass `PRESET=4b-instruct` for you. `modal deploy` likewise serves `4b` unless `LEV_SERVE_PRESET` is set, which `make deploy` does.
- `make deploy` replaces the live endpoint. `modal/app.py` must import inside the container, where `python-dotenv` is not installed — keep its import guarded, or every container crash-loops.
- Modal volumes: `lev-models` (base weights), `lev-checkpoints` (runs and `releases/`), `lev-data` (mixture and splits).

## Benchmarks

- S1Bench is 13 subsets pinned by Bespoke Nimble manifests at commit `62076b4`, in `packages/lev/src/lev/data/s1bench_subsets/`. All 13 are banned from training (ADR-009).
- **Never select anything on S1Bench** — a hyperparameter, a scale, a prompt. Select with leave-one-family-out over the subsets, as FINDINGS §19 does, or on data outside both training and S1Bench. The `calibration` split cannot select out-of-distribution knobs: it is drawn from the training mixture.
- Compare against Jev through identical task files, one complete run each.

## Safety

- **Ask before any paid call**: `levbench` against Jev, the Anthropic backend, or a sweep. They bill per token.
- `levbench` never sends `TYPESAFE_API_KEY` to a `--base-url` host; a custom endpoint gets `LEVBENCH_LOCAL_API_KEY` or `local`. Keep it that way.
- Do not read `.env`.
- This repository is mirrored by hand to the public [`InterfazeAI/lev`](https://github.com/InterfazeAI/lev); the two diverge, so a change meant for release must be ported, not assumed.

## Comments

The code should explain itself; a comment exists only for what it cannot say.

- **One line is the unit.** A comment is one line, a docstring is one line. Multi-line is earned — a module's two or three lines of orientation, or tensor shapes and wire formats. A second paragraph on a function belongs in `docs/` with a pointer from the code.
- **Do not restate the code.** If a reader would have to check the prose against the implementation to confirm they agree, the comment costs more than it gives.
- **Compress before deleting.** A real fact buried in a block above a statement usually becomes a short trailing comment on the statement.
- **Point, do not restate.** `(ADR-028)` beats a paragraph reproducing the decision. A contract the code states plainly — a guard that raises, a named check like `_refuse_held_out` — needs no prose.
- **Public surface is the exception.** `lev.load`, the server, and the `/v1/systemone` types are consumed without reading the body; document their contracts.
- Do not rewrite comments for style. When changing code, remove comments the change made obsolete, and do no unrelated cleanup.
