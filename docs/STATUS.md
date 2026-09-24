# Status

What is built, and how far each part has been exercised.

| Component | Evidence |
| --- | --- |
| Schema, prompt layouts, router, label codes | **done, tested** |
| Calibration fitting, ECE, profile I/O | **done, tested** |
| Contamination guard (all 13 eval subsets) | **done, tested** |
| Training config + budget arithmetic | **done, verified** |
| Benchmark harness | **done, tested** (offline) + run against live Jev |
| Data pipeline — 29 sources, 3 splits, augmented mixture | **done**, loads and splits verified |
| Held-out eval export | **done**, round-trips into levbench |
| Collator, both readouts, objective | **done, tested** |
| Training loop + checkpointing | **done** — 30 steps on Qwen3.5-0.8B-Base, both modes, losses finite |
| Modal app, image, volumes | **image builds on Modal** |
| Modal `download` / `build_data` / `smoke` | **run green on Modal** |
| Modal `train` / `calibrate` / `evaluate` | **complete 4B run, calibrated and scored** ([ADR-018](DECISIONS.md#adr-018--what-the-first-trained-checkpoint-actually-shows)) |
| Modal `serve` | **runs**; scored on S1Bench over HTTP |
| S1Bench harness | **done**; all 13 subsets on S1Bench's pinned items, Jev within 0.8 pp of its published numbers on each |
| Decision engine (batched forward, optional fork, readout) | **runs**; batched forward (ADR-023), skipped label codes and fitted temperatures (ADR-028) |
| Mode B head | **trains and serves**; weak unseen-taxonomy transfer measured on massive-en-US ([FINDINGS §12](FINDINGS.md#12-first-head-to-head-on-s1bench-two-named-failure-modes)) |
| `lev.load`, release packaging, Hub publishing | **done**; the release loads through both `lev.load` and the server on an H100 (`check_release`) |

Three checkpoints have been trained, calibrated and scored on S1Bench against Jev through identical task files: 0.489, then 0.697, then **0.725** macro against Jev's 0.754 on the earlier six-subset definitions. The released checkpoint scores **0.689** against Jev's 0.761 on all 13 subsets as S1Bench pins them. What each run changed and why is in [FINDINGS §12–17](FINDINGS.md); the reader bug that corrupted the labels of one intermediate run is [ADR-024](DECISIONS.md#adr-024--the-mixture-reader-merged-shuffled-questions-and-the-instruct-run-trained-on-it).
