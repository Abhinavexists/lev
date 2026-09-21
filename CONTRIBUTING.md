# Contributing

## Before you push

```bash
make check     # ruff + 68 tests
```

Both must be clean. The test suite needs no GPU, no network and no API keys — if a
change makes that untrue, the change is wrong.

## Where things go

| Change | Also update |
|---|---|
| An irreversible design choice | a new ADR in [`docs/DECISIONS.md`](docs/DECISIONS.md) |
| Anything about the readout, cache or objective | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §5 |
| A benchmark number | the snapshot it came from, and say which run it is |
| A new training knob | `TrainConfig`, so `make plan` reprices it |

## House rules

**Claims carry provenance.** Every number in the docs is tagged: verified here, taken
from someone's published result, or projected. Keep the distinction — most of this
repo's value is that a reader can tell which is which.

**Never quote a stopped benchmark run.** S1Bench has completed runs (6 subsets, 1,999
rows) and stopped ones (1 subset, 599 rows). Mixing them silently breaks every
conclusion. `jeff-gpu` at 0.6644 is stopped; `jeff-gpu-full` at 0.5595 is not.

**Verify, don't reason.** "It should work" is not a check, and neither is a passing
test that doesn't exercise the change. If you cannot run it, say so in the docstring —
several modules here do exactly that.

**The contamination guard is not negotiable.** Six subsets are banned from training
([ADR-009](docs/DECISIONS.md#adr-009--the-six-evaluation-subsets-are-banned-from-training)).
Contamination makes the headline number *better* while invalidating it, which is why
the guard raises instead of warning. Do not add a bypass flag.

**`levbench` must not import `lev`.** A measuring instrument that depends on the
thing it measures is not an instrument.

## Tests

Write the test that would have caught the bug. Two examples already in the tree:

- the fake tokenizer returns *more* than one token for unknown strings, because an
  earlier version returned one for single characters and made the router test vacuous;
- the calibration profile test asserts an unfitted bucket falls back to `T=1.0` rather
  than borrowing another bucket's scalar.

## Commits

One-line subject describing what the change ends up doing, not the route taken to it.
