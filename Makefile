# setup: Install workspace  |  setup-train: + training deps
# setup-serve: + serving deps  |  setup-modal: + Modal client

# test: Run tests  |  lint: Check code  |  fmt: Format code
# check: Lint + test  |  clean: Remove caches

# smoke-local: CPU smoke test  |  smoke: Modal H100 smoke test
# train: Full training run  |  calibrate: Fit temperatures

# serve: Dev server  |  deploy: Stable endpoint
# release: Package checkpoint  |  weights: Pull weights  |  publish: Push to HF

.DEFAULT_GOAL := help

PRESET  ?= 4b-instruct
RELEASE ?= $(PRESET)
NAME    ?=
REPO    ?=
OUT     ?= data/mixture
EVAL    ?= data/eval
S1      ?= data/s1bench
LIMIT   ?= 20000
STEPS   ?= 20
SOURCES ?=
URL     ?= http://localhost:8000
FRESH   ?=
RESUME  ?=

help:
	@awk '/^#/ {sub(/^# ?/, ""); print; next} /^$$/ {print; next} {exit}' $(MAKEFILE_LIST)

.PHONY: setup setup-train setup-serve setup-modal
setup:
	uv sync

setup-train:
	uv sync --extra train

setup-serve:
	uv sync --extra serve

setup-modal:
	uv sync --extra modal

.PHONY: test lint fmt check clean
test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

check: lint test

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache dist build


.PHONY: plan check-data data eval-set s1bench
plan:
	uv run lev plan --preset $(PRESET)

check-data:
	uv run lev check-data $(SOURCES)

data:
	uv run lev data build --out $(OUT) --limit-per-source $(LIMIT)

eval-set:
	uv run lev data eval --data $(OUT) --out $(EVAL)

s1bench:
	uv run lev s1bench export --out $(S1)


.PHONY: smoke-local smoke train calibrate
smoke-local:
	uv run lev data build --out /tmp/lev-mix --limit-per-source 2000 --n-examples 4000
	uv run lev train --preset smoke --data /tmp/lev-mix \
		--output-dir /tmp/lev-smoke --max-steps $(STEPS)

smoke:
	modal run modal/app.py::smoke

train:
	modal run modal/app.py::train --preset $(PRESET) $(if $(FRESH),--fresh,) $(if $(RESUME),--resume $(RESUME),)

calibrate:
	modal run modal/app.py::calibrate --preset $(PRESET)


.PHONY: serve deploy release weights publish
serve:
	LEV_SERVE_PRESET=$(PRESET) modal serve modal/app.py

deploy:
	LEV_SERVE_PRESET=$(PRESET) modal deploy modal/app.py

release:
	modal run modal/app.py::export_checkpoint --preset $(PRESET) $(if $(NAME),--name $(NAME),)

weights:
	mkdir -p weights
	modal volume get --force lev-checkpoints releases/$(RELEASE) weights/

publish:
	uv run lev release publish weights/$(RELEASE) --repo $(REPO)


.PHONY: bench bench-local sweep snake
bench: s1bench
	uv run levbench eval --backend jev --tasks $(S1)

bench-local: s1bench
	uv run levbench eval --backend lev --base-url $(URL) --tasks $(S1)

sweep:
	uv run levbench sweep --backend jev

snake:
	uv run levbench snake --backend lev --base-url $(URL) --record artifacts/snake/run.jsonl
