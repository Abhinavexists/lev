# lev — common tasks. `make help` lists everything.
.DEFAULT_GOAL := help

# Overridable on the command line: `make train PRESET=9b`.
PRESET ?= 4b
OUT    ?= data/mixture
EVAL   ?= data/eval
S1     ?= data/s1bench
LIMIT  ?= 20000
STEPS  ?= 20
URL    ?= http://localhost:8000
RELEASE ?= $(PRESET)
REPO   ?=

.PHONY: help setup setup-train setup-serve setup-modal test lint fmt check \
        bench bench-local sweep plan check-data data eval-set s1bench smoke smoke-local \
        train calibrate serve deploy release weights publish snake clean

help: ## Show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Install the workspace (CPU only; no torch)
	uv sync

setup-train: ## Install with the training extras (torch, transformers, peft)
	uv sync --extra train

setup-serve: ## Install with the serving extras (adds fastapi, uvicorn)
	uv sync --extra serve

setup-modal: ## Install with the Modal client
	uv sync --extra modal

test: ## Run every test (no GPU, no network, no API keys)
	uv run pytest

lint: ## Check formatting and lint rules
	uv run ruff check .
	uv run ruff format --check .

fmt: ## Apply formatting and autofixes
	uv run ruff check --fix .
	uv run ruff format .

check: lint test ## Lint then test — run before pushing

plan: ## Print the H100 training budget without spending it
	uv run lev plan --preset $(PRESET)

check-data: ## Contamination guard over a source list (SOURCES=path)
	uv run lev check-data $(SOURCES)

bench: ## Benchmark against Jev (needs TYPESAFE_API_KEY)
	uv run levbench eval --backend jev

bench-local: ## Benchmark a local /v1/systemone server (no API key)
	uv run levbench eval --backend lev --base-url $(URL)

sweep: ## Measure shared-state batching economics
	uv run levbench sweep --backend jev

data: ## Build the training mixture locally (LIMIT=rows per source)
	uv run lev data build --out $(OUT) --limit-per-source $(LIMIT)

eval-set: ## Export the held-out split as levbench task files
	uv run lev data eval --data $(OUT) --out $(EVAL)

s1bench: ## Export the S1Bench eval subsets as levbench task files
	uv run lev s1bench export --out $(S1)

smoke-local: ## Train 0.8B on CPU for a few steps — proves the path without Modal
	uv run lev data build --out /tmp/lev-mix --limit-per-source 2000 --n-examples 4000
	uv run lev train --preset smoke --data /tmp/lev-mix \
		--output-dir /tmp/lev-smoke --max-steps $(STEPS)

smoke: ## Modal: exercise the whole training path on 0.8B (~5 min of H100)
	modal run modal/app.py::smoke

train: ## Modal: the real run; resumes the newest checkpoint (FRESH=1 to start over, RESUME=path)
	modal run modal/app.py::train --preset $(PRESET) $(if $(FRESH),--fresh,) $(if $(RESUME),--resume $(RESUME),)

calibrate: ## Modal: fit per-bucket temperatures after training
	modal run modal/app.py::calibrate --preset $(PRESET)

serve: ## Modal: dev server on an H100 (ephemeral; any `modal run` on this app steals its URL)
	LEV_SERVE_PRESET=$(PRESET) modal serve modal/app.py

deploy: ## Modal: deploy /v1/systemone with a stable URL (PRESET selects the checkpoint)
	LEV_SERVE_PRESET=$(PRESET) modal deploy modal/app.py

release: ## Modal: package the newest PRESET checkpoint into a release dir on the volume
	modal run modal/app.py::export_checkpoint --preset $(PRESET) $(if $(NAME),--name $(NAME),)

weights: ## Pull a packaged release to weights/RELEASE (RELEASE=name, default PRESET)
	modal volume get --force lev-checkpoints releases/$(RELEASE) weights/$(RELEASE)

publish: ## Upload weights/RELEASE to the Hub (REPO=org/name; needs HF_TOKEN)
	uv run lev release publish weights/$(RELEASE) --repo $(REPO)

snake: ## The decision model plays Snake over /v1/systemone (URL=server; --record in artifacts/)
	uv run levbench snake --backend lev --base-url $(URL) --record artifacts/snake/run.jsonl

clean: ## Remove caches and build artefacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache dist build
