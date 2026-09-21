# lev — common tasks. `make help` lists everything.
.DEFAULT_GOAL := help
.PHONY: help setup setup-train setup-serve setup-modal test lint fmt check \
        bench bench-local sweep plan check-data smoke train calibrate serve clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
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
	uv run lev plan --preset $(or $(PRESET),4b)

check-data: ## Contamination guard over a source list (SOURCES=path)
	uv run lev check-data $(SOURCES)

bench: ## Benchmark against Jev (needs TYPESAFE_API_KEY)
	uv run levbench eval --backend jev

bench-local: ## Benchmark a local /v1/systemone server (no API key)
	uv run levbench eval --backend jev --base-url $(or $(URL),http://localhost:8000)

sweep: ## Measure shared-state batching economics
	uv run levbench sweep --backend jev

smoke: ## Modal: exercise the whole training path on 0.8B (~5 min of H100)
	modal run modal/app.py::smoke

train: ## Modal: the real run (PRESET=4b, ~16h on one H100)
	modal run modal/app.py::train --preset $(or $(PRESET),4b)

calibrate: ## Modal: fit per-bucket temperatures after training
	modal run modal/app.py::calibrate --preset $(or $(PRESET),4b)

serve: ## Modal: serve /v1/systemone on an H100
	modal serve modal/app.py

clean: ## Remove caches and build artefacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache dist build
