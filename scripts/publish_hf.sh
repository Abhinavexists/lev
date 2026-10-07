#!/usr/bin/env bash

# Package, validate, pull, and publish the 4b-instruct release to Hugging Face.
#
# scripts/publish_hf.sh                          # interfaze-ai/lev
# REPO=interfaze-ai/other scripts/publish_hf.sh  # custom repo
# DRY_RUN=1 scripts/publish_hf.sh                # skip upload

set -euo pipefail

PRESET="${PRESET:-4b-instruct}"
NAME="${NAME:-lev}"
REPO="${REPO:-interfaze-ai/lev}"

cd "$(dirname "$0")/.."
release="weights/$NAME"

# Prefer HF_TOKEN from .env over `hf auth login`.
if [[ -f .env ]] && grep -q '^HF_TOKEN=' .env; then
  set -a
  # shellcheck source=/dev/null
  source <(grep '^HF_TOKEN=' .env)
  set +a
fi

hf auth whoami >/dev/null || {
  echo "not logged in to the Hub: run 'hf auth login'" >&2
  exit 1
}

modal run modal/app.py::export_checkpoint --preset "$PRESET" --name "$NAME"
modal run modal/app.py::check_release --name "$NAME"

mkdir -p weights
modal volume get --force lev-checkpoints "releases/$NAME" weights/

cp hf/README.md "$release/README.md"
cp -R hf/assets "$release/"

echo "release ready in $release:"
ls -la "$release"

if [[ -n "${DRY_RUN:-}" ]]; then
  echo "DRY_RUN set: not uploading to $REPO"
  exit 0
fi

uv run lev release publish "$release" --repo "$REPO" --private
