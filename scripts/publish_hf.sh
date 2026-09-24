#!/usr/bin/env bash
# Package the trained 4b-instruct checkpoint on Modal, check that it loads and
# answers, pull it, swap in the Hub model card (hf/README.md) and upload it to a
# private Hugging Face repo.
#
#   scripts/publish_hf.sh                          # interfaze-ai/lev-4b
#   REPO=interfaze-ai/other scripts/publish_hf.sh  # another repo
#   DRY_RUN=1 scripts/publish_hf.sh                # everything but the upload
set -euo pipefail

PRESET="${PRESET:-4b-instruct}"
NAME="${NAME:-lev-4b}"
REPO="${REPO:-interfaze-ai/lev-4b}"

cd "$(dirname "$0")/.."
release="weights/$NAME"

hf auth whoami >/dev/null || { echo "not logged in to the Hub: run 'hf auth login'" >&2; exit 1; }

modal run modal/app.py::export_checkpoint --preset "$PRESET" --name "$NAME"
# Load the release as a user will (lev.load and the server) before anything is pulled or published.
modal run modal/app.py::check_release --name "$NAME"
# The target must be an existing directory: given a missing path, `volume get`
# writes every file of the release onto that one path.
mkdir -p weights
modal volume get --force lev-checkpoints "releases/$NAME" weights/
cp hf/README.md "$release/README.md"

echo "release ready in $release:"
ls -la "$release"
if [[ -n "${DRY_RUN:-}" ]]; then
  echo "DRY_RUN set: not uploading to $REPO"
  exit 0
fi

uv run lev release publish "$release" --repo "$REPO" --private
