#!/usr/bin/env bash
# Build step for the hosted API (render.yaml): install the backend, fetch the
# models, and index the generated 3-version demo corpus the console queries.
set -euo pipefail

index_root="${AXIOM_DEMO_INDEX:-.axiom-demo}"
corpus="${AXIOM_DEMO_CORPUS:-demo-corpus}"

python -m pip install --upgrade pip
python -m pip install -e ".[retrieval,structural,serve]"

bash scripts/fetch_models.sh data/models

rm -rf "$corpus" "$index_root"
python scripts/make_demo_repo.py "$corpus" --files 60 --versions 3
index_root="$(pwd)/$index_root"
(
  cd "$corpus"
  axiom index . --at v1.0.0 --version-id v1.0.0 --profile demo --index-root "$index_root"
  axiom reindex --from v1.0.0 --to v2.0.0 --version-id v2.0.0 --profile demo --index-root "$index_root"
  axiom reindex --from v2.0.0 --to v3.0.0 --version-id v3.0.0 --profile demo --index-root "$index_root"
)
axiom versions --index-root "$index_root"
