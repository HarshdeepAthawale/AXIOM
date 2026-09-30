#!/usr/bin/env bash
# Fetch the two small INT8 ONNX models Axiom's reported score used, pre-exported
# on the Hugging Face Hub, into data/models/onnx/. No torch or optimum needed,
# which is what makes this usable in a container build.
#
#   bash scripts/fetch_models.sh [target-dir]      # default: data/models
#
# Setup.md section 6 exports the same models locally instead; either works.
set -euo pipefail

root="${1:-data/models}"
hub="https://huggingface.co"

fetch() {
  local repo="$1" slug="$2" onnx="$3"
  local dir="$root/onnx/$slug"
  mkdir -p "$dir"
  curl -fsSL "$hub/$repo/resolve/main/$onnx" -o "$dir/model.onnx"
  for file in tokenizer.json tokenizer_config.json special_tokens_map.json vocab.txt config.json; do
    curl -fsSL "$hub/$repo/resolve/main/$file" -o "$dir/$file"
  done
  echo "fetched $repo -> $dir"
}

# quint8_avx2: dynamic INT8 for any x86-64 CPU with AVX2.
fetch sentence-transformers/all-MiniLM-L6-v2 all-minilm-l6-v2-int8 onnx/model_quint8_avx2.onnx
fetch cross-encoder/ms-marco-MiniLM-L6-v2 ms-marco-minilm-l-6-v2-int8 onnx/model_quint8_avx2.onnx
