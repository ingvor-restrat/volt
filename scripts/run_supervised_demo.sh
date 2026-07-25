#!/usr/bin/env bash
set -euo pipefail

python_command="${PYTHON:-python3}"
dataset="${1:-artifacts/supervised_gnn/graphs.jsonl}"
output_dir="${2:-artifacts/supervised_gnn}"

mkdir -p "$(dirname "$dataset")" "$output_dir"
cargo run --quiet --example graph -- --output "$dataset"
"$python_command" scripts/train_gnn.py \
  --input "$dataset" \
  --output-dir "$output_dir"
