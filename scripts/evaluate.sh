#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
CHECKPOINT="${2:?Usage: $0 <config> <checkpoint_path> [split]}"
SPLIT="${3:-test}"

echo "Evaluating checkpoint: $CHECKPOINT on split: $SPLIT"

python src/evaluate.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --split "$SPLIT"
