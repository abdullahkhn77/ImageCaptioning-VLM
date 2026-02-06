#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:?Usage: $0 <image_path> <checkpoint_path> [base_model]}"
CHECKPOINT="${2:?Usage: $0 <image_path> <checkpoint_path> [base_model]}"
BASE_MODEL="${3:-Salesforce/blip2-opt-2.7b}"

python src/inference.py \
    --image "$IMAGE" \
    --checkpoint "$CHECKPOINT" \
    --base-model "$BASE_MODEL"
