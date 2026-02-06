#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"

echo "Starting training with config: $CONFIG"

accelerate launch src/train.py --config "$CONFIG"
