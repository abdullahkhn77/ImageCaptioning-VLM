#!/usr/bin/env bash
# ============================================================================
# prepare_data.sh — Download COCO captions, convert to JSONL, validate.
#
# Usage:
#   bash scripts/prepare_data.sh                   # full pipeline
#   bash scripts/prepare_data.sh --skip-download    # skip image download
#   bash scripts/prepare_data.sh --skip-images      # annotations only
#   bash scripts/prepare_data.sh --config configs/custom.yaml
# ============================================================================
set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
CONFIG="configs/default.yaml"
SKIP_DOWNLOAD=false
SKIP_IMAGES=false

# ── Parse flags ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)       CONFIG="$2"; shift 2 ;;
        --skip-download) SKIP_DOWNLOAD=true; shift ;;
        --skip-images)  SKIP_IMAGES=true; shift ;;
        *)              echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ── Read paths from config (requires python + pyyaml) ──────────────────────
read_yaml() {
    python3 -c "
import yaml, sys
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
keys = '$1'.split('.')
val = cfg
for k in keys:
    val = val[k]
print(val)
"
}

COCO_ROOT="$(read_yaml data.coco.root)"
COCO_YEAR="$(read_yaml data.coco.year)"
IMAGE_ROOT="$(read_yaml data.image_root)"
TRAIN_ANN="$(read_yaml data.train_annotations)"
VAL_ANN="$(read_yaml data.val_annotations)"
TEST_ANN="$(read_yaml data.test_annotations)"

echo "============================================"
echo " VLM Captioning — Data Preparation"
echo "============================================"
echo "  Config:     $CONFIG"
echo "  COCO root:  $COCO_ROOT"
echo "  COCO year:  $COCO_YEAR"
echo "  Image root: $IMAGE_ROOT"
echo ""

# ── Step 1: Download COCO ──────────────────────────────────────────────────
if [ "$SKIP_DOWNLOAD" = true ]; then
    echo "[1/4] Skipping COCO download (--skip-download)"
else
    echo "[1/4] Downloading COCO ${COCO_YEAR} captions..."
    EXTRA_FLAGS=""
    if [ "$SKIP_IMAGES" = true ]; then
        EXTRA_FLAGS="--skip-images"
    fi
    python3 src/data_utils.py download-coco \
        --root "$COCO_ROOT" \
        --year "$COCO_YEAR" \
        $EXTRA_FLAGS
fi
echo ""

# ── Step 2: Convert to JSONL ───────────────────────────────────────────────
echo "[2/4] Converting COCO annotations to JSONL..."

COCO_ANN_DIR="${COCO_ROOT}/annotations"
COCO_TRAIN_ANN="${COCO_ANN_DIR}/captions_train${COCO_YEAR}.json"
COCO_VAL_ANN="${COCO_ANN_DIR}/captions_val${COCO_YEAR}.json"

COCO_TRAIN_IMGS="${COCO_ROOT}/train${COCO_YEAR}"
COCO_VAL_IMGS="${COCO_ROOT}/val${COCO_YEAR}"

# Symlink COCO images into our image root so paths stay relative
mkdir -p "$IMAGE_ROOT"
if [ -d "$COCO_TRAIN_IMGS" ] && [ ! -L "${IMAGE_ROOT}/train${COCO_YEAR}" ]; then
    ln -sf "$(cd "$COCO_TRAIN_IMGS" && pwd)" "${IMAGE_ROOT}/train${COCO_YEAR}" 2>/dev/null || true
fi
if [ -d "$COCO_VAL_IMGS" ] && [ ! -L "${IMAGE_ROOT}/val${COCO_YEAR}" ]; then
    ln -sf "$(cd "$COCO_VAL_IMGS" && pwd)" "${IMAGE_ROOT}/val${COCO_YEAR}" 2>/dev/null || true
fi

# Convert train split
python3 src/data_utils.py convert-coco \
    --annotation "$COCO_TRAIN_ANN" \
    --image-dir "$COCO_TRAIN_IMGS" \
    --output "data/coco_train_full.jsonl"

# Convert val split
python3 src/data_utils.py convert-coco \
    --annotation "$COCO_VAL_ANN" \
    --image-dir "$COCO_VAL_IMGS" \
    --output "data/coco_val_full.jsonl"

echo ""

# ── Step 3: Split val into val + test ──────────────────────────────────────
echo "[3/4] Splitting COCO val into val + test..."

# Use the COCO train as our train, and split the COCO val 50/50
cp "data/coco_train_full.jsonl" "$TRAIN_ANN"

python3 src/data_utils.py split \
    --input "data/coco_val_full.jsonl" \
    --output-dir data/ \
    --train-ratio 0.0 \
    --val-ratio 0.5 \
    --test-ratio 0.5 \
    --seed 42

# The split command writes train.jsonl (empty), val.jsonl, test.jsonl into data/
# Remove the empty train.jsonl produced by the split (we already set the real one)
echo ""

# ── Step 4: Validate ──────────────────────────────────────────────────────
echo "[4/4] Validating datasets..."
echo ""

for SPLIT_FILE in "$TRAIN_ANN" "$VAL_ANN" "$TEST_ANN"; do
    if [ -f "$SPLIT_FILE" ]; then
        echo "--- $(basename "$SPLIT_FILE") ---"
        python3 src/data_utils.py validate \
            --annotations "$SPLIT_FILE" \
            --image-root "$IMAGE_ROOT"
        echo ""
    fi
done

echo "============================================"
echo " Data preparation complete!"
echo "============================================"
