# Dataset Format

## Directory Structure

```
data/
  images/              # All images (or symlinks to COCO splits, etc.)
  train.jsonl          # Training annotations
  val.jsonl            # Validation annotations
  test.jsonl           # Test annotations
  coco/                # (optional) Raw COCO download
    annotations/
    train2017/
    val2017/
```

## Annotation Format (JSONL)

Each line is a self-contained JSON object with an `image` path and a `caption`:

```jsonl
{"image": "train2017/000000391895.jpg", "caption": "A man with a red helmet on a small moped on a dirt road."}
{"image": "train2017/000000522418.jpg", "caption": "A woman wearing a net on her head cutting a cake."}
```

### Multi-caption variant

If you have multiple captions per image you can use either format:

**One row per caption (preferred):**
```jsonl
{"image": "img_001.jpg", "caption": "A dog on a bench."}
{"image": "img_001.jpg", "caption": "A pet resting beside a person on a wooden bench."}
```

**One row per image with a list:**
```jsonl
{"image": "img_001.jpg", "captions": ["A dog on a bench.", "A pet resting beside a person."]}
```

Both formats are handled transparently by the dataset loader.

### Fields

| Field      | Type            | Description                                       |
|------------|-----------------|---------------------------------------------------|
| `image`    | string          | Filename relative to `data.image_root` in config  |
| `caption`  | string          | Single ground-truth caption                       |
| `captions` | list of strings | Alternative: multiple captions for the same image |

## Plain JSON format

Legacy `.json` files (a top-level JSON array) are also supported:

```json
[
  {"image": "img_001.jpg", "caption": "A dog on a bench."},
  {"image": "img_002.jpg", "caption": "An aerial view of a city skyline at sunset."}
]
```

## Using COCO Captions

The fastest way to get started:

```bash
bash scripts/prepare_data.sh
```

This downloads COCO 2017, converts annotations to JSONL, and validates the result.
To skip the (large) image download and convert annotations only:

```bash
bash scripts/prepare_data.sh --skip-images
```

To manually convert a COCO annotation file:

```bash
python src/data_utils.py convert-coco \
    --annotation data/coco/annotations/captions_train2017.json \
    --image-dir data/coco/train2017 \
    --output data/train.jsonl
```

## Using Flickr30k

```bash
python src/data_utils.py convert-flickr \
    --token-file /path/to/results_20130124.token \
    --output data/flickr30k.jsonl
```

## Splitting a custom dataset

```bash
python src/data_utils.py split \
    --input data/my_dataset.jsonl \
    --output-dir data/ \
    --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

All captions for the same image are kept in the same split to prevent data leakage.

## Validating a dataset

```bash
python src/data_utils.py validate \
    --annotations data/train.jsonl \
    --image-root data/images/
```

Reports total entries, unique images, missing files, and caption length statistics.
