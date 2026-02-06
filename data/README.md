# Dataset Format

## Directory Structure

```
data/
  images/          # All images referenced by annotation files
    img_001.jpg
    img_002.jpg
    ...
  train.json       # Training annotations
  val.json         # Validation annotations
  test.json        # Test annotations
```

## Annotation JSON Format

Each annotation file is a JSON array of objects with the following fields:

```json
[
  {
    "image": "img_001.jpg",
    "caption": "A dog sitting on a park bench next to its owner."
  },
  {
    "image": "img_002.jpg",
    "caption": "An aerial view of a city skyline at sunset."
  }
]
```

| Field     | Type   | Description                                          |
|-----------|--------|------------------------------------------------------|
| `image`   | string | Filename relative to `data/images/`                  |
| `caption` | string | Ground-truth caption for the image                   |

Multiple captions per image are supported — include one entry per caption:

```json
[
  {"image": "img_001.jpg", "caption": "A dog on a bench."},
  {"image": "img_001.jpg", "caption": "A pet resting beside a person on a wooden bench."}
]
```

## Using Existing Datasets

To use COCO Captions or Flickr30k, convert them to the format above. A conversion
utility is provided in `src/data_utils.py` (see the `convert_coco_to_json` function).
