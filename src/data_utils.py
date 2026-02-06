"""Data utilities: download, convert, validate, and split captioning datasets."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence
from zipfile import ZipFile

from tqdm import tqdm

logger = logging.getLogger(__name__)

# =========================================================================
# COCO download
# =========================================================================

_COCO_URLS = {
    "train_images": "http://images.cocodataset.org/zips/train{year}.zip",
    "val_images": "http://images.cocodataset.org/zips/val{year}.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval{year}.zip",
}


def _download_file(url: str, dest: Path) -> None:
    """Download *url* to *dest* with a tqdm progress bar."""
    if dest.exists():
        logger.info("Already exists, skipping download: %s", dest)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, dest)

    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name,
        ) as bar:
            while True:
                chunk = resp.read(1 << 20)  # 1 MiB
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract a zip archive into *dest_dir*."""
    logger.info("Extracting %s -> %s", zip_path, dest_dir)
    with ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def download_coco_captions(
    root: str | Path = "data/coco",
    year: int = 2017,
    skip_images: bool = False,
) -> Path:
    """Download and extract the COCO Captions dataset.

    Final layout::

        <root>/
            annotations/
                captions_train<year>.json
                captions_val<year>.json
            train<year>/
                000000000009.jpg  ...
            val<year>/
                000000000139.jpg  ...

    Parameters
    ----------
    root:
        Destination directory.
    year:
        COCO dataset year (default 2017).
    skip_images:
        If ``True`` only download the annotation zip (useful when images
        are already available).

    Returns
    -------
    Path to *root*.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    zip_dir = root / "zips"

    # -- annotations --
    ann_url = _COCO_URLS["annotations"].format(year=year)
    ann_zip = zip_dir / f"annotations_trainval{year}.zip"
    _download_file(ann_url, ann_zip)
    _extract_zip(ann_zip, root)

    # -- images --
    if not skip_images:
        for split in ("train_images", "val_images"):
            url = _COCO_URLS[split].format(year=year)
            zip_path = zip_dir / f"{split.replace('_images', '')}{year}.zip"
            _download_file(url, zip_path)
            _extract_zip(zip_path, root)

    logger.info("COCO %d ready at %s", year, root)
    return root


# =========================================================================
# Conversion
# =========================================================================

def convert_coco_to_jsonl(
    coco_annotation_path: str | Path,
    image_dir: str | Path,
    output_path: str | Path,
) -> int:
    """Convert a COCO captions JSON file to our standard JSONL format.

    Each output line: ``{"image": "<relative_filename>", "caption": "..."}``

    Parameters
    ----------
    coco_annotation_path:
        Path to the COCO annotation JSON (e.g. ``captions_train2017.json``).
    image_dir:
        Directory containing the COCO images for this split.  Used only to
        verify existence; the written paths are filenames relative to the
        image directory.
    output_path:
        Where to write the JSONL file.

    Returns
    -------
    Number of entries written.
    """
    with open(coco_annotation_path) as f:
        coco = json.load(f)

    id_to_filename: dict[int, str] = {
        img["id"]: img["file_name"] for img in coco["images"]
    }

    image_dir = Path(image_dir)
    missing = 0
    count = 0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as out:
        for ann in coco["annotations"]:
            filename = id_to_filename[ann["image_id"]]
            if not (image_dir / filename).exists():
                missing += 1
                continue
            line = json.dumps({"image": filename, "caption": ann["caption"]})
            out.write(line + "\n")
            count += 1

    if missing:
        logger.warning("%d annotations skipped (image file not found in %s)", missing, image_dir)
    logger.info("Wrote %d entries to %s", count, output_path)
    return count


def convert_flickr30k_to_jsonl(
    token_file: str | Path,
    output_path: str | Path,
) -> int:
    """Convert Flickr30k ``results_20130124.token`` to our JSONL format.

    Returns the number of entries written.
    """
    count = 0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(token_file) as fin, open(output_path, "w") as fout:
        for line in fin:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            filename = parts[0].split("#")[0]
            caption = parts[1]
            fout.write(json.dumps({"image": filename, "caption": caption}) + "\n")
            count += 1

    logger.info("Wrote %d entries to %s", count, output_path)
    return count


# =========================================================================
# Train / val / test splitting
# =========================================================================

def create_train_val_test_split(
    input_path: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    group_by_image: bool = True,
) -> dict[str, int]:
    """Split a JSONL file into train / val / test subsets.

    Parameters
    ----------
    input_path:
        Source JSONL file.
    output_dir:
        Directory to write ``train.jsonl``, ``val.jsonl``, ``test.jsonl``.
    train_ratio, val_ratio, test_ratio:
        Proportions (must sum to 1.0).
    seed:
        Random seed for reproducibility.
    group_by_image:
        If ``True`` (default), all captions for the same image stay in the
        same split — prevents data leakage.

    Returns
    -------
    Dict mapping split name to number of entries written.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1.0"
    )

    entries = _load_jsonl(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    if group_by_image:
        # Group entries by image filename
        grouped: dict[str, list[dict]] = defaultdict(list)
        for entry in entries:
            grouped[entry["image"]].append(entry)

        image_keys = list(grouped.keys())
        rng.shuffle(image_keys)

        n = len(image_keys)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits: dict[str, list[str]] = {
            "train": image_keys[:n_train],
            "val": image_keys[n_train : n_train + n_val],
            "test": image_keys[n_train + n_val :],
        }

        counts: dict[str, int] = {}
        for split_name, keys in splits.items():
            split_entries = [e for k in keys for e in grouped[k]]
            _write_jsonl(split_entries, output_dir / f"{split_name}.jsonl")
            counts[split_name] = len(split_entries)
    else:
        rng.shuffle(entries)
        n = len(entries)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        parts = {
            "train": entries[:n_train],
            "val": entries[n_train : n_train + n_val],
            "test": entries[n_train + n_val :],
        }
        counts = {}
        for split_name, split_entries in parts.items():
            _write_jsonl(split_entries, output_dir / f"{split_name}.jsonl")
            counts[split_name] = len(split_entries)

    for name, c in counts.items():
        logger.info("  %s: %d entries", name, c)
    return counts


# =========================================================================
# Validation
# =========================================================================

def validate_dataset(
    annotations_path: str | Path,
    image_root: str | Path,
) -> dict[str, Any]:
    """Scan a dataset and report statistics.

    Returns a dict with keys:
    - ``total_entries``
    - ``unique_images``
    - ``missing_images`` (list of filenames)
    - ``duplicate_images`` (images appearing more than once — expected for
      multi-caption datasets)
    - ``avg_caption_length`` (in characters)
    - ``min_caption_length``
    - ``max_caption_length``
    - ``avg_caption_words``
    """
    entries = _load_annotations(annotations_path)
    image_root = Path(image_root)

    image_counter: Counter[str] = Counter()
    caption_lengths: list[int] = []
    word_counts: list[int] = []
    missing: list[str] = []

    for entry in entries:
        fname = entry["image"]
        image_counter[fname] += 1
        if not (image_root / fname).exists():
            missing.append(fname)
        cap = entry["caption"]
        caption_lengths.append(len(cap))
        word_counts.append(len(cap.split()))

    n = len(entries)
    stats: dict[str, Any] = {
        "total_entries": n,
        "unique_images": len(image_counter),
        "missing_images": sorted(set(missing)),
        "num_missing": len(set(missing)),
        "duplicate_images": sum(1 for c in image_counter.values() if c > 1),
        "avg_caption_length": round(sum(caption_lengths) / max(n, 1), 1),
        "min_caption_length": min(caption_lengths, default=0),
        "max_caption_length": max(caption_lengths, default=0),
        "avg_caption_words": round(sum(word_counts) / max(n, 1), 1),
    }

    logger.info("Dataset validation for %s:", annotations_path)
    for k, v in stats.items():
        if k == "missing_images":
            continue  # list can be huge
        logger.info("  %s: %s", k, v)

    return stats


# =========================================================================
# Caption generation placeholder
# =========================================================================

def generate_captions_with_api(
    image_paths: Sequence[str | Path],
    *,
    api_key: str | None = None,
    model: str = "gpt-4o",
    prompt: str = "Describe this image in one sentence.",
    output_path: str | Path = "data/generated_captions.jsonl",
) -> Path:
    """Placeholder: generate captions for images via an external vision-language API.

    This function is a scaffold for bootstrapping captions on a custom dataset.
    Replace the inner loop with actual API calls to OpenAI, Anthropic, etc.

    Parameters
    ----------
    image_paths:
        List of image file paths to caption.
    api_key:
        API key for the external service.
    model:
        Model identifier to use.
    prompt:
        The text prompt sent alongside each image.
    output_path:
        Where to write the resulting JSONL.

    Returns
    -------
    Path to the written JSONL file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "generate_captions_with_api: %d images -> %s  [model=%s]",
        len(image_paths), output_path, model,
    )

    with open(output_path, "w") as f:
        for img_path in tqdm(image_paths, desc="Generating captions"):
            # ----------------------------------------------------------
            # TODO: replace with real API call, e.g.:
            #   import openai
            #   response = openai.chat.completions.create(
            #       model=model,
            #       messages=[{
            #           "role": "user",
            #           "content": [
            #               {"type": "text", "text": prompt},
            #               {"type": "image_url",
            #                "image_url": {"url": f"data:image/jpeg;base64,{b64img}"}},
            #           ],
            #       }],
            #   )
            #   caption = response.choices[0].message.content
            # ----------------------------------------------------------
            caption = f"[PLACEHOLDER] Caption for {Path(img_path).name}"

            line = json.dumps({
                "image": str(Path(img_path).name),
                "caption": caption,
            })
            f.write(line + "\n")

    logger.info("Wrote placeholder captions to %s", output_path)
    return output_path


# =========================================================================
# Internal helpers
# =========================================================================

def _load_jsonl(path: str | Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _load_annotations(path: str | Path) -> list[dict]:
    """Load annotations from JSON or JSONL, flattening multi-caption rows."""
    path = Path(path)
    if path.suffix == ".jsonl":
        raw = _load_jsonl(path)
    else:
        with open(path) as f:
            raw = json.load(f)

    flat: list[dict] = []
    for entry in raw:
        if "captions" in entry:
            for cap in entry["captions"]:
                flat.append({"image": entry["image"], "caption": cap})
        else:
            flat.append(entry)
    return flat


def _write_jsonl(entries: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


# =========================================================================
# CLI
# =========================================================================

def _cli() -> None:
    """Minimal CLI for running utilities standalone."""
    import argparse
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="VLM captioning data utilities")
    sub = parser.add_subparsers(dest="command")

    # -- download-coco --
    dl = sub.add_parser("download-coco", help="Download COCO Captions")
    dl.add_argument("--root", default="data/coco")
    dl.add_argument("--year", type=int, default=2017)
    dl.add_argument("--skip-images", action="store_true")

    # -- convert-coco --
    cv = sub.add_parser("convert-coco", help="Convert COCO annotations to JSONL")
    cv.add_argument("--annotation", required=True, help="Path to COCO captions JSON")
    cv.add_argument("--image-dir", required=True)
    cv.add_argument("--output", required=True)

    # -- convert-flickr --
    fl = sub.add_parser("convert-flickr", help="Convert Flickr30k token file to JSONL")
    fl.add_argument("--token-file", required=True)
    fl.add_argument("--output", required=True)

    # -- split --
    sp = sub.add_parser("split", help="Split JSONL into train/val/test")
    sp.add_argument("--input", required=True)
    sp.add_argument("--output-dir", required=True)
    sp.add_argument("--train-ratio", type=float, default=0.8)
    sp.add_argument("--val-ratio", type=float, default=0.1)
    sp.add_argument("--test-ratio", type=float, default=0.1)
    sp.add_argument("--seed", type=int, default=42)

    # -- validate --
    va = sub.add_parser("validate", help="Validate a dataset")
    va.add_argument("--annotations", required=True)
    va.add_argument("--image-root", required=True)

    args = parser.parse_args()

    if args.command == "download-coco":
        download_coco_captions(args.root, args.year, args.skip_images)
    elif args.command == "convert-coco":
        convert_coco_to_jsonl(args.annotation, args.image_dir, args.output)
    elif args.command == "convert-flickr":
        convert_flickr30k_to_jsonl(args.token_file, args.output)
    elif args.command == "split":
        create_train_val_test_split(
            args.input, args.output_dir,
            args.train_ratio, args.val_ratio, args.test_ratio, args.seed,
        )
    elif args.command == "validate":
        stats = validate_dataset(args.annotations, args.image_root)
        print(json.dumps(stats, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
