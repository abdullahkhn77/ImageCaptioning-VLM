"""PyTorch Dataset for image-caption pairs with configurable transforms."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from preprocessing import CaptioningProcessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_annotations(path: str | Path) -> list[dict[str, Any]]:
    """Load annotations from a JSON array or a JSONL file.

    Supports two multi-caption layouts:
      1. One row per (image, caption) pair  – ``{"image": "...", "caption": "..."}``
      2. One row per image with a list       – ``{"image": "...", "captions": ["...", ...]}``

    Layout 2 is flattened so every returned dict has a single ``caption`` key.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        entries: list[dict[str, Any]] = []
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSONL at %s:%d", path, lineno)
    elif suffix == ".json":
        with open(path) as f:
            entries = json.load(f)
    else:
        raise ValueError(f"Unsupported annotation format: {suffix} (expected .json or .jsonl)")

    # Flatten multi-caption entries
    flat: list[dict[str, str]] = []
    for entry in entries:
        if "captions" in entry:
            for cap in entry["captions"]:
                flat.append({"image": entry["image"], "caption": cap})
        else:
            flat.append({"image": entry["image"], "caption": entry["caption"]})

    return flat


def build_train_transforms(cfg_aug: dict, image_size: int) -> T.Compose:
    """Build torchvision augmentation pipeline from the config ``data.augmentation`` block."""
    ops: list[Any] = []

    if cfg_aug.get("random_resized_crop", {}).get("enabled", False):
        scale = tuple(cfg_aug["random_resized_crop"]["scale"])
        ops.append(T.RandomResizedCrop(image_size, scale=scale))
    else:
        ops.append(T.Resize((image_size, image_size)))

    if cfg_aug.get("random_horizontal_flip", 0) > 0:
        ops.append(T.RandomHorizontalFlip(p=cfg_aug["random_horizontal_flip"]))

    cj = cfg_aug.get("color_jitter")
    if cj:
        ops.append(T.ColorJitter(
            brightness=cj.get("brightness", 0),
            contrast=cj.get("contrast", 0),
            saturation=cj.get("saturation", 0),
            hue=cj.get("hue", 0),
        ))

    return T.Compose(ops)


def build_eval_transforms(image_size: int) -> T.Compose:
    """Deterministic resize only — no augmentation."""
    return T.Compose([T.Resize((image_size, image_size))])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CaptionDataset(Dataset):
    """Image-caption dataset backed by a JSON / JSONL annotation file.

    Parameters
    ----------
    annotations_path:
        Path to a ``.json`` or ``.jsonl`` annotation file.
    image_root:
        Directory that contains the image files referenced in the annotations.
    processor:
        A :class:`CaptioningProcessor` instance (wraps the HF processor).
    max_length:
        Maximum token length for captions.
    image_transforms:
        Optional ``torchvision.transforms.Compose`` applied to each PIL image
        **before** passing it to the HF processor.  If ``None`` no extra
        transforms are applied.
    """

    def __init__(
        self,
        annotations_path: str | Path,
        image_root: str | Path,
        processor: CaptioningProcessor,
        max_length: int = 128,
        image_transforms: T.Compose | None = None,
    ) -> None:
        self.annotations = _load_annotations(annotations_path)
        self.image_root = Path(image_root)
        self.processor = processor
        self.max_length = max_length
        self.image_transforms = image_transforms

        logger.info(
            "Loaded %d caption entries from %s (image_root=%s)",
            len(self.annotations), annotations_path, image_root,
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.annotations)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        entry = self.annotations[idx]
        image_path = self.image_root / entry["image"]

        # Robust image loading
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            logger.error("Image not found: %s — returning blank placeholder", image_path)
            image = Image.new("RGB", (224, 224))
        except Exception as exc:
            logger.error("Failed to load %s (%s) — returning blank placeholder", image_path, exc)
            image = Image.new("RGB", (224, 224))

        # Optional augmentation (before the HF processor's own resize/normalize)
        if self.image_transforms is not None:
            image = self.image_transforms(image)

        caption: str = entry["caption"]

        return self.processor.process_for_training(
            image=image,
            caption=caption,
            max_length=self.max_length,
        )


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def caption_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate variable-length sequences with right-padding.

    Each sample is a dict of tensors (already 1-D after squeezing in the
    processor).  This function pads every tensor key to the longest length in
    the batch and stacks them into a single batched dict.
    """
    keys = batch[0].keys()
    collated: dict[str, torch.Tensor] = {}

    for key in keys:
        tensors = [sample[key] for sample in batch]

        # Only pad 1-D tensors (token ids, attention masks).  Higher-dim
        # tensors (pixel_values) just get stacked.
        if tensors[0].dim() == 1:
            max_len = max(t.size(0) for t in tensors)
            padded = []
            for t in tensors:
                pad_size = max_len - t.size(0)
                if pad_size > 0:
                    # Pad with 0 (works for input_ids, attention_mask, labels)
                    t = torch.nn.functional.pad(t, (0, pad_size), value=0)
                padded.append(t)
            collated[key] = torch.stack(padded)
        else:
            collated[key] = torch.stack(tensors)

    return collated
