"""Dataset class for image-caption pairs."""

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class CaptionDataset(Dataset):
    """Loads image-caption pairs from a JSON annotation file.

    Expected JSON format: a list of {"image": "<filename>", "caption": "<text>"}.
    """

    def __init__(self, annotations_path: str, image_root: str, processor, max_length: int = 128):
        with open(annotations_path) as f:
            self.annotations = json.load(f)
        self.image_root = Path(image_root)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        entry = self.annotations[idx]
        image = Image.open(self.image_root / entry["image"]).convert("RGB")
        caption = entry["caption"]

        inputs = self.processor(
            images=image,
            text=caption,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Squeeze batch dimension added by the processor
        return {k: v.squeeze(0) for k, v in inputs.items()}
