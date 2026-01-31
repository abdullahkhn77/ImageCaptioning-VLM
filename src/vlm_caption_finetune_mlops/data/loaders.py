"""
Dataset loaders for image captioning: COCO, Flickr8k/30k, custom (image path + caption).
"""

from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, load_dataset


def get_captioning_dataset(
    name: str = "coco",
    split: str = "train",
    data_dir: Optional[Path] = None,
    custom_train_path: Optional[str] = None,
    custom_eval_path: Optional[str] = None,
    image_column: str = "image_path",
    caption_column: str = "caption",
    max_samples: Optional[int] = None,
    **kwargs: Any,
) -> Dataset:
    """
    Load a captioning dataset (COCO, Flickr8k/30k, or custom CSV/JSON).

    Args:
        name: Dataset name ('coco', 'flickr8k', 'flickr30k', 'custom').
        split: Dataset split ('train', 'val', 'test').
        data_dir: Root data directory for local files.
        custom_train_path: Path to custom train file (CSV/JSON).
        custom_eval_path: Path to custom eval file (CSV/JSON).
        image_column: Column name for image path or image.
        caption_column: Column name for caption text.
        max_samples: Cap number of samples (None = use all).
        **kwargs: Passed to HuggingFace load_dataset or pandas read.

    Returns:
        HuggingFace Dataset with at least image (or path) and caption.
    """
    if name == "custom":
        return _load_custom_dataset(
            split=split,
            train_path=custom_train_path,
            eval_path=custom_eval_path,
            image_column=image_column,
            caption_column=caption_column,
            max_samples=max_samples,
            **kwargs,
        )
    if name in ("coco", "flickr8k", "flickr30k"):
        return _load_hf_or_builtin(
            name=name,
            split=split,
            data_dir=data_dir,
            max_samples=max_samples,
            **kwargs,
        )
    # Generic HF dataset
    ds = load_dataset(name, **kwargs)
    if split in ds:
        out = ds[split]
    else:
        out = ds["train"] if "train" in ds else list(ds.values())[0]
    if max_samples is not None:
        out = out.select(range(min(max_samples, len(out))))
    return out


def _load_custom_dataset(
    split: str,
    train_path: Optional[str] = None,
    eval_path: Optional[str] = None,
    image_column: str = "image_path",
    caption_column: str = "caption",
    max_samples: Optional[int] = None,
    **kwargs: Any,
) -> Dataset:
    """Load custom CSV/JSON with image_path and caption columns."""
    import pandas as pd

    path = train_path if split == "train" else eval_path
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"Custom {split} path not found: {path}")
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path, **kwargs)
    elif ext in (".json", ".jsonl"):
        df = pd.read_json(path, lines=(ext == ".jsonl"), **kwargs)
    else:
        raise ValueError(f"Unsupported custom format: {ext}")
    if image_column not in df or caption_column not in df:
        raise ValueError(f"Need columns {image_column} and {caption_column}")
    if max_samples is not None:
        df = df.head(max_samples)
    return Dataset.from_pandas(df[[image_column, caption_column]].rename(columns={image_column: "image_path", caption_column: "caption"}))


def _load_hf_or_builtin(
    name: str,
    split: str,
    data_dir: Optional[Path] = None,
    max_samples: Optional[int] = None,
    **kwargs: Any,
) -> Dataset:
    """Load COCO/Flickr via HuggingFace Datasets or built-in helper."""
    # Example: use a public HF image-caption dataset; replace with actual COCO/Flickr IDs if needed
    hf_name = {
        "coco": "nlpconnect/vit-gpt2-image-captioning",
        "flickr8k": "nlpconnect/vit-gpt2-image-captioning",
        "flickr30k": "nlpconnect/vit-gpt2-image-captioning",
    }.get(name, name)
    ds = load_dataset(hf_name, **kwargs)
    key = "train" if split == "train" else "validation" if split == "val" else "test"
    if key not in ds:
        key = list(ds.keys())[0]
    out = ds[key]
    if max_samples is not None:
        out = out.select(range(min(max_samples, len(out))))
    return out
