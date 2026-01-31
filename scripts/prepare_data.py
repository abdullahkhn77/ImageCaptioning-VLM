#!/usr/bin/env python3
"""
Data preparation pipeline: load dataset (COCO/Flickr/custom) and write processed outputs.
Use with: python scripts/prepare_data.py --config-name default
"""

from pathlib import Path
from typing import Optional

import yaml


def load_config(config_name: str = "default", overrides: Optional[list] = None) -> dict:
    """Load merged config from configs/ directory."""
    base = Path(__file__).resolve().parent.parent / "configs" / config_name
    out = {}
    for f in ("config.yaml", "dataset.yaml", "model.yaml"):
        p = base / f
        if p.exists():
            with open(p) as fp:
                data = yaml.safe_load(fp) or {}
                for k, v in data.items():
                    if isinstance(v, dict) and k in out and isinstance(out[k], dict):
                        out[k].update(v)
                    else:
                        out[k] = v
    if overrides:
        for o in overrides:
            if "=" in o:
                key, val = o.split("=", 1)
                keys = key.split(".")
                d = out
                for k in keys[:-1]:
                    d = d.setdefault(k, {})
                d[keys[-1]] = val
    return out


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Prepare captioning dataset")
    parser.add_argument("--config-name", default="default", help="Config name")
    parser.add_argument("overrides", nargs="*", help="Config overrides key=val")
    args = parser.parse_args()
    cfg = load_config(args.config_name, args.overrides)
    paths = cfg.get("paths", {})
    raw_dir = Path(paths.get("raw_data", "data/raw"))
    processed_dir = Path(paths.get("processed_data", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    dataset_cfg = cfg.get("dataset", cfg)
    from vlm_caption_finetune_mlops.data import get_captioning_dataset
    ds = get_captioning_dataset(
        name=dataset_cfg.get("name", "coco"),
        split=dataset_cfg.get("split", "train"),
        data_dir=raw_dir,
        custom_train_path=dataset_cfg.get("custom", {}).get("train_path"),
        custom_eval_path=dataset_cfg.get("custom", {}).get("eval_path"),
        image_column=dataset_cfg.get("custom", {}).get("image_path_column", "image_path"),
        caption_column=dataset_cfg.get("custom", {}).get("caption_column", "caption"),
        max_samples=dataset_cfg.get("max_samples"),
    )
    out_path = processed_dir / "train"
    ds.save_to_disk(str(out_path))
    print(f"Saved processed dataset to {out_path}")


if __name__ == "__main__":
    main()
