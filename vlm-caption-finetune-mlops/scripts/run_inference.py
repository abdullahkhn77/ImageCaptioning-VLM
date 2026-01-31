#!/usr/bin/env python3
"""
Run inference: generate captions for image(s) using a trained checkpoint.
Use: python scripts/run_inference.py --config-name default checkpoint_dir=path image_path=img.jpg
"""

from pathlib import Path
from typing import Optional

import yaml


def load_config(config_name: str = "default", overrides: Optional[list] = None) -> dict:
    """Load merged config."""
    base = Path(__file__).resolve().parent.parent / "configs" / config_name
    out: dict = {}
    for f in ("config.yaml", "model.yaml"):
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
    parser = argparse.ArgumentParser(description="Run VLM caption inference")
    parser.add_argument("--config-name", default="default")
    parser.add_argument("--image-path", type=str, help="Path to image file")
    parser.add_argument("overrides", nargs="*", help="e.g. checkpoint_dir=path")
    args = parser.parse_args()
    cfg = load_config(args.config_name, args.overrides)
    ckpt = cfg.get("checkpoint_dir") or next((o.split("=", 1)[1] for o in args.overrides if o.startswith("checkpoint_dir=")), None)
    image_path = getattr(args, "image_path", None) or cfg.get("image_path") or next((o.split("=", 1)[1] for o in args.overrides if o.startswith("image_path=")), None)
    if not ckpt or not Path(ckpt).exists():
        print("Set checkpoint_dir to an existing path")
        return
    if not image_path or not Path(image_path).exists():
        print("Set image_path to an existing image file")
        return

    from vlm_caption_finetune_mlops.inference import load_predictor

    predictor = load_predictor(ckpt)
    caption = predictor.predict(image_path)
    print("Caption:", caption)


if __name__ == "__main__":
    main()
