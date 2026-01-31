#!/usr/bin/env python3
"""
Export / push fine-tuned model to Hugging Face Hub or local path.
Use: python scripts/export_model.py --config-name default checkpoint_dir=outputs/checkpoints/final
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
    parser = argparse.ArgumentParser(description="Export VLM to HF Hub or path")
    parser.add_argument("--config-name", default="default")
    parser.add_argument("--push-to-hub", action="store_true", help="Push to Hugging Face Hub")
    parser.add_argument("--export-path", default="outputs/export", help="Local export path")
    parser.add_argument("overrides", nargs="*", help="e.g. checkpoint_dir=path")
    args = parser.parse_args()
    cfg = load_config(args.config_name, args.overrides)
    ckpt = cfg.get("checkpoint_dir") or next((o.split("=", 1)[1] for o in args.overrides if o.startswith("checkpoint_dir=")), None)
    if not ckpt or not Path(ckpt).exists():
        print("Set checkpoint_dir to an existing path")
        return

    from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForVision2Seq
    import torch

    path = Path(ckpt)
    try:
        processor = AutoProcessor.from_pretrained(str(path))
    except Exception:
        from transformers import AutoTokenizer
        processor = AutoTokenizer.from_pretrained(str(path))
    try:
        model = AutoModelForVision2Seq.from_pretrained(str(path), torch_dtype=torch.bfloat16)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch.bfloat16)

    export_path = Path(args.export_path)
    export_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(export_path)
    processor.save_pretrained(export_path)
    print(f"Exported to {export_path}")

    if args.push_to_hub:
        repo_id = cfg.get("export", {}).get("hub_repo_id") or "your-username/vlm-caption-finetuned"
        model.push_to_hub(repo_id)
        processor.push_to_hub(repo_id)
        print(f"Pushed to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
