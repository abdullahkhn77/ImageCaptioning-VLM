#!/usr/bin/env python3
"""
Training entrypoint: fine-tune VLM for captioning with LoRA/QLoRA, logging (MLflow/W&B).
Use: python scripts/train.py --config-name default
     accelerate launch scripts/train.py --config-name default
"""

from pathlib import Path
from typing import Any, Optional

import yaml


def load_config(config_name: str = "default", overrides: Optional[list] = None) -> dict:
    """Load merged config from configs/default/."""
    base = Path(__file__).resolve().parent.parent / "configs" / config_name
    out: dict = {}
    for f in ("config.yaml", "model.yaml", "dataset.yaml", "lora.yaml"):
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
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        val = val.lower() == "true" if val.lower() in ("true", "false") else val
                d[keys[-1]] = val
    return out


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train VLM for captioning")
    parser.add_argument("--config-name", default="default")
    parser.add_argument("overrides", nargs="*", help="e.g. training.batch_size=8")
    args = parser.parse_args()
    cfg = load_config(args.config_name, args.overrides)

    training = cfg.get("training", cfg)
    model_cfg = cfg.get("model", cfg)
    dataset_cfg = cfg.get("dataset", cfg)
    lora_cfg = cfg.get("lora", {})

    if training.get("report_to") == "mlflow":
        import mlflow
        mlflow.set_experiment(cfg.get("experiment_name", "vlm_caption"))
        mlflow.start_run()
        mlflow.log_params({f"config.{k}": v for k, v in training.items()})
    if training.get("report_to") == "wandb":
        import wandb
        wandb.init(project=cfg.get("experiment_name", "vlm_caption"), config=cfg)

    from vlm_caption_finetune_mlops.models import load_vlm_and_processor, setup_peft_model
    from vlm_caption_finetune_mlops.data import get_captioning_dataset
    from vlm_caption_finetune_mlops.models.training import run_training

    model, processor = load_vlm_and_processor(
        model_name=model_cfg.get("model", {}).get("name", "google/paligemma-3b-pt-224"),
        processor_name=model_cfg.get("processor", {}).get("name"),
        torch_dtype=model_cfg.get("model", {}).get("torch_dtype", "bfloat16"),
        revision=model_cfg.get("model", {}).get("revision", "main"),
        trust_remote_code=model_cfg.get("model", {}).get("trust_remote_code", True),
    )
    if lora_cfg.get("enabled", True):
        model = setup_peft_model(
            model,
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            target_modules=lora_cfg.get("target_modules"),
            task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
            use_qlora=lora_cfg.get("qlora", {}).get("enabled", False),
        )

    train_ds = get_captioning_dataset(
        name=dataset_cfg.get("name", "coco"),
        split=dataset_cfg.get("split", "train"),
        custom_train_path=dataset_cfg.get("custom", {}).get("train_path"),
        custom_eval_path=dataset_cfg.get("custom", {}).get("eval_path"),
        image_column=dataset_cfg.get("custom", {}).get("image_path_column", "image_path"),
        caption_column=dataset_cfg.get("custom", {}).get("caption_column", "caption"),
        max_samples=dataset_cfg.get("max_samples"),
    )
    eval_ds = None
    if dataset_cfg.get("eval_split"):
        try:
            eval_ds = get_captioning_dataset(
                name=dataset_cfg.get("name", "coco"),
                split=dataset_cfg.get("eval_split", "val"),
                custom_train_path=dataset_cfg.get("custom", {}).get("train_path"),
                custom_eval_path=dataset_cfg.get("custom", {}).get("eval_path"),
                max_samples=dataset_cfg.get("max_samples"),
            )
        except Exception:
            pass

    run_training(
        model=model,
        processor=processor,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        output_dir=training.get("output_dir", "outputs/checkpoints"),
        num_epochs=training.get("num_epochs", 3),
        batch_size=training.get("batch_size", 4),
        gradient_accumulation_steps=training.get("gradient_accumulation_steps", 4),
        learning_rate=training.get("learning_rate", 2.0e-5),
        weight_decay=training.get("weight_decay", 0.01),
        warmup_ratio=training.get("warmup_ratio", 0.03),
        max_grad_norm=training.get("max_grad_norm", 1.0),
        logging_steps=training.get("logging_steps", 10),
        save_steps=training.get("save_steps", 500),
        eval_steps=training.get("eval_steps"),
        save_total_limit=training.get("save_total_limit", 3),
        bf16=training.get("bf16", True),
        fp16=training.get("fp16", False),
        gradient_checkpointing=training.get("gradient_checkpointing", True),
        report_to=training.get("report_to", "none"),
    )

    if training.get("report_to") == "mlflow":
        import mlflow
        mlflow.end_run()
    if training.get("report_to") == "wandb":
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
