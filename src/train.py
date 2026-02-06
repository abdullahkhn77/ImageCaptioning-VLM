"""Fine-tune a Vision-Language Model for image captioning."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForVision2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from dataset import CaptionDataset, build_eval_transforms, build_train_transforms, caption_collate_fn
from preprocessing import CaptioningProcessor

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    """Load a YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict):
    """Instantiate the VLM and optionally wrap it with LoRA."""
    model_name = cfg["model"]["name"]

    dtype = torch.float32
    if cfg["hardware"]["bf16"]:
        dtype = torch.bfloat16
    elif cfg["hardware"]["fp16"]:
        dtype = torch.float16

    model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=dtype)

    if cfg["model"]["freeze_vision_encoder"]:
        for param in model.vision_model.parameters():
            param.requires_grad = False

    if cfg["lora"]["enabled"]:
        task_type = getattr(TaskType, cfg["lora"]["task_type"])
        lora_config = LoraConfig(
            r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"]["dropout"],
            bias=cfg["lora"]["bias"],
            target_modules=cfg["lora"]["target_modules"],
            task_type=task_type,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Fine-tune a VLM for captioning")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── Processor ───────────────────────────────────────────────────────
    processor = CaptioningProcessor.from_config(cfg)

    # ── Model ───────────────────────────────────────────────────────────
    model = build_model(cfg)

    # ── Transforms ──────────────────────────────────────────────────────
    image_size = cfg["data"]["image_size"]
    aug_cfg = cfg["data"].get("augmentation", {})
    train_transforms = (
        build_train_transforms(aug_cfg, image_size)
        if aug_cfg.get("enabled", False)
        else build_eval_transforms(image_size)
    )
    eval_transforms = build_eval_transforms(image_size)

    # ── Datasets ────────────────────────────────────────────────────────
    train_dataset = CaptionDataset(
        annotations_path=cfg["data"]["train_annotations"],
        image_root=cfg["data"]["image_root"],
        processor=processor,
        max_length=cfg["data"]["max_length"],
        image_transforms=train_transforms,
    )
    val_dataset = CaptionDataset(
        annotations_path=cfg["data"]["val_annotations"],
        image_root=cfg["data"]["image_root"],
        processor=processor,
        max_length=cfg["data"]["max_length"],
        image_transforms=eval_transforms,
    )

    # ── Training args ───────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=cfg["training"]["output_dir"],
        num_train_epochs=cfg["training"]["num_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        max_grad_norm=cfg["training"]["max_grad_norm"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        logging_steps=cfg["training"]["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["training"]["eval_steps"],
        save_steps=cfg["training"]["save_steps"],
        save_total_limit=cfg["training"]["save_total_limit"],
        fp16=cfg["hardware"]["fp16"],
        bf16=cfg["hardware"]["bf16"],
        gradient_checkpointing=cfg["hardware"]["gradient_checkpointing"],
        dataloader_num_workers=cfg["hardware"]["dataloader_num_workers"],
        dataloader_pin_memory=cfg["hardware"]["dataloader_pin_memory"],
        deepspeed=cfg["hardware"]["deepspeed_config"],
        seed=cfg["training"]["seed"],
        load_best_model_at_end=True,
        metric_for_best_model=cfg["training"]["early_stopping_metric"],
        report_to="wandb" if cfg["wandb"]["enabled"] else "none",
        run_name=cfg["wandb"].get("run_name"),
    )

    callbacks = []
    patience = cfg["training"]["early_stopping_patience"]
    if patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    # ── Trainer ─────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=caption_collate_fn,
        callbacks=callbacks,
    )

    trainer.train()

    output_dir = Path(cfg["training"]["output_dir"]) / "final"
    trainer.save_model(output_dir)
    processor.hf_processor.save_pretrained(output_dir)
    logger.info("Model and processor saved to %s", output_dir)


if __name__ == "__main__":
    main()
