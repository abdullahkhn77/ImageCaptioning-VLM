"""Fine-tune a Vision-Language Model for image captioning."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from dataset import CaptionDataset


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict):
    model_name = cfg["model"]["name"]
    processor = AutoProcessor.from_pretrained(
        cfg["model"]["processor"] or model_name
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if cfg["hardware"]["fp16"] else torch.float32,
    )

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

    return model, processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, processor = build_model(cfg)

    train_dataset = CaptionDataset(
        cfg["data"]["train_annotations"],
        cfg["data"]["image_root"],
        processor,
        max_length=cfg["data"]["max_length"],
    )
    val_dataset = CaptionDataset(
        cfg["data"]["val_annotations"],
        cfg["data"]["image_root"],
        processor,
        max_length=cfg["data"]["max_length"],
    )

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(Path(cfg["training"]["output_dir"]) / "final")
    processor.save_pretrained(Path(cfg["training"]["output_dir"]) / "final")


if __name__ == "__main__":
    main()
