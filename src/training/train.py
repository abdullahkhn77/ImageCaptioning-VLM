"""
Fine-tuning script for BLIP model with MLflow tracking.
"""

import os
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration,
    AdamW,
    get_linear_schedule_with_warmup
)
from PIL import Image
import pandas as pd
import json
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import numpy as np


class ImageCaptionDataset(Dataset):
    """Dataset class for image-caption pairs."""
    
    def __init__(self, data_path: str, processor: BlipProcessor, image_dir: str):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.processor = processor
        self.image_dir = Path(image_dir)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = self.image_dir / item['image_path']
        caption = item['caption']
        
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        
        # Process image and caption
        inputs = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=50,
            return_tensors="pt"
        )
        
        # Flatten tensors
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        
        return inputs


def compute_metrics(predictions: list, references: list) -> Dict[str, float]:
    """Compute BLEU, ROUGE, and other metrics."""
    # Placeholder for actual metric computation
    # In practice, use nltk for BLEU and rouge-score for ROUGE
    return {
        'bleu': 0.0,  # TODO: Implement BLEU
        'rouge': 0.0,  # TODO: Implement ROUGE
        'cider': 0.0  # TODO: Implement CIDEr
    }


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in progress_bar:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, processor, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get loss
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            
            # Generate captions
            pixel_values = batch['pixel_values']
            generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
            generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Get reference captions
            input_ids = batch['input_ids']
            reference_captions = processor.batch_decode(input_ids, skip_special_tokens=True)
            
            predictions.extend(generated_captions)
            references.extend(reference_captions)
    
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(predictions, references)
    metrics['loss'] = avg_loss
    
    return metrics


def train(config_path: str):
    """Main training function."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load processor and model
    processor = BlipProcessor.from_pretrained(config['model']['name'])
    model = BlipForConditionalGeneration.from_pretrained(config['model']['name'])
    model.to(device)
    
    # Setup MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    # Load datasets
    data_dir = Path(config['paths']['data_dir'])
    train_dataset = ImageCaptionDataset(
        data_dir / 'train.json',
        processor,
        data_dir.parent / 'raw'
    )
    val_dataset = ImageCaptionDataset(
        data_dir / 'val.json',
        processor,
        data_dir.parent / 'raw'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    total_steps = len(train_loader) * config['training']['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['training']['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            'learning_rate': config['training']['learning_rate'],
            'batch_size': config['training']['batch_size'],
            'num_epochs': config['training']['num_epochs'],
            'model_name': config['model']['name']
        })
        
        # Training loop
        best_val_loss = float('inf')
        output_dir = Path(config['paths']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(1, config['training']['num_epochs'] + 1):
            print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
            
            # Train
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            
            # Evaluate
            if epoch % config['training']['eval_steps'] == 0:
                val_metrics = evaluate(model, val_loader, processor, device)
                mlflow.log_metrics(val_metrics, step=epoch)
                
                # Save best model
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    processor.save_pretrained(checkpoint_dir)
                    print(f"Saved best model to {checkpoint_dir}")
        
        # Log final model
        if config['mlflow']['log_model']:
            mlflow.pytorch.log_model(model, "model")
        
        print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BLIP model for image captioning")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                       help="Path to training config file")
    
    args = parser.parse_args()
    train(args.config)

