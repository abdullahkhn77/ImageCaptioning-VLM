"""
Data preprocessing script for VLM fine-tuning.
Handles image resizing, caption tokenization, and data validation.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image
import pandas as pd
from transformers import BlipProcessor
from sklearn.model_selection import train_test_split
import dvc.api


def load_dataset(data_path: str) -> pd.DataFrame:
    """Load dataset from JSON or CSV file."""
    data_path = Path(data_path)
    
    if data_path.suffix == '.json':
        with open(data_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    elif data_path.suffix == '.csv':
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    return df


def preprocess_image(image_path: str, target_size: int = 224) -> Image.Image:
    """Resize and normalize image."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    return image


def validate_data(df: pd.DataFrame, image_dir: str) -> pd.DataFrame:
    """Validate data: check for missing images, duplicates, and imbalances."""
    print("Validating data...")
    
    # Check for missing images
    valid_rows = []
    for idx, row in df.iterrows():
        image_path = os.path.join(image_dir, row['image_path'])
        if os.path.exists(image_path):
            valid_rows.append(idx)
        else:
            print(f"Warning: Missing image at {image_path}")
    
    df_valid = df.loc[valid_rows].copy()
    
    # Check for duplicates
    duplicates = df_valid.duplicated(subset=['image_path'])
    if duplicates.any():
        print(f"Warning: Found {duplicates.sum()} duplicate images")
        df_valid = df_valid[~duplicates]
    
    # Check caption length
    df_valid['caption_length'] = df_valid['caption'].str.len()
    print(f"Caption length stats: min={df_valid['caption_length'].min()}, "
          f"max={df_valid['caption_length'].max()}, "
          f"mean={df_valid['caption_length'].mean():.2f}")
    
    return df_valid


def split_data(df: pd.DataFrame, train_ratio: float = 0.8, 
               val_ratio: float = 0.1, test_ratio: float = 0.1) -> Tuple[pd.DataFrame, ...]:
    """Split data into train, validation, and test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df, test_size=(val_ratio + test_ratio), random_state=42, shuffle=True
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_size), random_state=42, shuffle=True
    )
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def preprocess_dataset(data_path: str, image_dir: str, output_dir: str,
                       target_size: int = 224, processor_name: str = "Salesforce/blip-image-captioning-base"):
    """Main preprocessing function."""
    print("Loading dataset...")
    df = load_dataset(data_path)
    
    print(f"Loaded {len(df)} samples")
    
    # Validate data
    df_valid = validate_data(df, image_dir)
    
    # Split data
    train_df, val_df, test_df = split_data(df_valid)
    
    # Load processor for tokenization
    processor = BlipProcessor.from_pretrained(processor_name)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        output_path = output_dir / f"{split_name}.json"
        split_df.to_json(output_path, orient='records', indent=2)
        print(f"Saved {split_name} split to {output_path}")
    
    # Save metadata
    metadata = {
        'total_samples': len(df_valid),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'image_size': target_size,
        'processor': processor_name
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Preprocessing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset for VLM fine-tuning")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset file")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--target_size", type=int, default=224, help="Target image size")
    parser.add_argument("--processor", type=str, default="Salesforce/blip-image-captioning-base",
                       help="Processor name")
    
    args = parser.parse_args()
    preprocess_dataset(
        args.data_path,
        args.image_dir,
        args.output_dir,
        args.target_size,
        args.processor
    )

