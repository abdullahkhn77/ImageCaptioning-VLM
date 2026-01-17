"""
Evaluation script for computing BLEU, ROUGE, and CIDEr metrics.
"""

import argparse
import json
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np
from tqdm import tqdm

from src.training.train import ImageCaptionDataset


def compute_bleu(references: list, predictions: list) -> float:
    """Compute BLEU score."""
    smoothing = SmoothingFunction().method1
    bleu_scores = []
    
    for ref, pred in zip(references, predictions):
        ref_tokens = ref.lower().split()
        pred_tokens = pred.lower().split()
        score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
        bleu_scores.append(score)
    
    return np.mean(bleu_scores)


def compute_rouge(references: list, predictions: list) -> dict:
    """Compute ROUGE scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for ref, pred in zip(references, predictions):
        scores = scorer.score(ref, pred)
        for metric in rouge_scores.keys():
            rouge_scores[metric].append(scores[metric].fmeasure)
    
    return {k: np.mean(v) for k, v in rouge_scores.items()}


def compute_cider(references: list, predictions: list) -> float:
    """Compute CIDEr score (simplified version)."""
    # Note: Full CIDEr implementation requires pycocoevalcap
    # This is a placeholder - implement full CIDEr if needed
    # For now, return a simple n-gram overlap metric
    from collections import Counter
    
    def get_ngrams(text, n):
        tokens = text.lower().split()
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    cider_scores = []
    for ref, pred in zip(references, predictions):
        ref_ngrams = Counter(get_ngrams(ref, 1) + get_ngrams(ref, 2))
        pred_ngrams = Counter(get_ngrams(pred, 1) + get_ngrams(pred, 2))
        
        intersection = sum((ref_ngrams & pred_ngrams).values())
        union = sum((ref_ngrams | pred_ngrams).values())
        
        if union > 0:
            cider_scores.append(intersection / union)
        else:
            cider_scores.append(0.0)
    
    return np.mean(cider_scores)


def evaluate_model(model_path: str, test_data_path: str, image_dir: str, 
                   config_path: str = None):
    """Evaluate model on test set."""
    # Load config
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_name = config.get('model_name', 'Salesforce/blip-image-captioning-base')
    else:
        model_name = 'Salesforce/blip-image-captioning-base'
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load processor and model
    processor = BlipProcessor.from_pretrained(model_path if Path(model_path).exists() else model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_path if Path(model_path).exists() else model_name)
    model.to(device)
    model.eval()
    
    # Load test dataset
    test_dataset = ImageCaptionDataset(test_data_path, processor, image_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Generate predictions
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Generate caption
            pixel_values = batch['pixel_values']
            generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
            generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Get reference captions
            input_ids = batch['input_ids']
            reference_captions = processor.batch_decode(input_ids, skip_special_tokens=True)
            
            predictions.extend(generated_captions)
            references.extend(reference_captions)
    
    # Compute metrics
    print("\nComputing metrics...")
    bleu_score = compute_bleu(references, predictions)
    rouge_scores = compute_rouge(references, predictions)
    cider_score = compute_cider(references, predictions)
    
    metrics = {
        'bleu': bleu_score,
        'rouge': rouge_scores,
        'cider': cider_score
    }
    
    print("\nEvaluation Results:")
    print(f"BLEU: {bleu_score:.4f}")
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
    print(f"CIDEr: {cider_score:.4f}")
    
    # Save results
    results_path = Path(model_path).parent / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on test set")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data JSON")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--config", type=str, default=None, help="Path to model config")
    
    args = parser.parse_args()
    evaluate_model(args.model_path, args.test_data, args.image_dir, args.config)

