"""
Inference utilities for VLM caption generation and serving.
"""

from vlm_caption_finetune_mlops.inference.predictor import CaptionPredictor, load_predictor

__all__ = [
    "CaptionPredictor",
    "load_predictor",
]
