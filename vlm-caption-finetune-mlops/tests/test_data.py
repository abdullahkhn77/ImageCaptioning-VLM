"""Tests for data loaders and preprocessing."""

from pathlib import Path

import pytest


def test_preprocess_image():
    """Test image preprocessing returns tensor of expected shape."""
    from vlm_caption_finetune_mlops.data.preprocessing import preprocess_image
    from PIL import Image
    import numpy as np

    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    out = preprocess_image(img, size=32, normalize=True)
    assert out.shape == (3, 32, 32)


def test_tokenize_captions():
    """Test tokenization returns dict with input_ids."""
    from vlm_caption_finetune_mlops.data.tokenization import tokenize_captions
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    captions = ["a cat sits", "a dog runs"]
    out = tokenize_captions(captions, tokenizer, max_length=16, padding=True)
    assert "input_ids" in out
    assert out["input_ids"].shape[0] == 2
