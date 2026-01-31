"""
Tokenization and collation for VLM captioning (using model processor/tokenizer).
"""

from typing import Any, Dict, List, Optional, Union

from transformers import AutoProcessor, AutoTokenizer


def get_processor(
    model_name: str,
    processor_name: Optional[str] = None,
    **kwargs: Any,
) -> Union[AutoProcessor, AutoTokenizer]:
    """Load HuggingFace processor or tokenizer for the VLM."""
    name = processor_name or model_name
    try:
        return AutoProcessor.from_pretrained(name, **kwargs)
    except Exception:
        return AutoTokenizer.from_pretrained(name, **kwargs)


def tokenize_captions(
    captions: List[str],
    processor: Union[AutoProcessor, AutoTokenizer],
    max_length: int = 512,
    padding: bool = True,
    truncation: bool = True,
    return_tensors: str = "pt",
) -> Dict[str, Any]:
    """
    Tokenize caption strings for training/eval.

    Args:
        captions: List of caption strings.
        processor: HF processor or tokenizer.
        max_length: Max sequence length.
        padding: Whether to pad to max_length.
        truncation: Whether to truncate.
        return_tensors: 'pt' for PyTorch.

    Returns:
        Dict with input_ids, attention_mask, etc.
    """
    out = processor(
        text=captions,
        max_length=max_length,
        padding="max_length" if padding else False,
        truncation=truncation,
        return_tensors=return_tensors,
    )
    return out
