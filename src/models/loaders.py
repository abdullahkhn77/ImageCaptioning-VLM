"""
Load Vision-Language Model and processor (PaliGemma, LLaVA, Qwen2-VL, Llama 3.2 Vision).
"""

from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, AutoModelForVision2Seq


def load_vlm_and_processor(
    model_name: str,
    processor_name: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    revision: str = "main",
    trust_remote_code: bool = True,
    device_map: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """
    Load VLM and its processor/tokenizer from HuggingFace.

    Args:
        model_name: HF model id (e.g. google/paligemma-3b-pt-224).
        processor_name: Override processor id (default: model_name).
        torch_dtype: Model dtype (bfloat16, float16, auto).
        revision: Git revision.
        trust_remote_code: Allow custom code.
        device_map: Device map for multi-GPU ('auto', 'cuda:0', etc.).
        **kwargs: Passed to from_pretrained.

    Returns:
        (model, processor_or_tokenizer)
    """
    if torch_dtype is None or torch_dtype == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif isinstance(torch_dtype, str):
        dtype = getattr(torch, torch_dtype, torch.float16)
    else:
        dtype = torch_dtype

    proc_name = processor_name or model_name
    try:
        processor = AutoProcessor.from_pretrained(
            proc_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
    except Exception:
        processor = AutoTokenizer.from_pretrained(
            proc_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    try:
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            device_map=device_map or "auto",
            **kwargs,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            device_map=device_map or "auto",
            **kwargs,
        )

    return model, processor
