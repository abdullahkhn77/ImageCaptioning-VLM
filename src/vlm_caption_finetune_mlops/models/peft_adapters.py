"""
PEFT LoRA / QLoRA setup for VLM fine-tuning.
"""

from typing import Any, Dict, List, Optional

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch.nn as nn


def setup_peft_model(
    model: nn.Module,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
    use_qlora: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """
    Wrap model with LoRA (or QLoRA) adapters.

    Args:
        model: Base VLM from HuggingFace.
        r: LoRA rank.
        lora_alpha: LoRA alpha scaling.
        lora_dropout: LoRA dropout.
        target_modules: List of module names to apply LoRA (e.g. q_proj, v_proj).
        bias: Bias training ('none', 'all', 'lora_only').
        task_type: PEFT task type (CAUSAL_LM for generation).
        use_qlora: If True, prepare for 4-bit and use QLoRA.
        **kwargs: Passed to LoraConfig.

    Returns:
        PEFT model ready for training.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    if use_qlora:
        model = prepare_model_for_kbit_training(model, **kwargs.get("qlora_kwargs", {}))

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type,
        **{k: v for k, v in kwargs.items() if k != "qlora_kwargs"},
    )
    return get_peft_model(model, lora_config)
