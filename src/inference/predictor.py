"""
Caption predictor: load model + processor and generate captions for images.
"""

from pathlib import Path
from typing import Any, List, Optional, Union

from PIL import Image


class CaptionPredictor:
    """
    Wrapper to run VLM inference for image captioning.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        device: Optional[str] = None,
        max_new_tokens: int = 128,
        **generate_kwargs: Any,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_new_tokens = max_new_tokens
        self.generate_kwargs = generate_kwargs

    def predict(
        self,
        image: Union[Image.Image, str, Path],
        prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a caption for a single image.

        Args:
            image: PIL Image or path to image file.
            prompt: Optional prefix prompt (e.g. "caption: " for PaliGemma).

        Returns:
            Generated caption string.
        """
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        # Model-specific encoding; in practice use processor(image=..., text=prompt)
        inputs = self.processor(images=image, text=prompt or "", return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            **self.generate_kwargs,
        )
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption.strip()

    def predict_batch(
        self,
        images: List[Union[Image.Image, str, Path]],
        prompt: Optional[str] = None,
    ) -> List[str]:
        """Generate captions for a list of images."""
        return [self.predict(img, prompt) for img in images]


def load_predictor(
    checkpoint_dir: str,
    device: Optional[str] = None,
    max_new_tokens: int = 128,
    **kwargs: Any,
) -> CaptionPredictor:
    """
    Load model and processor from checkpoint and return CaptionPredictor.

    Args:
        checkpoint_dir: Path to saved model (HF format or PEFT adapter).
        device: Device to run on.
        max_new_tokens: Max tokens to generate.
        **kwargs: Passed to CaptionPredictor.

    Returns:
        CaptionPredictor instance.
    """
    from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForVision2Seq
    import torch
    path = Path(checkpoint_dir)
    try:
        processor = AutoProcessor.from_pretrained(str(path), **kwargs)
    except Exception:
        from transformers import AutoTokenizer
        processor = AutoTokenizer.from_pretrained(str(path), **kwargs)
    try:
        model = AutoModelForVision2Seq.from_pretrained(str(path), torch_dtype=torch.bfloat16, **kwargs)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch.bfloat16, **kwargs)
    return CaptionPredictor(model, processor, device=device, max_new_tokens=max_new_tokens, **kwargs)
