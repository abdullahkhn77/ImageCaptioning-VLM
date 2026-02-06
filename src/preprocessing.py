"""Processor wrapper that normalises model-specific preprocessing differences."""

from __future__ import annotations

import logging
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor

logger = logging.getLogger(__name__)

# Maps model-family prefixes to the prompt template each model expects when
# the user provides a captioning task.  ``{caption}`` is replaced with the
# ground-truth caption during training; for inference it is left out.
_PROMPT_TEMPLATES: dict[str, dict[str, str]] = {
    "blip2": {
        "train": "{caption}",
        "inference": "",
    },
    "llava": {
        "train": "USER: <image>\nDescribe this image.\nASSISTANT: {caption}",
        "inference": "USER: <image>\nDescribe this image.\nASSISTANT:",
    },
    "florence": {
        "train": "<CAPTION> {caption}",
        "inference": "<CAPTION>",
    },
    "qwen": {
        "train": "Describe this image: {caption}",
        "inference": "Describe this image:",
    },
}


def _detect_family(model_name: str) -> str:
    """Return a canonical family key from a HuggingFace model identifier."""
    name_lower = model_name.lower()
    if "blip2" in name_lower or "blip-2" in name_lower:
        return "blip2"
    if "llava" in name_lower:
        return "llava"
    if "florence" in name_lower:
        return "florence"
    if "qwen" in name_lower:
        return "qwen"
    # Default: assume BLIP-2–style (no prompt wrapping).
    logger.warning(
        "Could not detect model family for '%s'; defaulting to blip2-style prompting.",
        model_name,
    )
    return "blip2"


class CaptioningProcessor:
    """Unified interface over HuggingFace ``AutoProcessor`` for captioning.

    Handles model-specific prompt templates and preprocessing so the rest of
    the codebase can be model-agnostic.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier (e.g. ``Salesforce/blip2-opt-2.7b``).
    processor_name:
        Override for the processor checkpoint.  Defaults to *model_name*.
    max_length:
        Default max token length for caption encoding.
    image_size:
        Expected image resolution (informational — the HF processor handles
        its own resizing; this value is exposed for external transform
        pipelines).
    """

    def __init__(
        self,
        model_name: str,
        processor_name: str | None = None,
        max_length: int = 128,
        image_size: int = 224,
    ) -> None:
        self.model_name = model_name
        self.family = _detect_family(model_name)
        self.max_length = max_length
        self.image_size = image_size

        self._processor = AutoProcessor.from_pretrained(processor_name or model_name)

        templates = _PROMPT_TEMPLATES[self.family]
        self._train_template = templates["train"]
        self._inference_template = templates["inference"]

        logger.info(
            "CaptioningProcessor ready — family=%s, model=%s",
            self.family, model_name,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_for_training(
        self,
        image: Image.Image,
        caption: str,
        max_length: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Preprocess a single (image, caption) pair for training.

        Returns a dict of **squeezed** tensors (no batch dimension) that the
        :class:`CaptionDataset` can store directly.
        """
        max_length = max_length or self.max_length
        text = self._train_template.format(caption=caption)

        inputs = self._processor(
            images=image,
            text=text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Squeeze the batch-of-1 dimension added by the processor
        squeezed: dict[str, torch.Tensor] = {
            k: v.squeeze(0) for k, v in inputs.items()
        }

        # Build labels: copy input_ids but mask padding tokens with -100 so
        # they are ignored by the cross-entropy loss.
        if "input_ids" in squeezed:
            labels = squeezed["input_ids"].clone()
            if "attention_mask" in squeezed:
                labels[squeezed["attention_mask"] == 0] = -100
            squeezed["labels"] = labels

        return squeezed

    def process_for_inference(
        self,
        image: Image.Image,
    ) -> dict[str, torch.Tensor]:
        """Preprocess a single image for generation (no caption)."""
        text = self._inference_template if self._inference_template else None

        kwargs: dict[str, Any] = {"images": image, "return_tensors": "pt"}
        if text:
            kwargs["text"] = text

        return self._processor(**kwargs)

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to a string, stripping special tokens."""
        return self._processor.batch_decode(
            token_ids, skip_special_tokens=True,
        )[0].strip()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict) -> "CaptioningProcessor":
        """Construct from the full YAML config dict."""
        return cls(
            model_name=cfg["model"]["name"],
            processor_name=cfg["model"].get("processor"),
            max_length=cfg["data"]["max_length"],
            image_size=cfg["data"]["image_size"],
        )

    @property
    def hf_processor(self) -> Any:
        """Access the underlying HuggingFace processor directly."""
        return self._processor
