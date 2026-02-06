"""Run inference with a fine-tuned VLM to caption a single image."""

from __future__ import annotations

import argparse
import logging

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForVision2Seq

from preprocessing import CaptioningProcessor

logger = logging.getLogger(__name__)


def caption_image(
    model: torch.nn.Module,
    processor: CaptioningProcessor,
    image_path: str,
    num_beams: int = 5,
    max_new_tokens: int = 50,
) -> str:
    """Generate a caption for a single image.

    Parameters
    ----------
    model:
        The (possibly LoRA-wrapped) VLM.
    processor:
        A :class:`CaptioningProcessor` instance.
    image_path:
        Path to the image file.
    num_beams:
        Beam search width.
    max_new_tokens:
        Maximum number of tokens to generate.

    Returns
    -------
    The generated caption string.
    """
    device = next(model.parameters()).device
    image = Image.open(image_path).convert("RGB")
    inputs = processor.process_for_inference(image).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )

    return processor.decode(output_ids)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Caption a single image")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to fine-tuned checkpoint")
    parser.add_argument("--base-model", type=str,
                        default="Salesforce/blip2-opt-2.7b")
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    args = parser.parse_args()

    processor = CaptioningProcessor(
        model_name=args.base_model,
        processor_name=args.checkpoint,
    )

    dtype = torch.float16
    base_model = AutoModelForVision2Seq.from_pretrained(
        args.base_model, torch_dtype=dtype,
    )
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    caption = caption_image(
        model, processor, args.image, args.num_beams, args.max_new_tokens,
    )
    print(caption)


if __name__ == "__main__":
    main()
