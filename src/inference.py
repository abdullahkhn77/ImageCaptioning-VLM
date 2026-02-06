"""Run inference with a fine-tuned VLM to caption a single image."""

import argparse

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel


def caption_image(model, processor, image_path: str, num_beams: int = 5,
                  max_new_tokens: int = 50) -> str:
    device = next(model.parameters()).device
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )

    return processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to fine-tuned checkpoint")
    parser.add_argument("--base-model", type=str,
                        default="Salesforce/blip2-opt-2.7b")
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    base_model = AutoModelForVision2Seq.from_pretrained(
        args.base_model, torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    caption = caption_image(model, processor, args.image,
                            args.num_beams, args.max_new_tokens)
    print(caption)


if __name__ == "__main__":
    main()
