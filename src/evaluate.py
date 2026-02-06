"""Evaluate a fine-tuned VLM on captioning metrics."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


METRIC_CLASSES = {
    "bleu": Bleu(4),
    "meteor": Meteor(),
    "rouge_l": Rouge(),
    "cider": Cider(),
}


def generate_captions(model, processor, annotations, image_root, cfg):
    device = next(model.parameters()).device
    generated = {}
    references = {}

    for idx, entry in enumerate(tqdm(annotations, desc="Generating captions")):
        image = Image.open(Path(image_root) / entry["image"]).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                num_beams=cfg["evaluation"]["num_beams"],
                max_new_tokens=cfg["evaluation"]["max_new_tokens"],
            )

        caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        generated[idx] = [caption]
        references.setdefault(idx, []).append(entry["caption"])

    return generated, references


def compute_metrics(generated, references, metric_names):
    results = {}
    for name in metric_names:
        scorer = METRIC_CLASSES.get(name)
        if scorer is None:
            continue
        score, _ = scorer.compute_score(references, generated)
        if name == "bleu":
            for i, s in enumerate(score, 1):
                results[f"BLEU-{i}"] = round(s, 4)
        else:
            results[name.upper()] = round(score, 4)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to fine-tuned model checkpoint")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    processor = AutoProcessor.from_pretrained(args.checkpoint)
    base_model = AutoModelForVision2Seq.from_pretrained(
        model_name, torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ann_path = cfg["data"][f"{args.split}_annotations"]
    with open(ann_path) as f:
        annotations = json.load(f)

    generated, references = generate_captions(
        model, processor, annotations, cfg["data"]["image_root"], cfg
    )
    metrics = compute_metrics(generated, references, cfg["evaluation"]["metrics"])

    print("\n--- Evaluation Results ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    output_path = Path(cfg["training"]["output_dir"]) / f"eval_{args.split}.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
