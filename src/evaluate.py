"""Evaluate a fine-tuned VLM on captioning metrics."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from PIL import Image
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForVision2Seq

from preprocessing import CaptioningProcessor

logger = logging.getLogger(__name__)

# Lazy-import scorers so the module can still be imported if pycocoevalcap
# is not installed (e.g. for linting).
_METRIC_CLASSES: dict | None = None


def _get_metric_classes() -> dict:
    global _METRIC_CLASSES
    if _METRIC_CLASSES is None:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge

        _METRIC_CLASSES = {
            "bleu": Bleu(4),
            "meteor": Meteor(),
            "rouge_l": Rouge(),
            "cider": Cider(),
        }
    return _METRIC_CLASSES


def _load_annotations(path: str | Path) -> list[dict]:
    """Load annotations from JSON or JSONL."""
    path = Path(path)
    if path.suffix == ".jsonl":
        entries = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
    with open(path) as f:
        return json.load(f)


def generate_captions(
    model: torch.nn.Module,
    processor: CaptioningProcessor,
    annotations: list[dict],
    image_root: str | Path,
    num_beams: int = 5,
    max_new_tokens: int = 50,
) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    """Generate captions and collect references, keyed by image index."""
    device = next(model.parameters()).device
    image_root = Path(image_root)

    generated: dict[int, list[str]] = {}
    references: dict[int, list[str]] = {}

    for idx, entry in enumerate(tqdm(annotations, desc="Generating captions")):
        try:
            image = Image.open(image_root / entry["image"]).convert("RGB")
        except Exception as exc:
            logger.warning("Skipping %s: %s", entry["image"], exc)
            continue

        inputs = processor.process_for_inference(image).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
            )

        caption = processor.decode(output_ids)
        generated[idx] = [caption]
        references.setdefault(idx, []).append(entry["caption"])

    return generated, references


def compute_metrics(
    generated: dict[int, list[str]],
    references: dict[int, list[str]],
    metric_names: list[str],
) -> dict[str, float]:
    """Compute captioning metrics using pycocoevalcap."""
    scorers = _get_metric_classes()
    results: dict[str, float] = {}

    for name in metric_names:
        scorer = scorers.get(name)
        if scorer is None:
            logger.warning("Unknown metric '%s', skipping", name)
            continue
        score, _ = scorer.compute_score(references, generated)
        if name == "bleu":
            for i, s in enumerate(score, 1):
                results[f"BLEU-{i}"] = round(s, 4)
        else:
            results[name.upper()] = round(score, 4)

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned captioning model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to fine-tuned model checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Load model + processor ──────────────────────────────────────────
    processor = CaptioningProcessor(
        model_name=cfg["model"]["name"],
        processor_name=args.checkpoint,
        max_length=cfg["data"]["max_length"],
        image_size=cfg["data"]["image_size"],
    )

    dtype = torch.bfloat16 if cfg["hardware"]["bf16"] else torch.float16
    base_model = AutoModelForVision2Seq.from_pretrained(
        cfg["model"]["name"], torch_dtype=dtype,
    )
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ── Load annotations ────────────────────────────────────────────────
    ann_path = cfg["data"][f"{args.split}_annotations"]
    annotations = _load_annotations(ann_path)
    logger.info("Loaded %d entries from %s", len(annotations), ann_path)

    # ── Generate + score ────────────────────────────────────────────────
    generated, references = generate_captions(
        model, processor, annotations,
        cfg["data"]["image_root"],
        num_beams=cfg["evaluation"]["num_beams"],
        max_new_tokens=cfg["evaluation"]["max_new_tokens"],
    )
    metrics = compute_metrics(generated, references, cfg["evaluation"]["metrics"])

    print("\n--- Evaluation Results ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    output_path = Path(cfg["training"]["output_dir"]) / f"eval_{args.split}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
