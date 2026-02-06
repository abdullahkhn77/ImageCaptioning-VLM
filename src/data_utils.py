"""Utilities for converting standard caption datasets to the expected JSON format."""

import json
from pathlib import Path


def convert_coco_to_json(coco_ann_path: str, output_path: str):
    """Convert COCO Captions annotation file to the simplified JSON format.

    Args:
        coco_ann_path: Path to the COCO annotation file
            (e.g., captions_train2017.json).
        output_path: Path to write the converted JSON file.
    """
    with open(coco_ann_path) as f:
        coco = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

    entries = []
    for ann in coco["annotations"]:
        entries.append({
            "image": id_to_filename[ann["image_id"]],
            "caption": ann["caption"],
        })

    with open(output_path, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"Wrote {len(entries)} entries to {output_path}")


def convert_flickr30k_to_json(token_file: str, output_path: str):
    """Convert Flickr30k token file to the simplified JSON format.

    Args:
        token_file: Path to Flickr30k results_20130124.token file.
        output_path: Path to write the converted JSON file.
    """
    entries = []
    with open(token_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            filename = parts[0].split("#")[0]
            caption = parts[1]
            entries.append({"image": filename, "caption": caption})

    with open(output_path, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"Wrote {len(entries)} entries to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert caption datasets")
    parser.add_argument("--format", choices=["coco", "flickr30k"], required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    if args.format == "coco":
        convert_coco_to_json(args.input, args.output)
    elif args.format == "flickr30k":
        convert_flickr30k_to_json(args.input, args.output)
