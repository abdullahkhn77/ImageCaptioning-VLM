# VLM Captioning

Fine-tune Vision-Language Models for image captioning using LoRA and the Hugging Face ecosystem.

## Supported Models

| Model | HF Identifier | Notes |
|-------|---------------|-------|
| BLIP-2 OPT-2.7B | `Salesforce/blip2-opt-2.7b` | Default. Good balance of quality and resource usage. |
| BLIP-2 Flan-T5-XL | `Salesforce/blip2-flan-t5-xl` | Encoder-decoder variant. |
| LLaVA-1.5 7B | `llava-hf/llava-1.5-7b-hf` | Strong general-purpose VLM. |
| LLaVA-1.5 13B | `llava-hf/llava-1.5-13b-hf` | Higher quality, requires more VRAM. |
| Florence-2 Base | `microsoft/Florence-2-base` | Lightweight, multi-task capable. |
| Florence-2 Large | `microsoft/Florence-2-large` | Larger Florence variant. |
| Qwen-VL Chat | `Qwen/Qwen-VL-Chat` | Multilingual support. |

## Project Structure

```
vlm-captioning/
  configs/          # YAML training configs
  data/             # Dataset annotations and images
  notebooks/        # Jupyter notebooks for exploration
  outputs/          # Checkpoints and logs (git-ignored)
  scripts/          # Shell scripts for train / eval / inference
  src/              # Python source code
    train.py        # Fine-tuning entry point
    evaluate.py     # Compute captioning metrics
    inference.py    # Caption a single image
    dataset.py      # Dataset class
    data_utils.py   # Convert COCO / Flickr30k to expected format
```

## Setup

```bash
# Clone and enter the project
git clone <repo-url> && cd vlm-captioning

# Create a virtual environment
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Format

Annotation files are JSON arrays of objects with `image` and `caption` fields:

```json
[
  {"image": "img_001.jpg", "caption": "A dog sitting on a park bench."},
  {"image": "img_002.jpg", "caption": "An aerial view of a city at sunset."}
]
```

Place images under `data/images/` and annotations as `data/train.json`, `data/val.json`, `data/test.json`. See `data/README.md` for full details.

To convert COCO Captions:

```bash
python src/data_utils.py --format coco \
    --input /path/to/captions_train2017.json \
    --output data/train.json
```

## Training

Edit `configs/default.yaml` to adjust hyperparameters, then:

```bash
bash scripts/train.sh                      # uses default config
bash scripts/train.sh configs/custom.yaml   # uses a custom config
```

Training uses `accelerate launch` under the hood, so multi-GPU and DeepSpeed are supported via your `accelerate` config.

## Evaluation

```bash
bash scripts/evaluate.sh configs/default.yaml outputs/final test
```

Reports BLEU-1 through BLEU-4, METEOR, ROUGE-L, CIDEr, and SPICE.

## Inference

```bash
bash scripts/inference.sh path/to/image.jpg outputs/final
```

Prints the generated caption to stdout.

## Roadmap

- [x] Project scaffolding and config system
- [x] Dataset loader with JSON annotation format
- [x] LoRA-based fine-tuning pipeline
- [x] Evaluation script with standard captioning metrics
- [x] Single-image inference script
- [ ] Add data augmentation options (random crop, color jitter)
- [ ] Support multi-GPU training with FSDP
- [ ] Add quantized inference (4-bit / 8-bit via bitsandbytes)
- [ ] Integrate Gradio demo for interactive captioning
- [ ] Add unit tests for dataset loading and metric computation
- [ ] Support instruction-tuned prompting formats per model
