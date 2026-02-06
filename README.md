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
  configs/              # YAML training configs
  data/                 # Dataset annotations and images
  notebooks/            # Jupyter notebooks for exploration
  outputs/              # Checkpoints and logs (git-ignored)
  scripts/              # Shell scripts for train / eval / inference / data prep
  src/                  # Python source code
    train.py            # Fine-tuning entry point
    evaluate.py         # Compute captioning metrics
    inference.py        # Caption a single image
    dataset.py          # CaptionDataset + collate function + transforms
    preprocessing.py    # CaptioningProcessor (wraps HF AutoProcessor)
    data_utils.py       # Download, convert, split, validate datasets
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

## Data Pipeline

### Quick start with COCO Captions

```bash
# Download COCO 2017, convert to JSONL, validate
bash scripts/prepare_data.sh

# Skip the image download (annotations only)
bash scripts/prepare_data.sh --skip-images
```

### Dataset format

Annotation files use **JSONL** (one JSON object per line):

```jsonl
{"image": "train2017/000000391895.jpg", "caption": "A man on a moped on a dirt road."}
{"image": "train2017/000000522418.jpg", "caption": "A woman cutting a cake."}
```

Multi-caption-per-image is supported — see `data/README.md` for full details.

### Converting other datasets

```bash
# COCO
python src/data_utils.py convert-coco \
    --annotation /path/to/captions_train2017.json \
    --image-dir /path/to/train2017 \
    --output data/train.jsonl

# Flickr30k
python src/data_utils.py convert-flickr \
    --token-file /path/to/results_20130124.token \
    --output data/flickr30k.jsonl
```

### Splitting a custom dataset

```bash
python src/data_utils.py split \
    --input data/my_dataset.jsonl \
    --output-dir data/ \
    --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

All captions for the same image are kept in the same split to avoid leakage.

### Validating

```bash
python src/data_utils.py validate \
    --annotations data/train.jsonl \
    --image-root data/images/
```

## Training

Edit `configs/default.yaml` to adjust hyperparameters, then:

```bash
bash scripts/train.sh                      # uses default config
bash scripts/train.sh configs/custom.yaml   # uses a custom config
```

Training uses `accelerate launch` under the hood, so multi-GPU and DeepSpeed are supported via your `accelerate` config.

The `CaptioningProcessor` class handles model-specific prompt formatting automatically (BLIP-2, LLaVA, Florence-2, Qwen-VL).

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

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_data_exploration.ipynb` | Dataset stats, caption distributions, vocabulary analysis, augmentation previews |
| `exploration.ipynb` | Model inference playground |

## Roadmap

- [x] Project scaffolding and config system
- [x] Robust data pipeline (JSONL loader, COCO/Flickr converters, validation)
- [x] Model-agnostic preprocessing (CaptioningProcessor)
- [x] Configurable augmentation (crop, flip, color jitter)
- [x] LoRA-based fine-tuning pipeline
- [x] Evaluation script with standard captioning metrics
- [x] Single-image inference script
- [x] Data preparation shell script
- [ ] Support multi-GPU training with FSDP
- [ ] Add quantized inference (4-bit / 8-bit via bitsandbytes)
- [ ] Integrate Gradio demo for interactive captioning
- [ ] Add unit tests for dataset loading and metric computation
- [ ] Support instruction-tuned prompting formats per model
