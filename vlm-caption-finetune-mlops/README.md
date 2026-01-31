# vlm-caption-finetune-mlops

A professional, MLOps-oriented Python project for **fine-tuning Vision-Language Models (VLMs)** for improved image captioning. Supports PaliGemma, LLaVA, Qwen2-VL, Llama 3.2 Vision and similar open VLMs using Hugging Face Transformers, PEFT (LoRA/QLoRA), TRL, and Accelerate.

## Project goals

- **Fine-tune open VLMs** (PaliGemma, LLaVA, Qwen2-VL, Llama 3.2 Vision) for image captioning with LoRA/QLoRA.
- **Standard datasets**: COCO, Flickr8k/30k, or custom image + caption data.
- **MLOps**: experiment tracking (MLflow / W&B), data/model versioning (DVC), config management (Hydra/YAML), reproducible envs, Docker, CI/CD hints, and model serving (FastAPI / Gradio).

## Directory structure

See [STRUCTURE.md](STRUCTURE.md) for the full tree. Summary:

- **app/** — FastAPI and Gradio serving
- **configs/** — YAML configs (training, model, dataset, LoRA)
- **data/** — Raw and processed data (DVC-tracked)
- **notebooks/** — Exploratory notebooks
- **scripts/** — CLI: prepare_data, train, evaluate, export_model, run_inference
- **src/vlm_caption_finetune_mlops/** — data, models, evaluation, inference
- **tests/** — Pytest
- **.github/workflows/** — CI (lint, test)
- **Dockerfile**, **docker-compose.yml**, **dvc.yaml**, **Makefile**

## Setup

### 1. Clone and create environment

```bash
git clone <repo-url>
cd vlm-caption-finetune-mlops
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
# Or: pip install -r requirements.txt
```

### 2. (Optional) Pull data with DVC

```bash
dvc pull   # if data is tracked with DVC
```

### 3. Configure

Edit `configs/default/` YAML files for model, dataset, LoRA, and training hyperparameters. Override via Hydra or env vars as needed.

## How to fine-tune

1. **Prepare data**: Place images and a caption file (e.g. CSV/JSON with `image_path`, `caption`) in `data/raw/`, or use a built-in dataset (COCO, Flickr8k/30k).

   ```bash
   python scripts/prepare_data.py --config-name default
   ```

2. **Train** (single-GPU example):

   ```bash
   python scripts/train.py --config-name default
   ```

   Multi-GPU with Accelerate:

   ```bash
   accelerate config   # configure once
   accelerate launch scripts/train.py --config-name default
   ```

   Training uses mixed precision (fp16/bf16), gradient checkpointing, and logs to MLflow or W&B when configured.

3. **Evaluate**:

   ```bash
   python scripts/evaluate.py --config-name default checkpoint_dir=path/to/checkpoint
   ```

4. **Export / push to Hub**:

   ```bash
   python scripts/export_model.py --config-name default checkpoint_dir=path/to/checkpoint
   ```

5. **Run inference** (API or Gradio):

   ```bash
   uvicorn app.api:app --reload
   # Or: python app/gradio_demo.py
   ```

## MLOps workflow overview

| Step        | Tool / Location |
|------------|------------------|
| Config     | `configs/` (YAML), Hydra CLI overrides |
| Data       | DVC (`dvc.yaml`), `data/raw`, `data/processed` |
| Training   | `scripts/train.py`, Accelerate, PEFT LoRA/QLoRA |
| Tracking   | MLflow or W&B (set in config) |
| Evaluation | `scripts/evaluate.py`, BLEU/METEOR/ROUGE/CLIPScore |
| Export     | `scripts/export_model.py` → Hugging Face Hub |
| Serving    | `app/api.py` (FastAPI), `app/gradio_demo.py` |
| CI/CD      | `.github/workflows/ci.yml` (lint, test) |
| Containers | `Dockerfile`, `docker-compose.yml` |

## GPU / multi-GPU

- **Single GPU**: `python scripts/train.py` (uses CUDA if available).
- **Multi-GPU**: `accelerate launch scripts/train.py` (config via `accelerate config`).
- **Mixed precision**: Set in config (`fp16`/`bf16`).
- **Gradient checkpointing**: Enabled in config to reduce memory.

## License

MIT.
