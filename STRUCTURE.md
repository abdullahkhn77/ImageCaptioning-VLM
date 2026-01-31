# Project structure

Full directory tree for **vlm-caption-finetune-mlops**:

```text
vlm-caption-finetune-mlops/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI: lint (ruff, black), pytest
├── app/
│   ├── __init__.py
│   ├── api.py                  # FastAPI: POST /caption, health
│   └── gradio_demo.py          # Gradio UI for image captioning
├── configs/
│   ├── default/
│   │   ├── config.yaml         # Main: training, paths, experiment
│   │   ├── model.yaml          # VLM name, processor, dtype
│   │   ├── dataset.yaml        # Dataset: coco/flickr/custom
│   │   └── lora.yaml           # LoRA/QLoRA settings
│   └── experiment/             # Override configs for experiments
├── data/
│   ├── raw/                    # Raw data (DVC-tracked or local)
│   │   └── .gitkeep
│   └── processed/              # Prepared datasets
│       └── .gitkeep
├── notebooks/
│   └── 01_explore_data.ipynb   # Explore dataset
├── scripts/
│   ├── prepare_data.py         # Load dataset, save processed
│   ├── train.py                # Train VLM (LoRA), MLflow/W&B
│   ├── evaluate.py             # BLEU, METEOR, ROUGE
│   ├── export_model.py         # Export / push to HF Hub
│   └── run_inference.py        # Single-image caption
├── src/
│   ├── __init__.py
│   └── vlm_caption_finetune_mlops/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loaders.py      # get_captioning_dataset (COCO/Flickr/custom)
│       │   ├── preprocessing.py # preprocess_image
│       │   └── tokenization.py # tokenize_captions, get_processor
│       ├── models/
│       │   ├── __init__.py
│       │   ├── loaders.py      # load_vlm_and_processor
│       │   ├── peft_adapters.py # setup_peft_model (LoRA/QLoRA)
│       │   └── training.py     # run_training (skeleton)
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── metrics.py     # BLEU, METEOR, ROUGE, compute_metrics
│       └── inference/
│           ├── __init__.py
│           └── predictor.py   # CaptionPredictor, load_predictor
├── tests/
│   ├── __init__.py
│   ├── test_data.py            # preprocess_image, tokenize_captions
│   └── test_metrics.py         # BLEU, ROUGE, compute_metrics
├── .gitignore
├── docker-compose.yml          # train + api services
├── Dockerfile                  # Training/inference image
├── dvc.yaml                    # DVC pipeline: prepare_data
├── Makefile                    # install, train, evaluate, api, gradio
├── pyproject.toml              # Package deps, black/ruff/mypy
├── README.md
├── requirements.txt
└── STRUCTURE.md                # This file
```

## Quick commands

| Action        | Command |
|---------------|---------|
| Install       | `pip install -e ".[dev]"` |
| Prepare data  | `make prepare-data` or `python scripts/prepare_data.py --config-name default` |
| Train         | `make train` or `accelerate launch scripts/train.py --config-name default` |
| Evaluate      | `make evaluate` (set `checkpoint_dir` in config or override) |
| Export        | `make export` |
| API           | `make api` or `uvicorn app.api:app --reload --port 8000` |
| Gradio        | `make gradio` or `python app/gradio_demo.py` |
| Lint          | `make lint` |
| Test          | `make test` |
