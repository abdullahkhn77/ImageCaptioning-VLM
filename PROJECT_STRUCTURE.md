# Project Structure Overview

This document provides a detailed overview of the VLM fine-tuning project structure.

## Directory Structure

```
VLM/
├── configs/                      # Configuration files
│   ├── training_config.yaml      # Training hyperparameters and settings
│   └── model_config.yaml         # Model and inference configuration
│
├── data/                         # Data directory
│   ├── raw/                      # Raw datasets (not versioned)
│   ├── processed/                # Preprocessed data (DVC versioned)
│   └── validation/               # Data validation scripts
│
├── models/                       # Model artifacts
│   ├── checkpoints/              # Training checkpoints
│   └── deployed/                 # Production-ready models
│
├── src/                          # Source code
│   ├── data_preparation/
│   │   └── preprocess.py         # Data preprocessing and validation
│   ├── training/
│   │   ├── train.py              # Main training script with MLflow
│   │   └── evaluate.py           # Evaluation metrics (BLEU, ROUGE, CIDEr)
│   ├── monitoring/
│   │   └── drift_detection.py    # Data and performance drift detection
│   └── inference/                # (Future) Inference utilities
│
├── deployment/                   # Deployment configurations
│   ├── api/
│   │   └── main.py               # FastAPI application
│   ├── docker/
│   │   └── Dockerfile            # Docker container configuration
│   └── kubernetes/               # (Future) K8s manifests
│
├── mlops/                        # MLOps configurations
│   ├── dvc/                      # DVC configuration
│   ├── mlflow/                   # MLflow tracking config
│   └── ci_cd/
│       └── github_actions.yml     # CI/CD pipeline
│
├── monitoring/                   # Monitoring configurations
│   ├── prometheus/
│   │   └── prometheus.yml        # Prometheus scraping config
│   └── grafana/
│       └── dashboard.json        # Grafana dashboard
│
├── tests/                        # Test suite
│   └── test_api.py               # API endpoint tests
│
├── notebooks/                    # Jupyter notebooks
│   └── exploration.ipynb         # Data exploration and experimentation
│
├── requirements.txt              # Python dependencies
├── docker-compose.yml            # Docker Compose for full stack
├── dvc.yaml                      # DVC pipeline definition
├── README.md                     # Main project documentation
├── SETUP.md                      # Detailed setup instructions
└── PROJECT_STRUCTURE.md          # This file
```

## Key Components

### 1. Data Preparation (`src/data_preparation/`)
- **preprocess.py**: Handles dataset loading, validation, splitting, and preprocessing
- Features: Image resizing, caption tokenization, data validation, train/val/test splitting
- Integrates with DVC for data versioning

### 2. Model Training (`src/training/`)
- **train.py**: Fine-tuning script with:
  - BLIP model loading and fine-tuning
  - MLflow experiment tracking
  - Checkpoint saving
  - Mixed precision training (FP16)
- **evaluate.py**: Model evaluation with:
  - BLEU score computation
  - ROUGE score computation
  - CIDEr score computation

### 3. Deployment (`deployment/`)
- **api/main.py**: FastAPI application with:
  - Image captioning endpoint
  - Health check endpoints
  - Prometheus metrics exposure
  - Error handling and logging
- **docker/Dockerfile**: Containerized deployment

### 4. Monitoring (`src/monitoring/`)
- **drift_detection.py**: 
  - Data drift detection using Alibi Detect
  - Performance drift detection
  - Feature extraction for drift analysis

### 5. MLOps Integration
- **DVC**: Data version control (see `dvc.yaml`)
- **MLflow**: Experiment tracking (configured in training script)
- **CI/CD**: GitHub Actions pipeline for automated testing and deployment

### 6. Monitoring Stack
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **Docker Compose**: Easy deployment of full stack

## Workflow

1. **Data Preparation**
   ```bash
   python src/data_preparation/preprocess.py --data_path data/raw/dataset.json --image_dir data/raw/images
   ```

2. **Training**
   ```bash
   python src/training/train.py --config configs/training_config.yaml
   ```

3. **Evaluation**
   ```bash
   python src/training/evaluate.py --model_path models/checkpoints/... --test_data data/processed/test.json
   ```

4. **Deployment**
   ```bash
   docker-compose up
   # or
   docker build -t vlm-captioning -f deployment/docker/Dockerfile .
   ```

5. **Monitoring**
   - Access Prometheus: http://localhost:9090
   - Access Grafana: http://localhost:3000
   - Access API: http://localhost:8000

## Configuration Files

- **training_config.yaml**: Training hyperparameters, data paths, MLflow settings
- **model_config.yaml**: Model architecture, inference parameters, deployment settings

## Next Steps

1. Add your dataset to `data/raw/`
2. Configure DVC remote storage
3. Customize training hyperparameters in `configs/training_config.yaml`
4. Set up CI/CD secrets and deployment targets
5. Configure monitoring alerts in Grafana

