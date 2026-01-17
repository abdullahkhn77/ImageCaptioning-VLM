# Vision-Language Model (VLM) Fine-Tuning for Custom Image Captioning with MLOps

This project focuses on fine-tuning a pre-trained VLM (BLIP) for domain-specific image captioning tasks, with full MLOps integration for production deployment.

## Project Overview

### Objectives
- **Fine-Tuning Goal**: Adapt a VLM to generate accurate, context-aware captions for images in a niche domain (e.g., fashion items)
- **MLOps Goal**: Implement version control, automated CI/CD pipelines, monitoring, and scalable inference
- **Tech Stack**: Hugging Face Transformers, MLflow/W&B, Docker/Kubernetes, Prometheus/Grafana

## Project Structure

```
VLM/
├── data/                    # Dataset storage and preprocessing
│   ├── raw/                 # Raw datasets
│   ├── processed/           # Preprocessed data
│   └── validation/          # Data validation scripts
├── models/                  # Model artifacts and checkpoints
│   ├── checkpoints/         # Training checkpoints
│   └── deployed/            # Production models
├── src/                     # Source code
│   ├── data_preparation/    # Data preprocessing and validation
│   ├── training/            # Fine-tuning scripts
│   ├── inference/           # Inference and serving code
│   └── monitoring/          # Monitoring and drift detection
├── deployment/              # Deployment configurations
│   ├── docker/              # Dockerfiles
│   ├── kubernetes/          # K8s manifests
│   └── api/                 # FastAPI application
├── mlops/                   # MLOps configurations
│   ├── dvc/                 # Data version control
│   ├── mlflow/              # MLflow tracking config
│   └── ci_cd/               # CI/CD pipelines
├── monitoring/              # Monitoring dashboards and configs
│   ├── prometheus/          # Prometheus configs
│   └── grafana/             # Grafana dashboards
├── tests/                   # Unit and integration tests
├── notebooks/               # Jupyter notebooks for exploration
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Steps

### Step 1: Data Preparation
- Dataset: Flickr30k/COCO + DeepFashion (10k-50k image-caption pairs)
- Preprocessing: Image resizing (224x224), tokenization
- Versioning: DVC for data version control
- Storage: Cloud bucket (S3) with Git versioning

### Step 2: Model Fine-Tuning
- Model: BLIP (Salesforce/blip-image-captioning-base)
- Framework: PyTorch with Hugging Face Transformers
- Hyperparameters: LR=1e-5, batch_size=16, epochs=5-10, FP16
- Metrics: BLEU, ROUGE, CIDEr
- Tracking: MLflow for experiment tracking

### Step 3: Model Deployment
- API: FastAPI for serving predictions
- Containerization: Docker
- Orchestration: Kubernetes with Helm
- CI/CD: GitHub Actions/Jenkins

### Step 4: Monitoring and Maintenance
- Metrics: Prometheus + Grafana
- Drift Detection: Alibi Detect
- Retraining: Apache Airflow automation
- Logging: ELK Stack

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Initialize DVC**
   ```bash
   dvc init
   dvc remote add -d storage s3://your-bucket/data
   ```

3. **Initialize MLflow**
   ```bash
   mlflow ui --backend-store-uri ./mlruns
   ```

4. **Run Data Preparation**
   ```bash
   python src/data_preparation/preprocess.py
   ```

5. **Train Model**
   ```bash
   python src/training/train.py --config configs/training_config.yaml
   ```

6. **Deploy API**
   ```bash
   docker build -t vlm-captioning -f deployment/docker/Dockerfile .
   docker run -p 8000:8000 vlm-captioning
   ```

## Configuration

- Training config: `configs/training_config.yaml`
- Model config: `configs/model_config.yaml`
- Deployment config: `configs/deployment_config.yaml`

## Monitoring

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- MLflow UI: `http://localhost:5000`

## License

MIT License

