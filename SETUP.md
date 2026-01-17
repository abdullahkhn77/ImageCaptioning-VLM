# Setup Guide

This guide will help you set up the VLM fine-tuning project from scratch.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for training)
- Docker (for deployment)
- Git
- Access to cloud storage (S3, GCS, etc.) for data versioning

## Step-by-Step Setup

### 1. Clone and Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Initialize DVC (Data Version Control)

```bash
# Initialize DVC
dvc init

# Configure remote storage (example for S3)
dvc remote add -d storage s3://your-bucket/vlm-data

# Or use local storage
dvc remote add -d storage /path/to/local/storage
```

### 3. Initialize MLflow

```bash
# Start MLflow tracking server
mlflow ui --backend-store-uri ./mlruns --port 5000
```

### 4. Prepare Data

```bash
# Download your dataset to data/raw/
# Format: JSON or CSV with 'image_path' and 'caption' columns

# Run preprocessing
python src/data_preparation/preprocess.py \
    --data_path data/raw/dataset.json \
    --image_dir data/raw/images \
    --output_dir data/processed
```

### 5. Train Model

```bash
# Train with default config
python src/training/train.py --config configs/training_config.yaml

# Or with custom parameters
python src/training/train.py \
    --config configs/training_config.yaml \
    --learning_rate 2e-5 \
    --batch_size 32
```

### 6. Deploy API

#### Using Docker:

```bash
# Build image
docker build -t vlm-captioning -f deployment/docker/Dockerfile .

# Run container
docker run -p 8000:8000 \
    -v $(pwd)/models/deployed:/app/models/deployed \
    vlm-captioning
```

#### Using Python directly:

```bash
cd deployment/api
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 7. Setup Monitoring

#### Prometheus:

```bash
# Download Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
tar xvfz prometheus-*.tar.gz
cd prometheus-*

# Start Prometheus
./prometheus --config.file=../monitoring/prometheus/prometheus.yml
```

#### Grafana:

```bash
# Install Grafana (example for Ubuntu)
sudo apt-get install -y software-properties-common
sudo add-apt-repository "deb https://packages.grafana.com/oss/deb stable main"
sudo apt-get update
sudo apt-get install grafana

# Start Grafana
sudo systemctl start grafana-server
```

Access Grafana at `http://localhost:3000` (default credentials: admin/admin)

## Testing

```bash
# Run tests
pytest tests/ -v

# Test API endpoint
curl -X POST "http://localhost:8000/caption" \
    -H "accept: application/json" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@path/to/image.jpg"
```

## Troubleshooting

### GPU Issues
- Ensure CUDA is properly installed: `nvidia-smi`
- Install PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### DVC Issues
- If DVC push fails, check your remote storage credentials
- Use `dvc status` to check data pipeline status

### MLflow Issues
- Ensure MLflow UI is running before training
- Check `mlruns/` directory for experiment logs

## Next Steps

1. Customize training configuration in `configs/training_config.yaml`
2. Add your dataset to `data/raw/`
3. Set up CI/CD pipeline (see `mlops/ci_cd/github_actions.yml`)
4. Configure Kubernetes deployment (see `deployment/kubernetes/`)

