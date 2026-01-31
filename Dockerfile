# vlm-caption-finetune-mlops — training / inference image
# Build: docker build -t vlm-caption-finetune-mlops .
# Run (GPU): docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs vlm-caption-finetune-mlops

FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir -r requirements.txt && pip install -e .

COPY configs ./configs
COPY scripts ./scripts
COPY app ./app

# Default: run training (override with CMD)
CMD ["python", "scripts/train.py", "--config-name", "default"]
