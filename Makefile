# vlm-caption-finetune-mlops — common commands
# Usage: make <target>

PYTHON := python
PIP := pip

.PHONY: install install-dev train evaluate export infer prepare-data lint test clean docker-build docker-run

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

prepare-data:
	$(PYTHON) scripts/prepare_data.py --config-name default

train:
	$(PYTHON) scripts/train.py --config-name default

train-accelerate:
	accelerate launch scripts/train.py --config-name default

evaluate:
	$(PYTHON) scripts/evaluate.py --config-name default

export:
	$(PYTHON) scripts/export_model.py --config-name default

infer:
	$(PYTHON) scripts/run_inference.py --config-name default

lint:
	ruff check src scripts app tests
	black --check src scripts app tests

format:
	black src scripts app tests
	ruff check --fix src scripts app tests

test:
	pytest tests -v

clean:
	rm -rf build dist *.egg-info .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

docker-build:
	docker build -t vlm-caption-finetune-mlops:latest .

docker-run:
	docker run --gpus all -v $$(pwd)/data:/app/data -v $$(pwd)/outputs:/app/outputs vlm-caption-finetune-mlops:latest

api:
	uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

gradio:
	$(PYTHON) app/gradio_demo.py
