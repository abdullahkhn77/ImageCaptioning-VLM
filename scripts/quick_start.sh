#!/bin/bash

# Quick Start Script for VLM Fine-Tuning Project

set -e

echo "🚀 VLM Fine-Tuning Project - Quick Start"
echo "========================================"

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Initialize DVC
if [ ! -d ".dvc" ]; then
    echo "📊 Initializing DVC..."
    dvc init
    echo "⚠️  Don't forget to configure DVC remote: dvc remote add -d storage <your-storage>"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/raw data/processed models/checkpoints models/deployed mlruns logs

# Check for GPU
echo "🎮 Checking for GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name --format=csv,noheader
    echo "✅ GPU detected"
else
    echo "⚠️  No GPU detected - training will use CPU (slower)"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Add your dataset to data/raw/"
echo "2. Run preprocessing: python src/data_preparation/preprocess.py --data_path data/raw/dataset.json --image_dir data/raw/images"
echo "3. Start training: python src/training/train.py --config configs/training_config.yaml"
echo "4. Deploy API: docker-compose up"
echo ""
echo "For detailed instructions, see SETUP.md"

