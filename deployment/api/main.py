"""
FastAPI application for serving VLM image captioning predictions.
"""

import os
import yaml
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import io
import logging
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('caption_requests_total', 'Total number of caption requests')
REQUEST_DURATION = Histogram('caption_request_duration_seconds', 'Caption request duration')
ERROR_COUNT = Counter('caption_errors_total', 'Total number of caption errors')

app = FastAPI(title="VLM Image Captioning API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and processor
model = None
processor = None
device = None


def load_model(model_path: str, config_path: Optional[str] = None):
    """Load model and processor."""
    global model, processor, device
    
    # Load config if provided
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_name = config.get('model_name', 'Salesforce/blip-image-captioning-base')
        device_name = config.get('deployment', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    else:
        model_name = 'Salesforce/blip-image-captioning-base'
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    device = torch.device(device_name)
    logger.info(f"Loading model from {model_path} on device {device}")
    
    # Load processor
    processor = BlipProcessor.from_pretrained(model_path if os.path.exists(model_path) else model_name)
    
    # Load model
    if os.path.exists(model_path):
        model = BlipForConditionalGeneration.from_pretrained(model_path)
    else:
        model = BlipForConditionalGeneration.from_pretrained(model_name)
    
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    model_path = os.getenv("MODEL_PATH", "models/deployed")
    config_path = os.getenv("CONFIG_PATH", "configs/model_config.yaml")
    load_model(model_path, config_path)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "VLM Image Captioning API"}


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else None
    }


@app.post("/caption")
async def caption_image(file: UploadFile = File(...)):
    """Generate caption for uploaded image."""
    REQUEST_COUNT.inc()
    
    try:
        with REQUEST_DURATION.time():
            # Read image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            
            # Process image
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            # Generate caption
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=4
                )
            
            # Decode caption
            caption = processor.decode(generated_ids[0], skip_special_tokens=True)
            
            logger.info(f"Generated caption: {caption}")
            
            return {
                "caption": caption,
                "status": "success"
            }
    
    except Exception as e:
        ERROR_COUNT.inc()
        logger.error(f"Error generating caption: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating caption: {str(e)}")


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

