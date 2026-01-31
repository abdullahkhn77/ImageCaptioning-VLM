"""
FastAPI inference endpoint for VLM image captioning.
Run: uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
"""

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(
    title="VLM Caption API",
    description="Generate image captions using a fine-tuned Vision-Language Model",
    version="0.1.0",
)

# Lazy-load predictor to avoid loading model at import time
_predictor = None
CHECKPOINT_ENV = "VLM_CHECKPOINT_PATH"


def get_predictor():
    """Load predictor from env VLM_CHECKPOINT_PATH or default path."""
    global _predictor
    if _predictor is None:
        ckpt = os.environ.get(CHECKPOINT_ENV, "outputs/checkpoints")
        if not Path(ckpt).exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model checkpoint not found. Set {CHECKPOINT_ENV} to a valid path.",
            )
        from vlm_caption_finetune_mlops.inference import load_predictor
        _predictor = load_predictor(ckpt)
    return _predictor


class CaptionResponse(BaseModel):
    caption: str


@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok"}


@app.post("/caption", response_model=CaptionResponse)
async def caption_image(file: UploadFile = File(...)):
    """
    Upload an image and receive a generated caption.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    contents = await file.read()
    from PIL import Image
    import io
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    predictor = get_predictor()
    caption = predictor.predict(image)
    return CaptionResponse(caption=caption)


@app.post("/caption/url")
async def caption_image_url(url: str):
    """Caption image from URL (optional)."""
    import urllib.request
    try:
        with urllib.request.urlopen(url) as resp:
            data = resp.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")
    from PIL import Image
    import io
    image = Image.open(io.BytesIO(data)).convert("RGB")
    predictor = get_predictor()
    caption = predictor.predict(image)
    return {"caption": caption}
