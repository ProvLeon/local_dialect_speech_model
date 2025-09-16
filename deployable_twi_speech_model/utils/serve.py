#!/usr/bin/env python3
"""
Simple FastAPI server for model inference.
"""

import os
import uvicorn
import logging
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the inference module
try:
    from inference import ModelInference
    logger.info("Successfully imported ModelInference")
except ImportError as e:
    logger.error(f"Failed to import ModelInference: {e}")
    raise

app = FastAPI(title="Speech Model API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
try:
    model_path = Path(__file__).parent.parent
    logger.info(f"Loading model from: {model_path}")
    model = ModelInference(model_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

@app.get("/")
async def root():
    return {"message": "Speech Model API", "model_info": model.get_model_info()}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Speech Model API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict intent from uploaded audio file."""
    start_time = time.time()
    try:
        logger.info(f"Received file: {file.filename}, content_type: {file.content_type}")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        logger.info(f"Saved file to: {tmp_path}")

        # Make prediction
        intent, confidence, top_predictions = model.predict_topk(tmp_path, top_k=5)
        logger.info(f"Prediction: {intent}, confidence: {confidence}")

        # Clean up
        os.unlink(tmp_path)

        processing_time_ms = (time.time() - start_time) * 1000

        return {
            "intent": intent,
            "confidence": float(confidence),
            "top_predictions": top_predictions,
            "filename": file.filename,
            "model_type": "intent_only",
            "processing_time_ms": round(processing_time_ms, 2)
        }

    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test-intent")
async def test_intent(file: UploadFile = File(...), top_k: int = 5):
    """Test intent from uploaded audio file (frontend-compatible endpoint)."""
    start_time = time.time()
    try:
        logger.info(f"Received file for test-intent: {file.filename}, content_type: {file.content_type}, top_k: {top_k}")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        logger.info(f"Saved file to: {tmp_path}")

        # Make prediction
        intent, confidence, top_predictions = model.predict_topk(tmp_path, top_k=top_k)
        logger.info(f"Prediction: {intent}, confidence: {confidence}")

        # Clean up
        os.unlink(tmp_path)

        processing_time_ms = (time.time() - start_time) * 1000

        return {
            "intent": intent,
            "confidence": float(confidence),
            "top_predictions": top_predictions,
            "filename": file.filename,
            "model_type": "intent_only",
            "processing_time_ms": round(processing_time_ms, 2),
            "top_k": top_k
        }

    except Exception as e:
        logger.error(f"Error in test-intent endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def model_info():
    """Get model information."""
    try:
        info = model.get_model_info()
        logger.info(f"Model info requested: {info}")
        return info
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting Speech Model API server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
