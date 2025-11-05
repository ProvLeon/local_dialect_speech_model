#!/usr/bin/env python3
"""
Simple FastAPI server for model inference.
"""

import os
import uvicorn
import logging
import time
import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Version info to verify correct file is being used
SERVE_VERSION = "2.0.0-fixed"
logger.info(f"🚀 Loading serve.py version {SERVE_VERSION} (self-contained)")

# Import the inference module
try:
    from .inference import ModelInference

    logger.info("Successfully imported ModelInference (relative import)")
except ImportError:
    try:
        from inference import ModelInference

        logger.info("Successfully imported ModelInference (direct import)")
    except ImportError as e:
        logger.error(f"Failed to import ModelInference: {e}")
        logger.error(
            "This means the inference.py file is not found or has import errors"
        )
        raise

app = FastAPI(title="Speech Model API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for production
    allow_credentials=False,  # Must be False when allow_origins=["*"]
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# CORS headers for all responses
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Max-Age": "86400",
}

# Load model
try:
    model_path = Path(__file__).parent.parent
    logger.info(f"Loading model from: {model_path}")
    model = ModelInference(str(model_path))
    logger.info("Model loaded successfully")
    logger.info(f"✅ Using serve.py version {SERVE_VERSION}")
    logger.info(f"Model info: {model.get_model_info()}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    logger.error(f"Model path attempted: {model_path}")
    logger.error(
        f"Available files: {list(model_path.iterdir()) if model_path.exists() else 'Path does not exist'}"
    )
    raise


@app.get("/")
async def root():
    try:
        model_info = model.get_model_info()
        return JSONResponse(
            content={
                "message": "Twi Speech Model API",
                "status": "running",
                "model_info": model_info,
            },
            headers=CORS_HEADERS,
        )
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return JSONResponse(
            content={
                "message": "Twi Speech Model API",
                "status": "running",
                "error": str(e),
            },
            headers=CORS_HEADERS,
        )


@app.options("/{path:path}")
async def options_handler(path: str):
    """Handle CORS preflight requests."""
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "86400",
        },
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        health_status = model.health_check()
        return JSONResponse(
            content={
                "status": "healthy",
                "message": "Twi Speech Model API is running",
                "model_health": health_status,
            },
            headers=CORS_HEADERS,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                "status": "degraded",
                "message": "API is running but model health check failed",
                "error": str(e),
            },
            headers=CORS_HEADERS,
        )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict intent from uploaded audio file."""
    start_time = time.time()
    tmp_path = None
    try:
        logger.info(
            f"Received file: {file.filename}, content_type: {file.content_type}"
        )

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        logger.info(f"Saved file to: {tmp_path}")

        # Make prediction with timeout
        try:
            intent, confidence, top_predictions = await asyncio.wait_for(
                asyncio.to_thread(model.predict_topk, tmp_path, 5, 60), timeout=90.0
            )
            logger.info(f"Prediction: {intent}, confidence: {confidence}")
        except asyncio.TimeoutError:
            logger.error(f"Prediction timed out for file: {file.filename}")
            raise HTTPException(
                status_code=408,
                detail="Prediction timed out. Please try a shorter audio file.",
            )
        except Exception as prediction_error:
            logger.error(f"Prediction error: {prediction_error}")
            # Handle specific model inference errors
            if "features" in str(prediction_error):
                raise HTTPException(
                    status_code=400,
                    detail="Failed to process audio features. Please check your audio file format.",
                )
            elif "CUDA" in str(prediction_error) or "device" in str(prediction_error):
                raise HTTPException(
                    status_code=500, detail="Model device error. Please try again."
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Model inference failed: {str(prediction_error)}",
                )

        processing_time_ms = (time.time() - start_time) * 1000

        return JSONResponse(
            content={
                "intent": intent,
                "confidence": float(confidence),
                "top_predictions": top_predictions,
                "filename": file.filename,
                "model_type": "intent_only",
                "processing_time_ms": round(processing_time_ms, 2),
            },
            headers=CORS_HEADERS,
        )

    except asyncio.TimeoutError:
        logger.error(f"Request timed out for file: {file.filename}")
        raise HTTPException(
            status_code=408,
            detail="Request timed out. Please try a shorter audio file.",
        )
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        if "timed out" in str(e).lower() or "timeout" in str(e).lower():
            raise HTTPException(
                status_code=408,
                detail="Processing timed out. Please try a shorter audio file.",
            )
        elif "features" in str(e).lower():
            raise HTTPException(
                status_code=400,
                detail="Audio processing failed. Please check your audio file format and try again.",
            )
        elif "not defined" in str(e).lower():
            raise HTTPException(
                status_code=500,
                detail="Internal processing error. The service is experiencing issues.",
            )
        else:
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # Clean up
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.debug(f"Cleaned up temp file: {tmp_path}")
            except Exception as cleanup_error:
                logger.warning(
                    f"Failed to cleanup temp file {tmp_path}: {cleanup_error}"
                )


@app.post("/test-intent")
async def test_intent(file: UploadFile = File(...), top_k: int = 5):
    """Test intent from uploaded audio file (frontend-compatible endpoint)."""
    start_time = time.time()
    tmp_path = None
    try:
        logger.info(
            f"Received file for test-intent: {file.filename}, content_type: {file.content_type}, top_k: {top_k}"
        )

        # Read file content first
        content = await file.read()

        # Detect WebM content by content type or magic bytes
        is_webm = (file.content_type and "webm" in file.content_type.lower()) or (
            content.startswith(b"\x1a\x45\xdf\xa3")  # WebM magic bytes
        )

        # Choose appropriate file extension
        suffix = ".webm" if is_webm else ".wav"

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        logger.info(
            f"Saved file to: {tmp_path} (detected format: {'WebM' if is_webm else 'WAV'})"
        )

        # Make prediction with timeout (longer for WebM files)
        timeout_duration = 120.0 if is_webm else 90.0
        try:
            intent, confidence, top_predictions = await asyncio.wait_for(
                asyncio.to_thread(
                    model.predict_topk, tmp_path, top_k, int(timeout_duration - 30)
                ),
                timeout=timeout_duration,
            )
            logger.info(f"Prediction: {intent}, confidence: {confidence}")
        except asyncio.TimeoutError:
            logger.error(
                f"Prediction timed out for file: {file.filename} (format: {'WebM' if is_webm else 'WAV'})"
            )
            raise HTTPException(
                status_code=408,
                detail=f"Prediction timed out. {'WebM files may take longer to process.' if is_webm else 'Please try a shorter audio file.'}",
            )
        except Exception as prediction_error:
            logger.error(f"Prediction error for {file.filename}: {prediction_error}")
            # Handle specific model inference errors
            if "librosa" in str(prediction_error).lower() and is_webm:
                raise HTTPException(
                    status_code=400,
                    detail="WebM audio processing failed. Please try converting to WAV format first.",
                )
            elif "features" in str(prediction_error):
                raise HTTPException(
                    status_code=400,
                    detail="Failed to process audio features. Please check your audio file format.",
                )
            elif "CUDA" in str(prediction_error) or "device" in str(prediction_error):
                raise HTTPException(
                    status_code=500, detail="Model device error. Please try again."
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Model inference failed: {str(prediction_error)}",
                )

        processing_time_ms = (time.time() - start_time) * 1000

        return JSONResponse(
            content={
                "filename": file.filename,
                "intent": intent,
                "confidence": float(confidence),
                "top_predictions": top_predictions,
                "model_type": "intent_only",
                "processing_time_ms": round(processing_time_ms, 2),
                "top_k": top_k,
            },
            headers=CORS_HEADERS,
        )

    except asyncio.TimeoutError:
        logger.error(f"Request timed out for file: {file.filename}")
        raise HTTPException(
            status_code=408,
            detail="Request timed out. Please try a shorter audio file.",
        )
    except Exception as e:
        logger.error(f"Error in test-intent endpoint: {e}")
        if "timed out" in str(e).lower() or "timeout" in str(e).lower():
            raise HTTPException(
                status_code=408,
                detail="Processing timed out. Please try a shorter audio file.",
            )
        elif "features" in str(e).lower():
            raise HTTPException(
                status_code=400,
                detail="Audio processing failed. Please check your audio file format and try again.",
            )
        elif "not defined" in str(e).lower():
            raise HTTPException(
                status_code=500,
                detail="Internal processing error. The service is experiencing issues.",
            )
        else:
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # Clean up
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.debug(f"Cleaned up temp file: {tmp_path}")
            except Exception as cleanup_error:
                logger.warning(
                    f"Failed to cleanup temp file {tmp_path}: {cleanup_error}"
                )


@app.get("/model-info")
async def model_info():
    """Get model information."""
    try:
        info = model.get_model_info()
        logger.info(f"Model info requested: {info}")
        return JSONResponse(content=info, headers=CORS_HEADERS)
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting Speech Model API server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
