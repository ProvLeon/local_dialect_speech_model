#!/usr/bin/env python3
"""
FastAPI Server for Optimized Twi Speech Recognition Engine
=========================================================

This module provides a production-ready FastAPI server for the optimized
Twi speech recognition system. It handles:

1. Audio file uploads and processing
2. Real-time speech recognition
3. Intent classification
4. WebM/WAV audio format support
5. Streaming audio processing
6. Performance monitoring and health checks

Author: AI Assistant
Date: 2025-11-05
"""

import os
import asyncio
import tempfile
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
import soundfile as sf

# Import optimized components
from src.speech_recognizer import OptimizedSpeechRecognizer, create_speech_recognizer
from config.config import OptimizedConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global recognizer instance
recognizer: Optional[OptimizedSpeechRecognizer] = None
config = OptimizedConfig()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global recognizer

    # Startup
    logger.info("Starting Optimized Twi Speech Recognition Server...")

    try:
        # Initialize speech recognizer
        recognizer = create_speech_recognizer()

        # Validate configuration
        config.validate_config()

        # Print startup summary
        config.print_config_summary()

        logger.info("Server initialization completed successfully")

    except Exception as e:
        logger.error(f"Failed to initialize server: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down server...")


# Create FastAPI app
app = FastAPI(
    title="Optimized Twi Speech Recognition API",
    description="Production-ready speech recognition for Twi language using Whisper + Intent Classification",
    version=config.DEPLOYMENT["version"],
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.API["cors_origins"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS headers for all responses
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Max-Age": "86400",
}


# Pydantic models
class RecognitionRequest(BaseModel):
    """Request model for speech recognition."""

    language: str = Field(
        default=None, description="Language code (auto-detect if None)"
    )
    include_alternatives: bool = Field(
        default=True, description="Include alternative intents"
    )
    confidence_threshold: float = Field(
        default=0.5, description="Minimum confidence threshold"
    )


class RecognitionResponse(BaseModel):
    """Response model for speech recognition."""

    transcription: Dict[str, Any]
    intent: Dict[str, Any]
    audio_info: Dict[str, Any]
    processing_time: float
    timestamp: float
    status: str


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    components: Dict[str, str]
    device_info: Dict[str, Any]
    timestamp: float


class IntentInfo(BaseModel):
    """Intent information model."""

    intent: str
    description: str
    examples: List[str]
    priority: str


# Helper functions
def detect_audio_format(content: bytes, filename: str) -> str:
    """Detect audio format from content or filename."""
    # Check WebM magic bytes
    if content.startswith(b"\x1a\x45\xdf\xa3"):
        return "webm"

    # Check WAV magic bytes
    if content.startswith(b"RIFF") and b"WAVE" in content[:12]:
        return "wav"

    # Check filename extension
    if filename:
        ext = Path(filename).suffix.lower()
        if ext in [".webm", ".opus"]:
            return "webm"
        elif ext in [".wav", ".wave"]:
            return "wav"
        elif ext in [".mp3", ".mpeg"]:
            return "mp3"
        elif ext in [".m4a", ".aac"]:
            return "m4a"

    # Default to wav
    return "wav"


def convert_audio_if_needed(input_path: str, detected_format: str) -> str:
    """Convert audio to WAV format if needed."""
    if detected_format == "wav":
        return input_path

    try:
        # Create output path
        output_path = input_path.replace(Path(input_path).suffix, "_converted.wav")

        # Load and convert
        import librosa

        audio, sr = librosa.load(input_path, sr=16000, mono=True)
        sf.write(output_path, audio, sr)

        logger.info(f"Converted {detected_format} to WAV: {output_path}")
        return output_path

    except Exception as e:
        logger.warning(f"Audio conversion failed: {e}, using original file")
        return input_path


async def cleanup_temp_files(*file_paths: str):
    """Clean up temporary files."""
    for file_path in file_paths:
        try:
            if file_path and Path(file_path).exists():
                Path(file_path).unlink()
                logger.debug(f"Cleaned up: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup {file_path}: {e}")


# API Routes


@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information."""
    try:
        stats = recognizer.get_statistics() if recognizer else {}

        return JSONResponse(
            content={
                "message": "Optimized Twi Speech Recognition API",
                "version": config.DEPLOYMENT["version"],
                "status": "running",
                "whisper_model": config.WHISPER["model_size"],
                "supported_intents": len(config.INTENTS),
                "device": config.get_device(),
                "statistics": stats,
            },
            headers=CORS_HEADERS,
        )
    except Exception as e:
        logger.error(f"Root endpoint error: {e}")
        return JSONResponse(
            content={
                "message": "Optimized Twi Speech Recognition API",
                "status": "running",
                "error": str(e),
            },
            headers=CORS_HEADERS,
        )


@app.options("/{path:path}")
async def options_handler(path: str):
    """Handle CORS preflight requests."""
    return JSONResponse(content={"message": "OK"}, headers=CORS_HEADERS)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        health_status = (
            recognizer.health_check()
            if recognizer
            else {
                "status": "unhealthy",
                "components": {"recognizer": "not_initialized"},
                "device_info": {},
                "timestamp": time.time(),
            }
        )

        return JSONResponse(content=health_status, headers=CORS_HEADERS)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={"status": "unhealthy", "error": str(e), "timestamp": time.time()},
            headers=CORS_HEADERS,
        )


@app.get("/intents", response_model=List[IntentInfo])
async def get_supported_intents():
    """Get list of supported intents."""
    try:
        intents = recognizer.get_supported_intents() if recognizer else []

        return JSONResponse(content=intents, headers=CORS_HEADERS)
    except Exception as e:
        logger.error(f"Failed to get intents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/statistics")
async def get_statistics():
    """Get system performance statistics."""
    try:
        stats = recognizer.get_statistics() if recognizer else {}

        return JSONResponse(content=stats, headers=CORS_HEADERS)
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
async def get_model_info():
    """Get model information."""
    try:
        if recognizer:
            info = {
                "whisper_model": config.WHISPER["model_size"],
                "language": config.WHISPER["language"],
                "supported_intents": len(config.INTENTS),
                "device": config.get_device(),
                "version": config.DEPLOYMENT["version"],
                "model_type": "optimized_whisper_intent",
            }
        else:
            info = {"status": "not_initialized"}

        return JSONResponse(content=info, headers=CORS_HEADERS)
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recognize", response_model=RecognitionResponse)
async def recognize_speech(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: str = Query(
        default=None, description="Language code (auto-detect if None)"
    ),
    include_alternatives: bool = Query(
        default=True, description="Include alternative intents"
    ),
    confidence_threshold: float = Query(
        default=0.5, description="Minimum confidence threshold"
    ),
):
    """
    Recognize speech from uploaded audio file.

    Supports WAV, WebM, MP3, and M4A formats.
    Returns transcription and intent classification.
    """
    if not recognizer:
        raise HTTPException(status_code=503, detail="Speech recognizer not available")

    start_time = time.time()
    temp_files = []

    try:
        logger.info(
            f"Processing audio: {file.filename}, content_type: {file.content_type}"
        )

        # Read file content
        content = await file.read()

        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")

        if len(content) > config.API["max_file_size"]:
            raise HTTPException(status_code=413, detail="File too large")

        # Detect audio format
        detected_format = detect_audio_format(content, file.filename or "")

        # Save uploaded file
        suffix = f".{detected_format}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(content)
            temp_file_path = tmp_file.name
            temp_files.append(temp_file_path)

        logger.info(f"Saved file: {temp_file_path} (format: {detected_format})")

        # Convert to WAV if needed
        if detected_format != "wav":
            converted_path = convert_audio_if_needed(temp_file_path, detected_format)
            if converted_path != temp_file_path:
                temp_files.append(converted_path)
                temp_file_path = converted_path

        # Perform recognition
        result = await recognizer.recognize_async(temp_file_path, language)

        if result.get("status") != "success":
            error_msg = result.get("error", "Recognition failed")
            raise HTTPException(status_code=500, detail=error_msg)

        # Apply confidence threshold
        intent_confidence = result["intent"].get("confidence", 0.0)
        if intent_confidence < confidence_threshold:
            result["intent"]["intent"] = "low_confidence"
            result["intent"]["original_intent"] = result["intent"]["intent"]

        # Filter alternatives if requested
        if include_alternatives and "alternatives" in result["intent"]:
            result["intent"]["alternatives"] = [
                alt
                for alt in result["intent"]["alternatives"]
                if alt.get("score", 0) >= confidence_threshold
            ]
        elif not include_alternatives:
            result["intent"].pop("alternatives", None)

        # Add file information
        result["file_info"] = {
            "filename": file.filename,
            "content_type": file.content_type,
            "detected_format": detected_format,
            "file_size": len(content),
        }

        processing_time = time.time() - start_time
        result["total_processing_time"] = processing_time

        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_files, *temp_files)

        logger.info(
            f"Recognition completed: {result['transcription']['text'][:50]}... -> {result['intent']['intent']}"
        )

        return JSONResponse(content=result, headers=CORS_HEADERS)

    except HTTPException:
        # Re-raise HTTP exceptions
        await cleanup_temp_files(*temp_files)
        raise
    except Exception as e:
        logger.error(f"Recognition error: {e}")
        await cleanup_temp_files(*temp_files)

        error_detail = str(e)
        if "timeout" in error_detail.lower():
            raise HTTPException(status_code=408, detail="Processing timeout")
        elif "format" in error_detail.lower() or "audio" in error_detail.lower():
            raise HTTPException(status_code=400, detail="Invalid audio format")
        else:
            raise HTTPException(status_code=500, detail=error_detail)


@app.post("/test-intent")
async def test_intent(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    top_k: int = Query(
        default=5, description="Number of top intent predictions to return"
    ),
):
    """
    Test intent classification from audio (compatible with existing frontend).

    This endpoint maintains compatibility with the existing frontend
    while using the optimized recognition engine.
    """
    if not recognizer:
        raise HTTPException(status_code=503, detail="Speech recognizer not available")

    start_time = time.time()
    temp_files = []

    try:
        logger.info(
            f"Test intent request: {file.filename}, content_type: {file.content_type}, top_k: {top_k}"
        )

        # Read and validate file
        content = await file.read()

        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")

        # Detect and save file
        detected_format = detect_audio_format(content, file.filename or "")
        suffix = f".{detected_format}"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(content)
            temp_file_path = tmp_file.name
            temp_files.append(temp_file_path)

        # Convert if needed
        if detected_format != "wav":
            converted_path = convert_audio_if_needed(temp_file_path, detected_format)
            if converted_path != temp_file_path:
                temp_files.append(converted_path)
                temp_file_path = converted_path

        # Perform recognition with extended timeout for WebM
        timeout_duration = 120 if detected_format == "webm" else 60

        try:
            result = await asyncio.wait_for(
                recognizer.recognize_async(temp_file_path, "tw"),
                timeout=timeout_duration,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=408,
                detail=f"Processing timeout ({timeout_duration}s). WebM files may require longer processing.",
            )

        if result.get("status") != "success":
            error_msg = result.get("error", "Recognition failed")
            raise HTTPException(
                status_code=500, detail=f"Model inference failed: {error_msg}"
            )

        # Format response for frontend compatibility
        processing_time_ms = (time.time() - start_time) * 1000

        # Get top predictions
        alternatives = result["intent"].get("alternatives", [])
        top_predictions = alternatives[:top_k]

        # Ensure we have the main prediction in the list
        main_intent = {
            "intent": result["intent"]["intent"],
            "confidence": result["intent"]["confidence"],
            "index": 0,
        }

        # Remove duplicates and ensure main prediction is first
        seen_intents = {main_intent["intent"]}
        final_predictions = [main_intent]

        for pred in top_predictions:
            if pred.get("label") not in seen_intents:
                final_predictions.append(
                    {
                        "intent": pred.get("label", "unknown"),
                        "confidence": pred.get("score", 0.0),
                        "index": len(final_predictions),
                    }
                )
                seen_intents.add(pred.get("label"))

        response_data = {
            "filename": file.filename or "audio.wav",
            "intent": result["intent"]["intent"],
            "confidence": float(result["intent"]["confidence"]),
            "top_predictions": final_predictions[:top_k],
            "transcription": result["transcription"]["text"],
            "model_type": "optimized_whisper_intent",
            "processing_time_ms": round(processing_time_ms, 2),
            "top_k": top_k,
            "whisper_info": {
                "model_size": config.WHISPER["model_size"],
                "language": result["transcription"].get("language", "tw"),
                "transcription_confidence": result["transcription"].get(
                    "confidence", 0.0
                ),
            },
        }

        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_files, *temp_files)

        logger.info(
            f"Test intent completed: '{result['transcription']['text'][:30]}...' -> {result['intent']['intent']} ({result['intent']['confidence']:.3f})"
        )

        return JSONResponse(content=response_data, headers=CORS_HEADERS)

    except HTTPException:
        await cleanup_temp_files(*temp_files)
        raise
    except Exception as e:
        logger.error(f"Test intent error: {e}")
        await cleanup_temp_files(*temp_files)

        if "timeout" in str(e).lower():
            raise HTTPException(status_code=408, detail="Processing timeout")
        elif "webm" in str(e).lower() or "format" in str(e).lower():
            raise HTTPException(
                status_code=400, detail="Audio format processing failed"
            )
        else:
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/batch-recognize")
async def batch_recognize(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    language: str = Query(
        default=None, description="Language code (auto-detect if None)"
    ),
):
    """Process multiple audio files in parallel."""
    if not recognizer:
        raise HTTPException(status_code=503, detail="Speech recognizer not available")

    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Too many files (max 10)")

    start_time = time.time()
    temp_files = []

    try:
        # Save all files first
        file_paths = []
        for i, file in enumerate(files):
            content = await file.read()
            detected_format = detect_audio_format(content, file.filename or "")

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{detected_format}"
            ) as tmp_file:
                tmp_file.write(content)
                temp_path = tmp_file.name
                temp_files.append(temp_path)

                # Convert if needed
                if detected_format != "wav":
                    converted_path = convert_audio_if_needed(temp_path, detected_format)
                    if converted_path != temp_path:
                        temp_files.append(converted_path)
                        temp_path = converted_path

                file_paths.append(temp_path)

        # Process in parallel
        results = recognizer.recognize_stream(file_paths, language)

        # Add file info to results
        for i, result in enumerate(results):
            result["file_info"] = {"filename": files[i].filename, "index": i}

        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_files, *temp_files)

        return JSONResponse(
            content={
                "results": results,
                "total_files": len(files),
                "total_processing_time": time.time() - start_time,
            },
            headers=CORS_HEADERS,
        )

    except Exception as e:
        logger.error(f"Batch recognition error: {e}")
        await cleanup_temp_files(*temp_files)
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(413)
async def payload_too_large_handler(request, exc):
    """Handle file too large errors."""
    return JSONResponse(
        status_code=413,
        content={"detail": "Audio file too large. Maximum size: 50MB"},
        headers=CORS_HEADERS,
    )


@app.exception_handler(422)
async def validation_error_handler(request, exc):
    """Handle validation errors."""
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request format"},
        headers=CORS_HEADERS,
    )


# Server startup function
def start_server():
    """Start the FastAPI server."""
    host = config.API["host"]
    port = int(os.environ.get("PORT", config.API["port"]))

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
        workers=1,  # Single worker for model consistency
        loop="asyncio",
    )
