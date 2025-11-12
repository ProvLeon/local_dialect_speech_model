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

import asyncio
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import soundfile as sf
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config.config import OptimizedConfig

# Import optimized components
from src.speech_recognizer import OptimizedSpeechRecognizer, create_speech_recognizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global recognizer instance
recognizer: Optional[OptimizedSpeechRecognizer] = None
config = OptimizedConfig()

# Performance optimization imports and setup
import gc
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Union

# Global performance tracking
performance_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "avg_processing_time": 0.0,
    "cache_hits": 0,
}

# Audio conversion cache and thread pool
audio_cache = {}
thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="audio_proc_")


def optimize_torch_settings():
    """Optimize PyTorch settings for faster inference."""
    import torch

    # Set optimal thread count
    torch.set_num_threads(min(4, os.cpu_count()))

    # Enable inference optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Disable gradients for inference
    torch.set_grad_enabled(False)

    if torch.cuda.is_available():
        # Clear CUDA cache
        torch.cuda.empty_cache()

        # Set memory fraction to prevent OOM
        torch.cuda.set_per_process_memory_fraction(0.8)

        # Enable TensorFloat-32 for newer GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        logger.info(
            f"✅ CUDA optimizations applied - Device: {torch.cuda.get_device_name()}"
        )


def optimize_model_loading(speech_recognizer):
    """Optimize model loading for better performance."""
    if not speech_recognizer:
        return

    try:
        import torch

        # Optimize Whisper model
        if (
            hasattr(speech_recognizer, "transcriber")
            and speech_recognizer.transcriber.model
        ):
            model = speech_recognizer.transcriber.model

            if torch.cuda.is_available() and not model.is_cuda:
                model = model.cuda()

                # Enable half precision for faster inference
                try:
                    model = model.half()
                    logger.info("✅ Half precision enabled for Whisper model")
                except Exception as e:
                    logger.warning(f"⚠️ Could not enable half precision: {e}")

        # Optimize intent classifier
        if (
            hasattr(speech_recognizer, "intent_classifier")
            and speech_recognizer.intent_classifier.pipeline
        ):
            pipeline = speech_recognizer.intent_classifier.pipeline

            if torch.cuda.is_available() and hasattr(pipeline, "model"):
                try:
                    pipeline.model = pipeline.model.cuda()
                    if hasattr(pipeline.model, "half"):
                        pipeline.model = pipeline.model.half()
                    logger.info("✅ Intent classifier optimized for GPU")
                except Exception as e:
                    logger.warning(f"⚠️ Intent classifier GPU optimization failed: {e}")

        logger.info("✅ Model optimization completed")

    except Exception as e:
        logger.error(f"❌ Model optimization failed: {e}")


async def convert_audio_fast(content: bytes, filename: str = "") -> str:
    """Ultra-fast audio conversion using optimized FFmpeg."""

    # Detect format quickly
    audio_format = detect_audio_format_optimized(content, filename)

    # Create temporary file
    suffix = f".{audio_format}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(content)
        temp_path = tmp_file.name

    # Skip conversion if already WAV
    if audio_format == "wav":
        return temp_path

    # Use optimized FFmpeg conversion
    output_path = temp_path.replace(f".{audio_format}", "_fast.wav")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        temp_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-f",
        "wav",
        "-threads",
        "0",
        output_path,
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
        )

        await asyncio.wait_for(process.communicate(), timeout=10.0)

        if process.returncode == 0 and os.path.exists(output_path):
            # Clean up original
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return output_path
        else:
            raise RuntimeError("FFmpeg conversion failed")

    except (asyncio.TimeoutError, FileNotFoundError, RuntimeError):
        # Fallback to librosa
        return await convert_with_librosa_async(temp_path, audio_format)


async def convert_with_librosa_async(input_path: str, input_format: str) -> str:
    """Async librosa conversion fallback."""
    loop = asyncio.get_event_loop()

    def _convert():
        import librosa

        audio, sr = librosa.load(input_path, sr=16000, mono=True)
        output_path = input_path.replace(f".{input_format}", "_librosa.wav")
        sf.write(output_path, audio, 16000)

        # Clean up original
        if os.path.exists(input_path):
            os.unlink(input_path)

        return output_path

    return await loop.run_in_executor(thread_pool, _convert)


def detect_audio_format_optimized(content: bytes, filename: str) -> str:
    """Fast audio format detection."""
    # Check filename extension first (fastest)
    if filename:
        ext = Path(filename).suffix.lower()
        if ext in [".wav", ".mp3", ".webm", ".ogg"]:
            return ext[1:]

    # Check magic bytes
    if content.startswith(b"RIFF"):
        return "wav"
    elif content.startswith(b"ID3") or content.startswith(b"\xff\xfb"):
        return "mp3"
    elif content.startswith(b"OggS"):
        return "ogg"
    elif b"webm" in content[:100] or content.startswith(b"\x1a\x45\xdf\xa3"):
        return "webm"

    return "unknown"


def cleanup_memory():
    """Optimize memory usage and cleanup."""
    try:
        gc.collect()

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Clear audio cache if too large
        if len(audio_cache) > 50:
            items = list(audio_cache.items())
            audio_cache.clear()
            audio_cache.update(dict(items[-25:]))  # Keep 25 most recent
            logger.info("🧹 Audio cache cleaned")

    except Exception as e:
        logger.warning(f"Memory cleanup warning: {e}")


def update_performance_stats(processing_time: float, success: bool = True):
    """Update performance statistics."""
    performance_stats["total_requests"] += 1

    if success:
        performance_stats["successful_requests"] += 1
    else:
        performance_stats["failed_requests"] += 1

    # Update rolling average processing time
    total = performance_stats["total_requests"]
    current_avg = performance_stats["avg_processing_time"]
    new_avg = (current_avg * (total - 1) + processing_time) / total
    performance_stats["avg_processing_time"] = new_avg


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with performance optimizations."""
    global recognizer

    # Startup
    logger.info("🚀 Starting Ultra-Fast Twi Speech Recognition Server...")

    try:
        # Apply PyTorch optimizations first
        logger.info("🔧 Applying performance optimizations...")
        optimize_torch_settings()

        # Initialize speech recognizer
        recognizer = create_speech_recognizer()

        # Optimize the loaded models for better performance
        optimize_model_loading(recognizer)

        # Validate configuration
        config.validate_config()

        # Print startup summary
        config.print_config_summary()

        # Log optimization status
        import torch

        device_info = (
            f"GPU: {torch.cuda.get_device_name()}"
            if torch.cuda.is_available()
            else "CPU"
        )
        logger.info(f"✅ Server ready with optimizations - Device: {device_info}")
        logger.info("📊 Performance monitoring available at /performance-stats")

    except Exception as e:
        logger.error(f"❌ Failed to initialize server: {e}")
        raise

    yield

    # Shutdown
    logger.info("🛑 Shutting down server...")

    # Cleanup resources
    if thread_pool:
        thread_pool.shutdown(wait=True)

    audio_cache.clear()
    cleanup_memory()


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
            f"🎵 Fast processing: {file.filename} ({file.content_type}), top_k: {top_k}"
        )

        # Read and validate file
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")

        # Check cache first
        content_hash = str(hash(content))
        if content_hash in audio_cache:
            performance_stats["cache_hits"] += 1
            logger.info(f"✅ Cache hit for {file.filename}")
            cached_result = audio_cache[content_hash]
            update_performance_stats(0.1, success=True)  # Very fast cache response
            return cached_result

        # Use optimized audio conversion
        temp_file_path = await convert_audio_fast(content, file.filename or "")
        temp_files.append(temp_file_path)

        # Perform recognition with optimized timeout and async processing
        timeout_duration = 45  # Reduced from 60-120s to 45s max

        try:
            # Use thread pool for better async performance
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    thread_pool, recognizer.recognize, temp_file_path, "tw"
                ),
                timeout=timeout_duration,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=408,
                detail=f"Processing timeout ({timeout_duration}s). Please try a shorter audio file.",
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
            "model_type": "optimized_whisper_intent_fast",
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

        # Cache successful results for future requests
        if processing_time_ms < 10000:  # Only cache results under 10s
            audio_cache[content_hash] = response_data

        # Update performance statistics
        update_performance_stats(processing_time_ms / 1000.0, success=True)

        # Schedule cleanup and memory optimization
        background_tasks.add_task(cleanup_temp_files, *temp_files)
        background_tasks.add_task(cleanup_memory)

        logger.info(
            f"✅ Completed in {processing_time_ms:.0f}ms: '{result['transcription']['text'][:50]}...' -> {result['intent']['intent']} ({result['intent']['confidence']:.3f})"
        )

        return JSONResponse(content=response_data, headers=CORS_HEADERS)

    except HTTPException:
        await cleanup_temp_files(*temp_files)
        update_performance_stats(time.time() - start_time, success=False)
        raise
    except Exception as e:
        logger.error(f"❌ Request failed: {e}")
        await cleanup_temp_files(*temp_files)
        update_performance_stats(time.time() - start_time, success=False)

        if "timeout" in str(e).lower():
            raise HTTPException(
                status_code=408, detail="Processing timeout - try shorter audio"
            )
        elif "webm" in str(e).lower() or "format" in str(e).lower():
            raise HTTPException(
                status_code=400, detail="Audio format processing failed"
            )
        else:
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/performance-stats")
async def get_performance_stats():
    """Get detailed performance statistics and optimization status."""
    import torch

    stats = {
        "performance_metrics": performance_stats.copy(),
        "optimization_status": {
            "torch_optimized": True,
            "cuda_available": torch.cuda.is_available(),
            "cache_enabled": True,
            "thread_pool_active": thread_pool._threads if thread_pool else 0,
        },
        "cache_info": {
            "audio_cache_size": len(audio_cache),
            "cache_hit_rate": performance_stats["cache_hits"]
            / max(1, performance_stats["total_requests"]),
        },
        "recommendations": [
            "Use GPU for 3-5x faster processing",
            "Keep audio files under 30 seconds",
            "Use WAV format when possible",
            f"Current avg response time: {performance_stats['avg_processing_time']:.2f}s",
        ],
    }

    if torch.cuda.is_available():
        stats["gpu_info"] = {
            "device_name": torch.cuda.get_device_name(),
            "memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
        }

    return JSONResponse(content=stats)


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
