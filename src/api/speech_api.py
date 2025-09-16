# src/api/speech_api.py - Akan (Twi) Speech-to-Action API
"""
Akan (Twi) Speech-to-Action API

Features:
* Enhanced and standard model loading with automatic path resolution
* Maps always from data/processed unless explicitly overridden
* Models from data/models by default, with custom base directory support
* Runtime model reconfiguration via API endpoints
* Comprehensive diagnostics and metrics
* E-commerce action integration
"""
from __future__ import annotations

import os
import json
import time
import tempfile
import logging
import asyncio
from typing import Optional, Dict, Any, List, Tuple
from threading import RLock

import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.models.speech_model import ImprovedTwiSpeechModel, EnhancedTwiSpeechModel
from src.preprocessing.audio_processor import AudioProcessor
from src.preprocessing.enhanced_audio_processor import EnhancedAudioProcessor
from src.utils.ecommerce_integration import EcommerceIntegration
from src.utils.model_utils import load_label_map, get_model_input_dim
from src.utils.audio_converter import convert_bytes_to_wav, validate_audio_file
from config.model_config import MODEL_CONFIG
from config.api_config import API_CONFIG

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI App & CORS
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Akan (Twi) Speech-to-Action API",
    description="Recognize Twi speech commands and map to e-commerce actions.",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CONFIG.get("allowed_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Model state
enhanced_classifier: Optional[Dict[str, Any]] = None
standard_classifier: Optional[Dict[str, Any]] = None

# Configuration
confidence_threshold = API_CONFIG.get("confidence_threshold", 0.7)
request_count = 0
start_time = time.time()
processing_times: List[float] = []

# E-commerce integration
ecommerce = EcommerceIntegration()

# Path configuration
MODEL_BASE_DIR: str = ""
DEFAULT_MODELS_ROOT = "data/models"
DEFAULT_MAPS_ROOT = "data/processed"
CANDIDATE_PATHS: Dict[str, List[str]] = {}

# Resolved paths
ENHANCED_MODEL_PATH: str = ""
STANDARD_MODEL_PATH: str = ""
LABEL_MAP_PATH: str = ""

# Thread safety
_model_lock = RLock()

# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    available_intents: Optional[int] = None
    api_version: str = "1.2.0"

class RecognitionResponse(BaseModel):
    filename: str
    intent: str
    confidence: float
    model_type: str
    processing_time_ms: float

class ActionResponse(BaseModel):
    recognition: Dict[str, Any]
    action: Dict[str, Any]

# ---------------------------------------------------------------------------
# Path Resolution Functions
# ---------------------------------------------------------------------------
def _file_exists(path: str) -> bool:
    """Check if file exists"""
    return os.path.exists(path) and os.path.isfile(path)

def resolve_model_paths(
    model_base_dir: Optional[str] = None,
    explicit_enhanced: Optional[str] = None,
    explicit_standard: Optional[str] = None,
    explicit_label: Optional[str] = None
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Resolve model and label map paths with proper fallback logic.

    Priority:
    1. Explicit paths (if provided)
    2. Files in custom base directory (if provided)
    3. Files in default data/models
    4. 47-class training results directories
    5. Legacy fallback paths
    """
    base_dir = (model_base_dir or "").strip().rstrip("/")
    models_root = base_dir if base_dir else DEFAULT_MODELS_ROOT

    logger.debug(f"Resolving paths with base_dir='{base_dir}', models_root='{models_root}'")

    # Build candidate lists
    enhanced_candidates: List[str] = []
    standard_candidates: List[str] = []

    if base_dir:
        # Custom base directory - look for models inside it
        enhanced_candidates.extend([
            os.path.join(models_root, "enhanced", "best_model.pt"),
            os.path.join(models_root, "best_model.pt"),
            os.path.join(models_root, "model.pt"),
        ])
        standard_candidates.extend([
            os.path.join(models_root, "standard", "best_model.pt"),
            os.path.join(models_root, "best_model.pt"),
            os.path.join(models_root, "model.pt"),
        ])
    else:
        # Default case - look in data/models
        enhanced_candidates.extend([
            os.path.join(DEFAULT_MODELS_ROOT, "enhanced", "best_model.pt"),
            os.path.join(DEFAULT_MODELS_ROOT, "best_model.pt"),
            os.path.join(DEFAULT_MODELS_ROOT, "model.pt"),
        ])
        standard_candidates.extend([
            os.path.join(DEFAULT_MODELS_ROOT, "standard", "best_model.pt"),
            os.path.join(DEFAULT_MODELS_ROOT, "best_model.pt"),
            os.path.join(DEFAULT_MODELS_ROOT, "model.pt"),
        ])

        # Look for 47-class training results
        results_dir = os.path.join(DEFAULT_MODELS_ROOT, "47_class_results")
        if os.path.exists(results_dir):
            # Find the most recent training result
            subdirs = [d for d in os.listdir(results_dir)
                      if os.path.isdir(os.path.join(results_dir, d))]
            if subdirs:
                # Sort by modification time, get most recent
                subdirs.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
                latest_result = os.path.join(results_dir, subdirs[0], "best_model.pt")
                if os.path.exists(latest_result):
                    enhanced_candidates.insert(1, latest_result)
                    standard_candidates.insert(1, latest_result)

    # Add legacy fallback (only if not already covered)
    legacy_path = "data/models_improved/best_model.pt"
    if legacy_path not in enhanced_candidates:
        enhanced_candidates.append(legacy_path)
    if legacy_path not in standard_candidates:
        standard_candidates.append(legacy_path)

    # Label map candidates (always from processed unless explicit override)
    label_candidates: List[str] = []
    if explicit_label:
        label_candidates.append(explicit_label)
    else:
        label_candidates.extend([
            os.path.join(DEFAULT_MAPS_ROOT, "label_map.json"),
            os.path.join(DEFAULT_MAPS_ROOT, "label_map.npy"),
        ])

    # Resolve paths
    paths: Dict[str, str] = {}

    # Enhanced model
    if explicit_enhanced:
        paths["ENHANCED_MODEL_PATH"] = explicit_enhanced
    else:
        for candidate in enhanced_candidates:
            if _file_exists(candidate):
                paths["ENHANCED_MODEL_PATH"] = candidate
                break
        else:
            # No existing file found, use first candidate as fallback
            paths["ENHANCED_MODEL_PATH"] = enhanced_candidates[0]

    # Standard model (can reuse enhanced)
    if explicit_standard:
        paths["STANDARD_MODEL_PATH"] = explicit_standard
    else:
        # Add resolved enhanced path as first candidate for standard
        standard_candidates.insert(0, paths["ENHANCED_MODEL_PATH"])
        for candidate in standard_candidates:
            if _file_exists(candidate):
                paths["STANDARD_MODEL_PATH"] = candidate
                break
        else:
            paths["STANDARD_MODEL_PATH"] = standard_candidates[0]

    # Label map
    for candidate in label_candidates:
        if os.path.exists(candidate):
            paths["LABEL_MAP_PATH"] = candidate
            break
    else:
        paths["LABEL_MAP_PATH"] = os.path.join(DEFAULT_MAPS_ROOT, "label_map.json")

    candidates = {
        "enhanced": enhanced_candidates,
        "standard": standard_candidates,
        "label_map": label_candidates
    }

    # Log resolution results
    logger.info(f"Path resolution completed:")
    logger.info(f"  MODEL_BASE_DIR: {base_dir or '(default)'}")
    logger.info(f"  ENHANCED_MODEL_PATH: {paths['ENHANCED_MODEL_PATH']}")
    logger.info(f"  STANDARD_MODEL_PATH: {paths['STANDARD_MODEL_PATH']}")
    logger.info(f"  LABEL_MAP_PATH: {paths['LABEL_MAP_PATH']}")

    # Warn about fallbacks
    if base_dir and not _file_exists(paths["ENHANCED_MODEL_PATH"]):
        logger.warning(f"Enhanced model not found in custom base directory: {base_dir}")

    return paths, candidates

def apply_resolved_paths(resolved: Dict[str, str], candidates: Dict[str, List[str]]):
    """Apply resolved paths to global variables"""
    global ENHANCED_MODEL_PATH, STANDARD_MODEL_PATH, LABEL_MAP_PATH, CANDIDATE_PATHS

    ENHANCED_MODEL_PATH = resolved["ENHANCED_MODEL_PATH"]
    STANDARD_MODEL_PATH = resolved["STANDARD_MODEL_PATH"]
    LABEL_MAP_PATH = resolved["LABEL_MAP_PATH"]
    CANDIDATE_PATHS = candidates

    logger.info(f"Applied resolved paths: E={ENHANCED_MODEL_PATH}, S={STANDARD_MODEL_PATH}, L={LABEL_MAP_PATH}")

def initialize_paths():
    """Initialize model paths from environment variables - called at startup"""
    global MODEL_BASE_DIR

    # Get MODEL_BASE_DIR from environment (set by app.py if provided via CLI)
    MODEL_BASE_DIR = os.environ.get("MODEL_BASE_DIR", "").strip().rstrip("/")

    # Resolve paths
    resolved_paths, candidates = resolve_model_paths(
        MODEL_BASE_DIR,
        explicit_enhanced=os.environ.get("ENHANCED_MODEL_PATH"),
        explicit_standard=os.environ.get("MODEL_PATH"),
        explicit_label=os.environ.get("LABEL_MAP_PATH")
    )

    # Apply resolved paths
    apply_resolved_paths(resolved_paths, candidates)

# ---------------------------------------------------------------------------
# Model Loading Functions
# ---------------------------------------------------------------------------
def create_processor() -> AudioProcessor:
    """Create audio processor with standard settings"""
    return AudioProcessor(
        sample_rate=MODEL_CONFIG.get("sample_rate", 16000),
        n_mfcc=MODEL_CONFIG.get("n_mfcc", 13),
        n_fft=MODEL_CONFIG.get("n_fft", 2048),
        hop_length=MODEL_CONFIG.get("hop_length", 512),
        enable_deltas=True,
        enable_audio_augment=False,
        enable_spec_augment=False
    )

def load_label_map_safe() -> Dict[str, int]:
    """Load label map with error handling"""
    try:
        return load_label_map(LABEL_MAP_PATH)
    except Exception as e:
        logger.error(f"Failed to load label map from {LABEL_MAP_PATH}: {e}")
        return {}

def create_model(model_path: str, label_map: Dict[str, int]) -> Optional[ImprovedTwiSpeechModel]:
    """Create and load model from file"""
    if not os.path.exists(model_path):
        logger.warning(f"Model file missing: {model_path}")
        return None

    # Default input dimension for AudioProcessor with deltas
    input_dim = 39  # 13 MFCC + 13 delta + 13 delta-delta

    try:
        # Try to get from model metadata if available
        model_dir = os.path.dirname(model_path)
        results_path = os.path.join(model_dir, 'final_results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
                model_info = results.get('model_info', {})
                input_dim = model_info.get('input_dim', input_dim)
                logger.info(f"Loaded input_dim={input_dim} from training metadata")
    except Exception as e:
        logger.warning(f"Could not load training metadata: {e}, using default input_dim={input_dim}")

    # Create model with slot support
    model = ImprovedTwiSpeechModel(
        input_dim=input_dim,
        hidden_dim=MODEL_CONFIG.get("hidden_dim", 128),
        num_classes=len(label_map) if label_map else MODEL_CONFIG.get("num_classes", 47),
        num_layers=MODEL_CONFIG.get("num_layers", 2),
        dropout=MODEL_CONFIG.get("dropout", 0.5),
        num_heads=MODEL_CONFIG.get("num_heads", 4),
        num_slot_classes=0,  # No slot classification for API
        slot_value_maps={}   # Empty slot maps for API
    )

    try:
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        model.to(device)
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return None

def load_enhanced_model() -> bool:
    """Load enhanced model"""
    global enhanced_classifier

    with _model_lock:
        label_map = load_label_map_safe()
        model = create_model(ENHANCED_MODEL_PATH, label_map)

        if model is None:
            enhanced_classifier = None
            return False

        enhanced_classifier = {
            "model": model,
            "processor": create_processor(),
            "label_map": label_map,
            "input_dim": getattr(model, "input_dim", None),
            "model_type": "enhanced"
        }

        logger.info(f"Enhanced model loaded successfully from {ENHANCED_MODEL_PATH}")
        return True

def load_standard_model() -> bool:
    """Load standard model"""
    global standard_classifier

    with _model_lock:
        label_map = load_label_map_safe()
        model = create_model(STANDARD_MODEL_PATH, label_map)

        if model is None:
            standard_classifier = None
            return False

        standard_classifier = {
            "model": model,
            "processor": create_processor(),
            "label_map": label_map,
            "input_dim": getattr(model, "input_dim", None),
            "model_type": "standard"
        }

        logger.info(f"Standard model loaded successfully from {STANDARD_MODEL_PATH}")
        return True

def get_active_classifier() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Get active classifier with preference for enhanced"""
    if enhanced_classifier:
        return enhanced_classifier, "enhanced"
    if standard_classifier:
        return standard_classifier, "standard"
    return None, None

# ---------------------------------------------------------------------------
# Audio Processing Functions
# ---------------------------------------------------------------------------
def preprocess_audio(audio_path: str, classifier_obj: Dict[str, Any]) -> torch.Tensor:
    """Preprocess audio file for model input with timeout handling"""
    processor = classifier_obj["processor"]

    logger.info(f"Preprocessing audio: {audio_path}")

    # Extract features using AudioProcessor with timeout
    features = processor.preprocess(audio_path, timeout_seconds=60)
    logger.info(f"Features extracted with shape: {features.shape}")

    # Handle variable length by padding/truncating to reasonable size
    if features.shape[1] < 50:
        pad_width = 50 - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
        logger.debug(f"Padded features to shape: {features.shape}")
    elif features.shape[1] > 200:
        features = features[:, :200]
        logger.debug(f"Truncated features to shape: {features.shape}")

    # Convert to tensor
    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    logger.info(f"Created tensor with shape: {tensor.shape}")
    return tensor

def classify_audio(audio_path: str, classifier_obj: Dict[str, Any]) -> Tuple[str, float]:
    """Classify audio file and return intent and confidence"""
    tensor = preprocess_audio(audio_path, classifier_obj)
    model = classifier_obj["model"]
    label_map = classifier_obj["label_map"]

    with torch.no_grad():
        outputs = model(tensor)

        # Handle tuple output from model (intent_logits, slot_logits)
        if isinstance(outputs, tuple):
            intent_logits = outputs[0]
        else:
            intent_logits = outputs

        probs = torch.softmax(intent_logits, dim=1)
        conf, idx = probs.max(1)

    # Convert index to label
    idx_to_label = {v: k for k, v in label_map.items()}
    intent = idx_to_label.get(idx.item(), f"cls_{idx.item()}")

    return intent, float(conf.item())

# ---------------------------------------------------------------------------
# Startup Event
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global start_time
    start_time = time.time()

    # Initialize paths first (reads environment variables set by app.py)
    initialize_paths()

    # Try to load enhanced model first
    enhanced_loaded = load_enhanced_model()

    # If enhanced model failed, try standard model
    if not enhanced_loaded:
        standard_loaded = load_standard_model()
    else:
        standard_loaded = True  # Enhanced model can serve as standard too

    # Check if any model was loaded
    clf, model_type = get_active_classifier()
    if clf:
        logger.info(f"API started successfully with {model_type} model")
    else:
        logger.error("No models could be loaded at startup!")

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    """Root endpoint"""
    clf, model_type = get_active_classifier()
    return {
        "message": "Akan (Twi) Speech-to-Action API",
        "version": "1.2.0",
        "model_type": model_type or "none",
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    clf, model_type = get_active_classifier()

    if not clf:
        return JSONResponse(
            status_code=503,
            content=HealthResponse(
                status="error",
                model_loaded=False,
                model_type="none"
            ).dict()
        )

    num_intents = len(clf.get("label_map", {}))
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_type=model_type or "unknown",
        available_intents=num_intents
    )

@app.post("/recognize", response_model=RecognitionResponse)
async def recognize(file: UploadFile = File(...)):
    """Recognize speech intent from audio file"""
    global request_count, processing_times
    request_count += 1

    clf, model_type = get_active_classifier()
    if not clf:
        raise HTTPException(status_code=503, detail="No model loaded")

    # Validate file
    if not file.filename or not file.filename.lower().endswith((".wav", ".mp3", ".webm", ".ogg", ".m4a", ".aac", ".flac")):
        raise HTTPException(status_code=400, detail="Audio format not supported. Please use WAV, MP3, WebM, OGG, M4A, AAC, or FLAC format.")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    max_size = API_CONFIG.get("max_upload_size", 10 * 1024 * 1024)
    if len(content) > max_size:
        raise HTTPException(status_code=413, detail="File too large")

    tmp_path = None
    try:
        # Convert audio bytes to WAV format using robust converter
        tmp_path = convert_bytes_to_wav(content, file.filename or "audio.wav")

        if not tmp_path:
            raise HTTPException(status_code=400, detail="Failed to convert audio to supported format")

        # Validate the converted audio
        if not validate_audio_file(tmp_path):
            raise HTTPException(status_code=400, detail="Converted audio file is invalid or corrupted")

        # Process audio
        start_time_proc = time.time()
        intent, confidence = classify_audio(tmp_path, clf)
        proc_time_ms = (time.time() - start_time_proc) * 1000

        processing_times.append(proc_time_ms)

        return RecognitionResponse(
            filename=file.filename,
            intent=intent,
            confidence=confidence,
            model_type=model_type or "unknown",
            processing_time_ms=proc_time_ms
        )

    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        if "Format not recognised" in str(e) or "NoBackendError" in str(e):
            raise HTTPException(status_code=400, detail="Audio format not supported. Please use WAV, MP3, WebM, OGG, M4A, AAC, or FLAC format.")
        elif "NaN" in str(e) or "Input X contains NaN" in str(e) or "not finite everywhere" in str(e):
            raise HTTPException(status_code=400, detail="Audio processing failed. The audio may be corrupted, too short, or contains invalid data.")
        elif "Audio buffer is not finite everywhere" in str(e):
            raise HTTPException(status_code=400, detail="Audio contains invalid data (NaN or infinite values). Please check your recording.")
        elif "divide by zero" in str(e) or "division by zero" in str(e):
            raise HTTPException(status_code=400, detail="Audio processing failed due to silence or very low signal. Please record with more volume.")
        else:
            raise HTTPException(status_code=500, detail=f"Error processing audio: {e}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.post("/action", response_model=ActionResponse)
async def take_action(
    file: UploadFile = File(...),
    user_id: str = Query(..., description="User ID for e-commerce actions")
):
    """Recognize speech and take e-commerce action"""
    # First recognize the speech
    recognition_result = await recognize(file)

    # Check confidence threshold
    if recognition_result.confidence < confidence_threshold:
        return ActionResponse(
            recognition=recognition_result.dict(),
            action={
                "status": "low_confidence",
                "message": f"Confidence {recognition_result.confidence:.2f} below threshold {confidence_threshold}",
                "confidence": recognition_result.confidence
            }
        )

    # Execute e-commerce action
    try:
        action_result = ecommerce.execute_action(
            intent=recognition_result.intent,
            user_id=user_id,
            confidence=recognition_result.confidence
        )

        return ActionResponse(
            recognition=recognition_result.dict(),
            action=action_result
        )

    except Exception as e:
        logger.error(f"Error executing action: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing action: {e}")

@app.get("/intents")
async def list_intents():
    """List all available intents"""
    clf, model_type = get_active_classifier()
    if not clf:
        raise HTTPException(status_code=503, detail="No model loaded")

    intents = list(clf["label_map"].keys())
    result = [{"intent": intent, "description": "No description available"} for intent in intents]

    return {
        "intents": result,
        "count": len(result),
        "model_type": model_type
    }

@app.get("/model-info")
async def model_info():
    """Get detailed model information"""
    clf, model_type = get_active_classifier()
    if not clf:
        raise HTTPException(status_code=503, detail="No model loaded")

    return {
        "model_type": model_type,
        "input_dim": clf.get("input_dim"),
        "num_classes": len(clf.get("label_map", {})),
        "processor": clf.get("processor").__class__.__name__ if clf.get("processor") else None,
        "model_path": ENHANCED_MODEL_PATH if model_type == "enhanced" else STANDARD_MODEL_PATH
    }

@app.get("/model-paths")
async def model_paths():
    """Get current model paths and candidates"""
    def summarize(path: str):
        return {
            "path": path,
            "exists": os.path.exists(path),
            "size_bytes": os.path.getsize(path) if os.path.exists(path) and os.path.isfile(path) else None
        }

    return {
        "MODEL_BASE_DIR": MODEL_BASE_DIR or "(default)",
        "resolved": {
            "ENHANCED_MODEL_PATH": summarize(ENHANCED_MODEL_PATH),
            "STANDARD_MODEL_PATH": summarize(STANDARD_MODEL_PATH),
            "LABEL_MAP_PATH": summarize(LABEL_MAP_PATH),
        },
        "candidates": CANDIDATE_PATHS
    }

@app.get("/metrics")
async def metrics():
    """Get API metrics"""
    clf, model_type = get_active_classifier()
    if not clf:
        raise HTTPException(status_code=503, detail="No model loaded")

    uptime = int(time.time() - start_time)
    avg_ms = sum(processing_times) / len(processing_times) if processing_times else 0.0
    rpm = round(request_count / (uptime / 60), 2) if uptime > 0 else 0.0

    return {
        "model_type": model_type,
        "uptime_seconds": uptime,
        "total_requests": request_count,
        "requests_per_minute": rpm,
        "avg_processing_time_ms": round(avg_ms, 2)
    }

@app.post("/reload-model")
async def reload_model(
    model_type: str = Query(..., description="Model type: 'enhanced' or 'standard'"),
    model_path: Optional[str] = Query(None, description="Optional explicit model path")
):
    """Reload a model"""
    if model_type not in ("enhanced", "standard"):
        raise HTTPException(status_code=400, detail="model_type must be 'enhanced' or 'standard'")

    with _model_lock:
        # Get previous labels for comparison
        if model_type == "enhanced" and enhanced_classifier:
            prev_labels = set(enhanced_classifier.get("label_map", {}).keys())
        elif model_type == "standard" and standard_classifier:
            prev_labels = set(standard_classifier.get("label_map", {}).keys())
        else:
            prev_labels = set()

        # Override path if provided
        if model_path:
            if not os.path.exists(model_path):
                raise HTTPException(status_code=400, detail=f"Model path does not exist: {model_path}")

            global ENHANCED_MODEL_PATH, STANDARD_MODEL_PATH
            if model_type == "enhanced":
                ENHANCED_MODEL_PATH = model_path
            else:
                STANDARD_MODEL_PATH = model_path

        # Reload model
        success = load_enhanced_model() if model_type == "enhanced" else load_standard_model()
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to reload {model_type} model")

        # Get new labels
        clf = enhanced_classifier if model_type == "enhanced" else standard_classifier
        new_labels = set(clf.get("label_map", {}).keys()) if clf else set()

        return {
            "status": "success",
            "model_type": model_type,
            "reloaded": True,
            "model_path": model_path or "(unchanged)",
            "num_classes": len(new_labels),
            "label_diff": {
                "added": sorted(list(new_labels - prev_labels)),
                "removed": sorted(list(prev_labels - new_labels)),
                "unchanged": len(new_labels & prev_labels)
            }
        }

@app.post("/configure-model-base")
async def configure_model_base(
    model_base_dir: Optional[str] = Query(None, description="New base directory"),
    enhanced_path: Optional[str] = Query(None, description="Explicit enhanced model path"),
    standard_path: Optional[str] = Query(None, description="Explicit standard model path"),
    label_map_path: Optional[str] = Query(None, description="Explicit label map path")
):
    """Reconfigure model paths and reload"""
    global MODEL_BASE_DIR

    with _model_lock:
        # Update base directory
        if model_base_dir is not None:
            MODEL_BASE_DIR = model_base_dir.strip().rstrip("/")

        # Resolve new paths
        resolved, candidates = resolve_model_paths(
            MODEL_BASE_DIR,
            explicit_enhanced=enhanced_path,
            explicit_standard=standard_path,
            explicit_label=label_map_path
        )

        # Apply new paths
        apply_resolved_paths(resolved, candidates)

        # Reload models
        enhanced_ok = load_enhanced_model()
        standard_ok = load_standard_model()

        # Determine active model
        active = "enhanced" if enhanced_classifier else "standard" if standard_classifier else "none"

        return {
            "status": "success",
            "active_model": active,
            "paths": resolved,
            "candidates": candidates,
            "enhanced_loaded": enhanced_ok,
            "standard_loaded": standard_ok
        }

# Additional utility endpoints for testing and debugging
@app.post("/test-intent")
async def test_intent(
    file: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=50, description="Number of top intents to return")
):
    """Test intent recognition with top-k results"""
    clf, model_type = get_active_classifier()
    if not clf:
        raise HTTPException(status_code=503, detail="No model loaded")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    if not file.filename or not file.filename.lower().endswith((".wav", ".mp3", ".webm", ".ogg", ".m4a", ".aac", ".flac")):
        raise HTTPException(status_code=400, detail="Audio format not supported. Please use WAV, MP3, WebM, OGG, M4A, AAC, or FLAC format.")

    logger.info(f"Received file for test-intent: {file.filename}, content_type: {file.content_type}, top_k: {top_k}")

    tmp_path = None
    try:
        start_time_total = time.time()

        # Convert audio bytes to WAV format using robust converter with timeout
        logger.info(f"Starting audio conversion for: {file.filename}")
        try:
            tmp_path = await asyncio.wait_for(
                asyncio.to_thread(convert_bytes_to_wav, content, file.filename or "audio.wav", None, 30),
                timeout=30.0
            )
            logger.info(f"Audio conversion completed, temp file: {tmp_path}")
        except asyncio.TimeoutError:
            logger.error(f"Audio conversion timed out for: {file.filename}")
            raise HTTPException(status_code=408, detail="Audio conversion timed out. Please try a shorter audio file or different format.")

        if not tmp_path:
            logger.error(f"Audio conversion failed for: {file.filename}")
            raise HTTPException(status_code=400, detail="Failed to convert audio to supported format. Please try a different audio file or check the audio format.")

        logger.info(f"Saved file to: {tmp_path}")

        # Validate the converted audio with timeout
        try:
            validation_result = await asyncio.wait_for(
                asyncio.to_thread(validate_audio_file, tmp_path, 10),
                timeout=15.0
            )
            if not validation_result:
                logger.error(f"Audio validation failed for: {tmp_path}")
                raise HTTPException(status_code=400, detail="Converted audio file is invalid, corrupted, or contains only silence. Please check your recording.")
        except asyncio.TimeoutError:
            logger.error(f"Audio validation timed out for: {tmp_path}")
            raise HTTPException(status_code=408, detail="Audio validation timed out. Please try a different audio file.")

        # Additional validation - check file size
        if os.path.getsize(tmp_path) == 0:
            logger.error(f"Converted audio file is empty: {tmp_path}")
            raise HTTPException(status_code=400, detail="Converted audio file is empty. Please check your recording.")

        # Preprocess audio with timeout
        logger.info(f"Starting audio preprocessing for: {tmp_path}")
        start_time_proc = time.time()
        try:
            tensor = await asyncio.wait_for(
                asyncio.to_thread(preprocess_audio, tmp_path, clf),
                timeout=60.0
            )
            logger.info(f"Audio preprocessing completed in {(time.time() - start_time_proc)*1000:.2f}ms")
        except asyncio.TimeoutError:
            logger.error(f"Audio preprocessing timed out for: {tmp_path}")
            raise HTTPException(status_code=408, detail="Audio preprocessing timed out. Please try a shorter or simpler audio file.")

        # Model inference with timeout
        logger.info("Starting model inference")
        inference_start = time.time()
        try:
            def run_inference():
                with torch.no_grad():
                    outputs = clf["model"](tensor)
                    # Handle tuple output from model (intent_logits, slot_logits)
                    if isinstance(outputs, tuple):
                        intent_logits = outputs[0]
                    else:
                        intent_logits = outputs
                    probs = torch.softmax(intent_logits, dim=1)
                    conf, idx = probs.max(1)
                    return probs, conf, idx

            probs, conf, idx = await asyncio.wait_for(
                asyncio.to_thread(run_inference),
                timeout=30.0
            )
            logger.info(f"Model inference completed in {(time.time() - inference_start)*1000:.2f}ms")
        except asyncio.TimeoutError:
            logger.error("Model inference timed out")
            raise HTTPException(status_code=408, detail="Model inference timed out. Please try again.")

        # Get top-k predictions
        label_map = clf["label_map"]
        idx_to_label = {v: k for k, v in label_map.items()}

        k = min(top_k, probs.shape[1])
        top_vals, top_idxs = torch.topk(probs, k)

        top_predictions = []
        for score, pred_idx in zip(top_vals[0], top_idxs[0]):
            intent = idx_to_label.get(pred_idx.item(), f"cls_{pred_idx.item()}")
            top_predictions.append({
                "intent": intent,
                "confidence": float(score.item())
            })

        primary_intent = idx_to_label.get(idx.item(), f"cls_{idx.item()}")
        processing_time_ms = (time.time() - start_time_proc) * 1000
        total_time_ms = (time.time() - start_time_total) * 1000

        logger.info(f"Intent classification completed: {primary_intent} ({float(conf.item()):.3f}) in {total_time_ms:.2f}ms")

        return {
            "filename": file.filename,
            "intent": primary_intent,
            "confidence": float(conf.item()),
            "top_predictions": top_predictions,
            "model_type": model_type,
            "processing_time_ms": processing_time_ms,
            "total_time_ms": total_time_ms
        }

    except asyncio.TimeoutError:
        logger.error(f"Timeout occurred during audio processing for: {file.filename}")
        raise HTTPException(status_code=408, detail="Request timed out. Please try a shorter audio file or different format.")
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error in test_intent: {e}")
        logger.error(f"Full traceback: {error_details}")

        # Log additional debugging info
        logger.error(f"Original filename: {file.filename}")
        if tmp_path:
            logger.error(f"Temporary file: {tmp_path}")
            logger.error(f"Temp file exists: {os.path.exists(tmp_path) if tmp_path else 'N/A'}")
            if tmp_path and os.path.exists(tmp_path):
                logger.error(f"Temp file size: {os.path.getsize(tmp_path)} bytes")

        if "timed out" in str(e).lower() or "timeout" in str(e).lower():
            raise HTTPException(status_code=408, detail="Processing timed out. Please try a shorter audio file or different format.")
        elif "Format not recognised" in str(e) or "NoBackendError" in str(e):
            raise HTTPException(status_code=400, detail="Audio format not supported. Please use WAV, MP3, WebM, OGG, M4A, AAC, or FLAC format.")
        elif "NaN" in str(e) or "Input X contains NaN" in str(e) or "not finite everywhere" in str(e):
            raise HTTPException(status_code=400, detail="Audio processing failed. The audio may be corrupted, too short, or contains invalid data.")
        elif "Audio buffer is not finite everywhere" in str(e):
            raise HTTPException(status_code=400, detail="Audio contains invalid data (NaN or infinite values). Please check your recording.")
        elif "divide by zero" in str(e) or "division by zero" in str(e):
            raise HTTPException(status_code=400, detail="Audio processing failed due to silence or very low signal. Please record with more volume.")
        elif "Failed to convert audio" in str(e):
            raise HTTPException(status_code=400, detail="Failed to convert audio to supported format. Please try a different audio file or format.")
        else:
            raise HTTPException(status_code=500, detail=f"Error processing audio: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.debug(f"Cleaned up temporary file: {tmp_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary file {tmp_path}: {cleanup_error}")

@app.get("/reset-metrics")
async def reset_metrics():
    """Reset API metrics"""
    global request_count, start_time, processing_times

    request_count = 0
    start_time = time.time()
    processing_times = []

    return {"status": "success", "message": "Metrics reset"}
