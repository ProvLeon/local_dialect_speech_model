# src/api/speech_api.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import tempfile
import os
import soundfile as sf
import numpy as np
import json
import time  # Add this import for timestamp generation
import logging
from typing import Optional, Dict, Any, List  # Add List for proper typing
from pydantic import BaseModel

# Import model components
from src.models.speech_model import IntentClassifier, EnhancedTwiSpeechModel
from src.preprocessing.enhanced_audio_processor import EnhancedAudioProcessor
from src.utils.ecommerce_integration import EcommerceIntegration
from src.utils.model_utils import load_label_map, get_model_input_dim
from config.model_config import MODEL_CONFIG
from config.api_config import API_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Akan (Twi) Speech-to-Action API",
    description="API for recognizing Twi speech commands and translating them to actions",
    version="1.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CONFIG.get("allowed_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define path constants with environment variable overrides
ENHANCED_MODEL_PATH = os.environ.get("ENHANCED_MODEL_PATH", "data/models_improved/best_model.pt")
STANDARD_MODEL_PATH = os.environ.get("MODEL_PATH", "data/models_improved/best_model.pt")
LABEL_MAP_PATH = os.environ.get("LABEL_MAP_PATH", "data/processed_augmented/label_map.npy")

# Response models for better documentation
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    available_intents: Optional[int] = None
    api_version: str = "1.1.0"

class RecognitionResponse(BaseModel):
    filename: str
    intent: str
    confidence: float
    model_type: str
    processing_time_ms: float

class ActionResponse(BaseModel):
    recognition: Dict[str, Any]
    action: Dict[str, Any]

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Model state
classifier = None
enhanced_classifier = None
confidence_threshold = API_CONFIG.get('confidence_threshold', 0.7)

# Stats tracking
request_count = 0
start_time = time.time()
processing_times = []

# Initialize e-commerce integration
ecommerce = EcommerceIntegration()

def get_enhanced_processor():
    """Get enhanced audio processor with optimized settings"""
    return EnhancedAudioProcessor(
        feature_type="combined",
        augment=False,  # No augmentation for inference
        sample_rate=MODEL_CONFIG.get('sample_rate', 16000),
        n_mfcc=MODEL_CONFIG.get('n_mfcc', 13),
        n_mels=MODEL_CONFIG.get('n_mels', 40)
    )

def load_enhanced_model():
    """Load enhanced model for speech recognition"""
    global enhanced_classifier

    try:
        if os.path.exists(ENHANCED_MODEL_PATH):
            # Load label map and get input dimension
            label_map = load_label_map(LABEL_MAP_PATH)
            input_dim = get_model_input_dim(ENHANCED_MODEL_PATH)

            # Get number of classes
            num_classes = len(label_map)
            logger.info(f"Loading enhanced model with input_dim={input_dim}, num_classes={num_classes}")

            # Initialize the model
            model = EnhancedTwiSpeechModel(
                input_dim=input_dim,
                hidden_dim=128,
                num_classes=num_classes,
                dropout=0.3,
                num_heads=8
            )

            # Load weights
            checkpoint = torch.load(ENHANCED_MODEL_PATH, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded enhanced model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")

                # Extract checkpoint info for metrics
                checkpoint_info = {}
                if 'config' in checkpoint:
                    checkpoint_info = checkpoint['config']
                if 'final_metrics' in checkpoint:
                    checkpoint_info['final_metrics'] = checkpoint['final_metrics']
            else:
                model.load_state_dict(checkpoint)
                logger.info("Loaded enhanced model weights")
                checkpoint_info = {}

            # Set to evaluation mode
            model.to(device)
            model.eval()

            # Create custom classifier with the loaded model and processor
            enhanced_classifier = {
                'model': model,
                'processor': get_enhanced_processor(),
                'label_map': label_map,
                'input_dim': input_dim,
                'model_type': 'enhanced',
                'checkpoint_info': checkpoint_info
            }

            logger.info("Enhanced model loaded successfully!")
            return True
        else:
            logger.warning(f"Enhanced model not found at {ENHANCED_MODEL_PATH}")
            return False
    except Exception as e:
        logger.error(f"Error loading enhanced model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def load_standard_model():
    """Load standard IntentClassifier model"""
    global classifier

    try:
        # Use standard model
        processor = get_enhanced_processor()
        classifier = IntentClassifier(
            STANDARD_MODEL_PATH,
            device,
            processor=processor,
            label_map_path=LABEL_MAP_PATH
        )
        logger.info("Standard model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading standard model: {e}")
        classifier = None
        return False

def preprocess_audio(audio_path, classifier_obj):
    """Process audio with dimension adjustment to match model requirements"""
    if isinstance(classifier_obj, dict):  # Enhanced model
        # Get processor and expected input dimension
        processor = classifier_obj['processor']
        expected_input_dim = classifier_obj['input_dim']

        # Extract features using the enhanced processor
        features = processor.preprocess(audio_path)

        # Adjust dimensions if needed
        if features.shape[0] != expected_input_dim:
            if features.shape[0] > expected_input_dim:
                # Truncate features if we have too many
                features = features[:expected_input_dim, :]
            else:
                # Pad features if we have too few
                padding = np.zeros((expected_input_dim - features.shape[0], features.shape[1]))
                features = np.vstack((features, padding))

        # Convert to tensor and add batch dimension
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        return features_tensor
    else:
        # For standard classifier, use its internal processing
        return None

def classify_with_enhanced_model(audio_path, classifier_obj):
    """Classify audio using enhanced model"""
    try:
        # Preprocess audio
        features_tensor = preprocess_audio(audio_path, classifier_obj)

        # Move tensor to device
        features_tensor = features_tensor.to(device)

        # Get model and label map
        model = classifier_obj['model']
        label_map = classifier_obj['label_map']

        # Get predictions
        with torch.no_grad():
            outputs = model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = probabilities.max(1)

        # Convert to intent
        idx_to_label = {idx: label for label, idx in label_map.items()}
        predicted_intent = idx_to_label[predicted_idx.item()]
        confidence_value = confidence.item()

        return predicted_intent, confidence_value
    except Exception as e:
        logger.error(f"Error classifying with enhanced model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def get_active_classifier():
    """Get the currently active classifier, with preference for enhanced"""
    if enhanced_classifier:
        return enhanced_classifier, 'enhanced'
    elif classifier:
        return classifier, 'standard'
    else:
        return None, None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global start_time
    start_time = time.time()

    # Try to load enhanced model first
    enhanced_loaded = load_enhanced_model()

    # If enhanced model failed, load standard model
    if not enhanced_loaded:
        load_standard_model()

    # Check if any model was loaded
    classifier_obj, model_type = get_active_classifier()
    if classifier_obj:
        logger.info(f"API started with {model_type} model")
    else:
        logger.error("No models could be loaded. API will return errors for recognition requests.")

@app.get("/", response_model=dict)
async def root():
    """API root endpoint"""
    return {
        "message": "Akan (Twi) Speech-to-Action API is running",
        "version": "1.1.0",
        "model_type": "enhanced" if enhanced_classifier else "standard" if classifier else "none"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with detailed status"""
    classifier_obj, model_type = get_active_classifier()

    if not classifier_obj:
        return JSONResponse(
            status_code=503,
            content=HealthResponse(
                status="error",
                model_loaded=False,
                model_type="none"
            ).dict()
        )

    # Get number of available intents
    num_intents = None
    if model_type == 'enhanced' and 'label_map' in classifier_obj:
        num_intents = len(classifier_obj['label_map'])
    elif model_type == 'standard' and hasattr(classifier_obj, 'label_map'):
        num_intents = len(classifier_obj.label_map)

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_type=model_type,
        available_intents=num_intents
    )

@app.post("/recognize", response_model=RecognitionResponse)
async def recognize_speech(file: UploadFile = File(...)):
    """
    Recognize speech intent from audio file using the best available model

    Args:
        file: Audio file (WAV, MP3)

    Returns:
        Recognized intent and confidence score
    """
    global request_count, processing_times
    request_count += 1
    classifier_obj, model_type = get_active_classifier()

    if not classifier_obj:
        raise HTTPException(status_code=503, detail="No speech recognition model is loaded")

    # Check file type
    if file.filename is None or not file.filename.lower().endswith(('.wav', '.mp3')):
        raise HTTPException(status_code=400, detail="Only WAV and MP3 files are supported")

    # Handle file size limit
    content = await file.read()
    max_size = API_CONFIG.get("max_upload_size", 10 * 1024 * 1024)  # Default 10MB
    if len(content) > max_size:
        raise HTTPException(status_code=413, detail=f"File size exceeds the {max_size/1024/1024}MB limit")

    try:
        # Create temporary file
        suffix = os.path.splitext(file.filename or "audio.wav")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            # Write uploaded file to temporary file
            tmp.write(content)
            tmp_path = tmp.name

        start_time_proc = time.time()

        # Process audio and predict intent based on model type
        if model_type == 'enhanced':
            intent, confidence = classify_with_enhanced_model(tmp_path, classifier_obj)
        else:
            # Use standard classifier
            intent, confidence = classifier_obj.classify(tmp_path)

        processing_time = (time.time() - start_time_proc) * 1000  # ms
        processing_times.append(processing_time)

        # Clean up temporary file
        os.unlink(tmp_path)

        return RecognitionResponse(
            filename=file.filename or "audio.wav",
            intent=intent,
            confidence=confidence,
            model_type=model_type,
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/action", response_model=ActionResponse)
async def take_action(
    file: UploadFile = File(...),
    user_id: str = Query(None, description="User ID for e-commerce actions")
):
    """
    Recognize speech and take appropriate e-commerce action

    Args:
        file: Audio file
        user_id: User ID for e-commerce integration

    Returns:
        Action taken based on recognized intent
    """
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    # First recognize the speech
    recognition_result = await recognize_speech(file)

    # Check confidence threshold
    if recognition_result.confidence < confidence_threshold:
        return ActionResponse(
            recognition=recognition_result.dict(),
            action={
                "status": "low_confidence",
                "message": f"Intent recognition confidence below threshold ({confidence_threshold}). Please try again.",
                "confidence": recognition_result.confidence
            }
        )

    # Take action based on the intent
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
        raise HTTPException(status_code=500, detail=f"Error executing action: {str(e)}")

@app.get("/intents")
async def list_intents():
    """List all available intents with descriptions"""
    classifier_obj, model_type = get_active_classifier()

    if not classifier_obj:
        raise HTTPException(status_code=503, detail="No speech recognition model is loaded")

    # Extract intents based on model type
    if model_type == 'enhanced':
        if 'label_map' not in classifier_obj:
            raise HTTPException(status_code=503, detail="Label map not available in enhanced model")
        intents = list(classifier_obj['label_map'].keys())
    else:
        if not hasattr(classifier_obj, 'label_map') or classifier_obj.label_map is None:
            raise HTTPException(status_code=503, detail="Label map not available in standard model")
        intents = list(classifier_obj.label_map.keys())

    # Add intent descriptions (this could be loaded from a separate file)
    intent_descriptions = {
        "purchase": "Purchase an item directly",
        "add_to_cart": "Add an item to the shopping cart",
        "search": "Search for items or products",
        "remove_from_cart": "Remove an item from the cart",
        "checkout": "Proceed to checkout",
        "intent_to_buy": "Express intention to purchase",
        "continue": "Continue with the current flow",
        "go_back": "Return to the previous step",
        "show_items": "Display available items",
        "show_cart": "Show items in the cart",
        "confirm_order": "Confirm the current order",
        "make_payment": "Process payment for an order",
        "ask_questions": "Ask questions about products",
        "help": "Request help or customer support",
        "cancel": "Cancel the current action",
        "show_price_images": "Show prices and images of items",
        "change_quantity": "Change quantity of an item",
        "show_categories": "Display product categories",
        "show_description": "Show description of an item",
        "save_for_later": "Save an item for later purchase"
    }

    # Format response with descriptions when available
    result = []
    for intent in intents:
        result.append({
            "intent": intent,
            "description": intent_descriptions.get(intent, "No description available")
        })

    return {
        "intents": result,
        "count": len(result),
        "model_type": model_type
    }


@app.get("/model-info")
async def model_info():
    """Get detailed information about the active model"""
    classifier_obj, model_type = get_active_classifier()

    if not classifier_obj:
        raise HTTPException(status_code=503, detail="No speech recognition model is loaded")

    if model_type == 'enhanced':
        # Extract model information from enhanced classifier
        info = {
            "model_type": "enhanced",
            "input_dim": classifier_obj.get('input_dim', 'unknown'),
            "num_classes": len(classifier_obj.get('label_map', {})),
            "processor_type": classifier_obj.get('processor').__class__.__name__,
            "feature_type": getattr(classifier_obj.get('processor'), 'feature_type', 'unknown')
        }

        # Add training information if available
        if 'checkpoint_info' in classifier_obj and classifier_obj['checkpoint_info']:
            checkpoint = classifier_obj['checkpoint_info']
            info["training"] = {
                "hidden_dim": checkpoint.get('hidden_dim', 128),
                "dropout": checkpoint.get('dropout', 0.3),
                "num_heads": checkpoint.get('num_heads', 8)
            }

            # Add metrics if available
            if 'final_metrics' in classifier_obj['checkpoint_info']:
                metrics = classifier_obj['checkpoint_info']['final_metrics']
                info["metrics"] = {
                    "val_accuracy": metrics.get('val_acc', 'unknown'),
                    "test_accuracy": metrics.get('test_acc', 'unknown')
                }
    else:
        # Standard model info
        info = {
            "model_type": "standard",
            "processor_type": classifier_obj.processor.__class__.__name__
        }

        if hasattr(classifier_obj, 'model') and classifier_obj.model:
            model = classifier_obj.model
            info["input_dim"] = getattr(model, 'input_dim', 'unknown')

        if hasattr(classifier_obj, 'label_map') and classifier_obj.label_map:
            info["num_classes"] = len(classifier_obj.label_map)

    return info

@app.get("/switch-model/{model_type}")
async def switch_model(model_type: str):
    """
    Switch between available models (enhanced or standard)

    Args:
        model_type: Type of model to use - 'enhanced' or 'standard'

    Returns:
        Status of the model switch
    """
    global enhanced_classifier, classifier

    if model_type not in ['enhanced', 'standard']:
        raise HTTPException(status_code=400, detail="Model type must be 'enhanced' or 'standard'")

    if model_type == 'enhanced':
        # Try to load enhanced model if not already loaded
        if not enhanced_classifier:
            success = load_enhanced_model()
            if not success:
                raise HTTPException(
                    status_code=503,
                    detail="Enhanced model couldn't be loaded. Check logs for details."
                )
        return {"status": "success", "active_model": "enhanced"}
    else:
        # Try to load standard model if not already loaded
        if not classifier:
            success = load_standard_model()
            if not success:
                raise HTTPException(
                    status_code=503,
                    detail="Standard model couldn't be loaded. Check logs for details."
                )
        return {"status": "success", "active_model": "standard"}

@app.post("/process-batch")
async def process_batch(files: List[UploadFile] = File(...)):
    """
    Process multiple audio files in a batch

    Args:
        files: List of audio files

    Returns:
        Recognition results for each file
    """
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum batch size is 10 files")

    results = []
    for file in files:
        try:
            result = await recognize_speech(file)
            results.append(result.dict())
        except HTTPException as e:
            # Include error information in results
            results.append({
                "filename": file.filename,
                "error": e.detail,
                "status_code": e.status_code
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "status_code": 500
            })

    return {"batch_results": results, "successful": len([r for r in results if "error" not in r])}

@app.get("/metrics")
async def get_metrics():
    """Get model metrics and statistics"""
    global request_count, start_time, processing_times

    classifier_obj, model_type = get_active_classifier()

    if not classifier_obj:
        raise HTTPException(status_code=503, detail="No speech recognition model is loaded")

    # Calculate runtime metrics
    uptime_seconds = int(time.time() - start_time)
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0

    # Basic metrics that apply to both model types
    metrics = {
        "model_type": model_type,
        "uptime_seconds": uptime_seconds,
        "total_requests": request_count,
        "requests_per_minute": round(request_count / (uptime_seconds / 60), 2) if uptime_seconds > 0 else 0,
        "avg_processing_time_ms": round(avg_processing_time, 2)
    }

    # Add model-specific metrics
    if model_type == 'enhanced' and 'checkpoint_info' in classifier_obj:
        checkpoint = classifier_obj['checkpoint_info']
        if 'final_metrics' in checkpoint:
            final_metrics = checkpoint['final_metrics']
            metrics.update({
                "validation_accuracy": final_metrics.get('val_acc', 'unknown'),
                "test_accuracy": final_metrics.get('test_acc', 'unknown'),
                "training_loss": final_metrics.get('train_loss', 'unknown')
            })

    return metrics

@app.post("/feedback")
async def submit_feedback(
    intent: str,
    correct_intent: str,
    audio_file: Optional[UploadFile] = File(None),
    confidence: float = Query(None),
    notes: Optional[str] = Query(None)
):
    """
    Submit feedback for incorrect predictions to improve the model

    Args:
        intent: The intent that was predicted
        correct_intent: The correct intent that should have been predicted
        audio_file: Optional audio file for retraining
        confidence: Confidence score of the prediction
        notes: Additional notes about the feedback

    Returns:
        Status of the feedback submission
    """
    try:
        # In a production system, you would store this feedback in a database
        # for later analysis and model improvement

        # For now, just log it
        logger.info(
            f"Feedback received: Predicted '{intent}' → Correct '{correct_intent}' "
            f"(Confidence: {confidence})"
        )

        if audio_file:
            # Save the audio file for later retraining
            audio_dir = "data/feedback"
            os.makedirs(audio_dir, exist_ok=True)

            timestamp = int(time.time())
            file_path = os.path.join(
                audio_dir,
                f"feedback_{timestamp}_{correct_intent}.wav"
            )

            content = await audio_file.read()
            with open(file_path, "wb") as f:
                f.write(content)

            logger.info(f"Saved feedback audio to {file_path}")

        # Save detailed feedback to a log file
        feedback_log = os.path.join("data/feedback", "feedback_log.jsonl")

        with open(feedback_log, "a") as f:
            feedback_entry = {
                "timestamp": time.time(),
                "predicted_intent": intent,
                "correct_intent": correct_intent,
                "confidence": confidence,
                "notes": notes,
                "has_audio": audio_file is not None
            }
            f.write(json.dumps(feedback_entry) + "\n")

        return {"status": "success", "message": "Feedback received and logged"}

    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing feedback: {str(e)}")

@app.get("/reset-metrics")
async def reset_metrics():
    """Reset runtime metrics counters"""
    global request_count, start_time, processing_times

    request_count = 0
    start_time = time.time()
    processing_times = []

    return {"status": "success", "message": "Metrics reset successfully"}

@app.post("/debug")
async def debug_audio(file: UploadFile = File(...)):
    """
    Debug endpoint for audio processing - returns intermediate processing steps

    Args:
        file: Audio file to debug

    Returns:
        Detailed processing information
    """
    classifier_obj, model_type = get_active_classifier()

    if not classifier_obj:
        raise HTTPException(status_code=503, detail="No speech recognition model is loaded")

    try:
        # Save the uploaded file
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        debug_info = {
            "model_type": model_type,
            "audio_info": {}
        }

        # Extract audio information
        try:
            audio, sr = sf.read(tmp_path)
            debug_info["audio_info"] = {
                "sample_rate": sr,
                "duration": len(audio) / sr,
                "channels": audio.shape[1] if len(audio.shape) > 1 else 1,
                "min": float(np.min(audio)),
                "max": float(np.max(audio)),
                "rms": float(np.sqrt(np.mean(audio**2)))
            }
        except Exception as e:
            debug_info["audio_info"]["error"] = str(e)

        # Process features
        if model_type == 'enhanced':
            processor = classifier_obj['processor']
            try:
                features = processor.preprocess(tmp_path)
                debug_info["features"] = {
                    "shape": list(features.shape),
                    "feature_type": processor.feature_type,
                    "n_mfcc": processor.n_mfcc,
                    "n_mels": processor.n_mels
                }

                # Adjust dimensions if needed
                input_dim = classifier_obj['input_dim']
                if features.shape[0] != input_dim:
                    debug_info["dimension_mismatch"] = {
                        "expected": input_dim,
                        "actual": features.shape[0],
                        "action": "adjust_dimensions"
                    }

                # Run inference - don't save the result, just for debugging
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                features_tensor = features_tensor.to(device)
                _ = classifier_obj['model'](features_tensor)
                debug_info["inference"] = "successful"

            except Exception as e:
                debug_info["feature_extraction_error"] = str(e)
                import traceback
                debug_info["traceback"] = traceback.format_exc()

        # Clean up temp file
        os.unlink(tmp_path)

        return debug_info

    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")
