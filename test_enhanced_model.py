# test_enhanced_model.py
import os
import torch
import numpy as np
import argparse
import time
import sounddevice as sd
import soundfile as sf
import tempfile
import logging
from src.models.speech_model import EnhancedTwiSpeechModel
from src.preprocessing.enhanced_audio_processor import EnhancedAudioProcessor
from src.utils.model_utils import load_label_map, get_model_input_dim

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path, label_map_path, device):
    """Load the enhanced model and return it"""
    # Load label map
    label_map = load_label_map(label_map_path)
    num_classes = len(label_map)

    # Get input dimension from model files
    input_dim = get_model_input_dim(model_path)

    logger.info(f"Initializing model with input_dim={input_dim}, num_classes={num_classes}")

    # Initialize model
    model = EnhancedTwiSpeechModel(
        input_dim=input_dim,
        hidden_dim=128,
        num_classes=num_classes,
        dropout=0.3,
        num_heads=8
    )

    # Load model weights
    try:
        checkpoint = torch.load(model_path, map_location=device)

        # Handle various checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")

            # Extract expected input dimension from config if available
            if 'config' in checkpoint and 'input_dim' in checkpoint['config']:
                expected_input_dim = checkpoint['config']['input_dim']
                logger.info(f"Model expects input_dim={expected_input_dim} from checkpoint config")
                return model, label_map, expected_input_dim
        else:
            model.load_state_dict(checkpoint)
            logger.info("Loaded model weights")

        model = model.to(device)
        model.eval()
        return model, label_map, input_dim

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None, None

def record_audio(duration=3, sample_rate=16000):
    """Record audio from microphone"""
    print("Recording in 3...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        sd.sleep(1000)  # Sleep for 1 second

    print("Speak now!")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Recording complete!")

    # Normalize
    audio_data = audio_data.flatten()
    audio_data = audio_data / np.max(np.abs(audio_data)) if np.max(np.abs(audio_data)) > 0 else audio_data

    return audio_data, sample_rate

def preprocess_audio(audio_path, processor, expected_input_dim):
    """
    Process audio with dimension adjustment to match model requirements

    Args:
        audio_path: Path to audio file
        processor: Audio processor instance
        expected_input_dim: Expected input dimension for the model

    Returns:
        Preprocessed features as tensor
    """
    # Extract features using the enhanced processor
    features = processor.preprocess(audio_path)

    logger.info(f"Extracted features with shape: {features.shape}")

    # Adjust dimensions if needed
    if features.shape[0] != expected_input_dim:
        # logger.info(f"Adjusting feature dimensions from {features.shape[0]} to {expected_input_dim}")

        if features.shape[0] > expected_input_dim:
            # Truncate features if we have too many
            features = features[:expected_input_dim, :]
        else:
            # Pad features if we have too few
            padding = np.zeros((expected_input_dim - features.shape[0], features.shape[1]))
            features = np.vstack((features, padding))

        # logger.info(f"Adjusted features shape: {features.shape}")

    # Convert to tensor and add batch dimension
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    return features_tensor

def classify_audio(model, audio_tensor, label_map, device):
    """Classify audio features using the model"""
    # Move tensor to device
    audio_tensor = audio_tensor.to(device)

    # Get predictions
    with torch.no_grad():
        outputs = model(audio_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = probabilities.max(1)

    # Convert to intent
    idx_to_label = {idx: label for label, idx in label_map.items()}
    predicted_intent = idx_to_label[predicted_idx.item()]
    confidence_value = confidence.item()

    return predicted_intent, confidence_value

def main():
    parser = argparse.ArgumentParser(description="Test enhanced speech model with live audio")
    parser.add_argument("--model", type=str, default="data/models_improved/best_model.pt", help="Path to model file")
    parser.add_argument("--label-map", type=str, default="data/processed_augmented/label_map.npy", help="Path to label map")
    parser.add_argument("--duration", type=int, default=3, help="Recording duration in seconds")
    parser.add_argument("--save-audio", action="store_true", help="Save recorded audio")
    parser.add_argument("--input-dim", type=int, help="Force specific input dimension")

    args = parser.parse_args()

    # Check if model file exists
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        return

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize audio processor
    processor = EnhancedAudioProcessor(feature_type="combined")

    # Load model
    model, label_map, expected_input_dim = load_model(args.model, args.label_map, device)

    # Override input dimension if specified
    if args.input_dim:
        expected_input_dim = args.input_dim
        logger.info(f"Overriding input dimension to: {expected_input_dim}")

    if model is None:
        logger.error("Failed to load model. Exiting.")
        return

    logger.info("Model loaded successfully!")
    logger.info(f"Model expected input dimension: {expected_input_dim}")
    logger.info("Available intents:")
    for intent in label_map.keys():
        logger.info(f"- {intent}")

    while True:
        try:
            input("\nPress Enter to record audio (or Ctrl+C to exit)...")

            # Record audio
            audio_data, sample_rate = record_audio(duration=args.duration)

            # Save audio to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=not args.save_audio) as tmp:
                sf.write(tmp.name, audio_data, sample_rate)
                # logger.info(f"Audio saved temporarily to {tmp.name}")

                # Process audio with dimension adjustment
                features_tensor = preprocess_audio(tmp.name, processor, expected_input_dim)

                # Classify
                intent, confidence = classify_audio(model, features_tensor, label_map, device)

                logger.info(f"Predicted intent: {intent}")
                logger.info(f"Confidence: {confidence:.4f}")

                if args.save_audio:
                    save_path = f"test_recording_{int(time.time())}.wav"
                    os.rename(tmp.name, save_path)
                    logger.info(f"Audio saved to {save_path}")

        except KeyboardInterrupt:
            logger.info("Exiting...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
