#!/usr/bin/env python3
"""
Test script for the optimized speech model.
Handles consistent feature extraction and works with the simplified model architecture.
"""

import os
import torch
import numpy as np
import argparse
import time
import sounddevice as sd
import soundfile as sf
import tempfile
import logging
import librosa
import json
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplifiedSpeechModel(torch.nn.Module):
    """
    Simplified speech model optimized for small datasets with class imbalance.
    Must match the architecture in train_optimized_model.py
    """
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super(SimplifiedSpeechModel, self).__init__()

        self.input_dim = input_dim

        # Feature extraction layers with moderate complexity
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),

            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),

            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout/2)
        )

        # Temporal modeling with GRU (lighter than LSTM)
        self.gru = torch.nn.GRU(
            input_size=64,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            dropout=0.0,  # Only 1 layer, no dropout
            bidirectional=True
        )

        # Attention mechanism (simplified)
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1)
        )

        # Classification head
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout/2),
            torch.nn.Linear(32, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size, seq_len, features = x.shape

        # Reshape for feature extraction: (B*T, F)
        x_reshaped = x.view(-1, features)

        # Extract features
        features = self.feature_extractor(x_reshaped)

        # Reshape back: (B, T, F)
        features = features.view(batch_size, seq_len, -1)

        # GRU processing
        gru_out, _ = self.gru(features)

        # Attention-based pooling
        attn_weights = self.attention(gru_out)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=1)

        # Weighted sum
        pooled = torch.sum(gru_out * attn_weights, dim=1)

        # Classification
        output = self.classifier(pooled)

        return output

def load_model(model_path, device):
    """Load the optimized model and return it with metadata"""
    try:
        checkpoint = torch.load(model_path, map_location=device)

        # Extract configuration
        config = checkpoint.get('config', {})
        label_map = checkpoint.get('label_map', {})

        input_dim = config.get('input_dim', 117)
        num_classes = config.get('num_classes', len(label_map))
        dropout = config.get('dropout', 0.3)

        logger.info(f"Model config: input_dim={input_dim}, num_classes={num_classes}, dropout={dropout}")

        # Initialize model
        model = SimplifiedSpeechModel(
            input_dim=input_dim,
            num_classes=num_classes,
            dropout=dropout
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        logger.info(f"Loaded model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
        logger.info(f"Best validation F1: {checkpoint.get('best_val_f1', 'unknown')}")

        return model, label_map, input_dim

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None, None

def load_label_map(model_dir):
    """Load label map from model directory"""
    label_map_path = os.path.join(model_dir, 'label_map.json')

    if os.path.exists(label_map_path):
        with open(label_map_path, 'r') as f:
            return json.load(f)

    logger.warning(f"Label map not found at {label_map_path}")
    return {}

def extract_features_from_audio(audio_path, target_dim=117):
    """
    Extract consistent features from audio file to match training pipeline.
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)

        # Extract MFCC features (39 dimensions: 13 + 13 delta + 13 delta-delta)
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=13,
            n_fft=512,
            hop_length=160,
            n_mels=40
        )

        # Apply delta enrichment to get 117 dimensions (39 * 3)
        if target_dim == 117:
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            features = np.concatenate([mfcc, delta, delta2], axis=0)
        else:
            features = mfcc

        # Adjust dimensions if needed
        current_dim = features.shape[0]
        if current_dim != target_dim:
            if current_dim > target_dim:
                features = features[:target_dim, :]
            else:
                padding = np.zeros((target_dim - current_dim, features.shape[1]))
                features = np.vstack((features, padding))

        logger.info(f"Extracted features with shape: {features.shape}")

        # Transpose to (time, features) for model input
        features = features.T

        # Convert to tensor and add batch dimension
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        return features_tensor

    except Exception as e:
        logger.error(f"Error extracting features from {audio_path}: {e}")
        return None

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
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))

    return audio_data, sample_rate

def classify_audio(model, audio_tensor, label_map, device):
    """Classify audio features using the model"""
    try:
        # Move tensor to device
        audio_tensor = audio_tensor.to(device)

        # Get predictions
        with torch.no_grad():
            outputs = model(audio_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = probabilities.max(1)

        # Convert to intent
        idx_to_label = {idx: label for label, idx in label_map.items()}
        predicted_intent = idx_to_label.get(predicted_idx.item(), "unknown")
        confidence_value = confidence.item()

        return predicted_intent, confidence_value, probabilities.cpu().numpy()[0]

    except Exception as e:
        logger.error(f"Error during classification: {e}")
        return "error", 0.0, None

def show_top_predictions(probabilities, label_map, top_k=5):
    """Show top K predictions with confidence scores"""
    if probabilities is None:
        return

    idx_to_label = {idx: label for label, idx in label_map.items()}

    # Get top predictions
    top_indices = np.argsort(probabilities)[::-1][:top_k]

    print(f"\nTop {top_k} predictions:")
    for i, idx in enumerate(top_indices):
        label = idx_to_label.get(idx, f"class_{idx}")
        confidence = probabilities[idx]
        print(f"  {i+1}. {label}: {confidence:.4f} ({confidence*100:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description="Test optimized speech model with live audio")
    parser.add_argument("--model-dir", type=str, default="data/models_optimized",
                       help="Directory containing the trained model")
    parser.add_argument("--model-file", type=str, default="best_model.pt",
                       help="Model filename")
    parser.add_argument("--duration", type=int, default=3,
                       help="Recording duration in seconds")
    parser.add_argument("--save-audio", action="store_true",
                       help="Save recorded audio")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Show top K predictions")

    args = parser.parse_args()

    # Check if model directory exists
    if not os.path.exists(args.model_dir):
        logger.error(f"Model directory not found: {args.model_dir}")
        return

    model_path = os.path.join(args.model_dir, args.model_file)
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    logger.info("Loading model...")
    model, label_map, expected_input_dim = load_model(model_path, device)

    if model is None:
        logger.error("Failed to load model. Exiting.")
        return

    # Load label map if not included in checkpoint
    if not label_map:
        label_map = load_label_map(args.model_dir)
        if not label_map:
            logger.error("Could not load label map. Exiting.")
            return

    logger.info("Model loaded successfully!")
    logger.info(f"Expected input dimension: {expected_input_dim}")
    logger.info(f"Number of classes: {len(label_map)}")

    print("\nAvailable intents:")
    for intent in sorted(label_map.keys()):
        print(f"  - {intent}")

    print(f"\nModel ready! Press Enter to start recording {args.duration}-second audio clips...")

    while True:
        try:
            input("\nPress Enter to record audio (or Ctrl+C to exit)...")

            # Record audio
            audio_data, sample_rate = record_audio(duration=args.duration)

            # Save audio to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=not args.save_audio) as tmp:
                sf.write(tmp.name, audio_data, sample_rate)

                # Extract features
                features_tensor = extract_features_from_audio(tmp.name, expected_input_dim)

                if features_tensor is None:
                    logger.error("Failed to extract features")
                    continue

                # Classify
                intent, confidence, probabilities = classify_audio(
                    model, features_tensor, label_map, device
                )

                print(f"\n🎯 Predicted intent: {intent}")
                print(f"📊 Confidence: {confidence:.4f} ({confidence*100:.2f}%)")

                # Show top predictions
                show_top_predictions(probabilities, label_map, args.top_k)

                # Save audio if requested
                if args.save_audio:
                    save_path = f"test_recording_{int(time.time())}.wav"
                    os.rename(tmp.name, save_path)
                    print(f"💾 Audio saved to: {save_path}")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
