#!/usr/bin/env python3
"""
Test script for the 47-class preserving speech model

This script loads the trained 47-class model and allows testing with live audio
or pre-recorded files. It uses the correct model architecture (ImprovedTwiSpeechModel)
and handles the 47-class output properly.

Usage:
    python test_47_class_model.py --model data/models/47_class_results/hybrid_20250827_074529/best_model.pt
    python test_47_class_model.py --model data/models/47_class_results/hybrid_20250827_074529/best_model.pt --file test_audio.wav
"""

import os
import sys
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

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.speech_model import ImprovedTwiSpeechModel
from src.preprocessing.audio_processor import AudioProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Model47ClassTester:
    def __init__(self, model_path, label_map_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Load model info from the training results
        model_dir = os.path.dirname(model_path)

        # Try to load model info from final_results.json
        results_path = os.path.join(model_dir, 'final_results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
                model_info = results.get('model_info', {})
                self.input_dim = model_info.get('input_dim', 39)
                self.num_classes = model_info.get('num_classes', 47)
        else:
            # Default values
            self.input_dim = 39
            self.num_classes = 47

        logger.info(f"Model config: input_dim={self.input_dim}, num_classes={self.num_classes}")

        # Load label map
        self.label_map = self.load_label_map(label_map_path or os.path.join(model_dir, '../../../processed/label_map.json'))

        # Create and load model
        self.model = self.load_model(model_path)

        # Audio processor
        self.audio_processor = AudioProcessor()

    def load_label_map(self, label_map_path):
        """Load label mapping from file or use default processed data"""
        possible_paths = [
            label_map_path,
            'data/processed/label_map.json',
            'data/processed_augmented/label_map.json'
        ]

        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Loading label map from: {path}")
                with open(path, 'r') as f:
                    label_map = json.load(f)
                break
        else:
            # Create a default label map if none found
            logger.warning("No label map found, creating default 47-class map")
            label_map = {f"class_{i}": i for i in range(self.num_classes)}

        # Create reverse mapping
        self.idx_to_label = {v: k for k, v in label_map.items()}
        logger.info(f"Loaded {len(label_map)} classes")

        return label_map

    def load_model(self, model_path):
        """Load the trained model"""
        logger.info(f"Loading model from: {model_path}")

        # Create model with correct architecture
        model = ImprovedTwiSpeechModel(
            input_dim=self.input_dim,
            hidden_dim=128,  # Default from training
            num_classes=self.num_classes,
            num_layers=2,
            dropout=0.3
        )

        # Load state dict
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Checkpoint format with multiple state dicts
                state_dict = checkpoint['model_state_dict']
            else:
                # Direct state dict format
                state_dict = checkpoint

            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            logger.info("Model loaded successfully!")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def extract_features_from_audio(self, audio_path):
        """Extract features from audio file"""
        try:
            # Use AudioProcessor to extract features like in training
            processor = AudioProcessor()
            features = processor.preprocess(audio_path)

            # Handle variable length by padding/truncating to reasonable size
            if features.shape[1] < 50:
                # Pad short sequences
                pad_width = 50 - features.shape[1]
                features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
            elif features.shape[1] > 200:
                # Truncate very long sequences
                features = features[:, :200]

            return features

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise

    def predict(self, features):
        """Make prediction on features"""
        try:
            # Convert to tensor and add batch dimension
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            return predicted_class, confidence, probabilities[0].cpu().numpy()

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def show_top_predictions(self, probabilities, top_k=5):
        """Show top K predictions with confidence scores"""
        top_indices = np.argsort(probabilities)[::-1][:top_k]

        print(f"\nTop {top_k} predictions:")
        for i, idx in enumerate(top_indices):
            intent = self.idx_to_label.get(idx, f"unknown_{idx}")
            confidence = probabilities[idx]
            print(f"  {i+1}. {intent}: {confidence:.4f} ({confidence*100:.1f}%)")

    def record_audio(self, duration=3, sample_rate=16000):
        """Record audio from microphone"""
        print(f"Recording for {duration} seconds... Speak now!")

        try:
            audio = sd.rec(int(duration * sample_rate),
                          samplerate=sample_rate,
                          channels=1,
                          dtype='float32')
            sd.wait()
            print("Recording finished!")

            return audio.flatten(), sample_rate

        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            raise

    def test_audio_file(self, audio_path):
        """Test prediction on audio file"""
        print(f"\nTesting audio file: {audio_path}")

        # Extract features
        features = self.extract_features_from_audio(audio_path)
        print(f"Extracted features shape: {features.shape}")

        # Make prediction
        predicted_class, confidence, probabilities = self.predict(features)
        predicted_intent = self.idx_to_label.get(predicted_class, f"unknown_{predicted_class}")

        print(f"\nPredicted intent: {predicted_intent}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.1f}%)")

        # Show top predictions
        self.show_top_predictions(probabilities)

        return predicted_intent, confidence

    def test_live_audio(self, duration=3):
        """Test prediction on live recorded audio"""
        print(f"\nRecording live audio for {duration} seconds...")

        # Record audio
        audio, sr = self.record_audio(duration)

        # Save temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio, sr)
            temp_path = tmp_file.name

        try:
            # Test the recorded audio
            return self.test_audio_file(temp_path)
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def main():
    parser = argparse.ArgumentParser(description="Test 47-class speech intent model")
    parser.add_argument("--model", type=str,
                       default="data/models/47_class_results/hybrid_20250827_074529/best_model.pt",
                       help="Path to model file")
    parser.add_argument("--label-map", type=str,
                       help="Path to label map JSON file")
    parser.add_argument("--file", type=str,
                       help="Audio file to test (if not provided, will record live)")
    parser.add_argument("--duration", type=int, default=3,
                       help="Recording duration in seconds for live audio")
    parser.add_argument("--loop", action="store_true",
                       help="Keep testing in a loop")

    args = parser.parse_args()

    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        print("Available models:")
        models_dir = "data/models/47_class_results"
        if os.path.exists(models_dir):
            for subdir in os.listdir(models_dir):
                model_path = os.path.join(models_dir, subdir, "best_model.pt")
                if os.path.exists(model_path):
                    print(f"  {model_path}")
        return

    try:
        # Initialize tester
        print("🔧 Initializing 47-class model tester...")
        tester = Model47ClassTester(args.model, args.label_map)
        print("✅ Model loaded successfully!")

        print(f"📊 Model info:")
        print(f"  - Input dimension: {tester.input_dim}")
        print(f"  - Number of classes: {tester.num_classes}")
        print(f"  - Device: {tester.device}")
        print(f"  - Classes loaded: {len(tester.label_map)}")

        if args.file:
            # Test specific file
            if not os.path.exists(args.file):
                print(f"❌ Audio file not found: {args.file}")
                return

            tester.test_audio_file(args.file)

        else:
            # Live audio testing
            print("\n🎤 Live Audio Testing Mode")
            print("Press Ctrl+C to exit")

            try:
                while True:
                    print("\n" + "="*50)
                    input("Press Enter to start recording (or Ctrl+C to exit)...")

                    tester.test_live_audio(args.duration)

                    if not args.loop:
                        break

                    print("\nTest again? (Enter to continue, Ctrl+C to exit)")

            except KeyboardInterrupt:
                print("\n👋 Exiting...")

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
