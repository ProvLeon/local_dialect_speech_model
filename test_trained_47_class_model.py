#!/usr/bin/env python3
"""
Test script for the 47-class preserving speech model (V2)

This script loads a trained 47-class model and allows testing with live audio
or pre-recorded files. It dynamically loads model architecture details
(e.g., hidden_dim, dropout) from training artifacts, making it robust for models
trained with different hyperparameters.

Usage:
    python test_trained_47_class_model.py --model <path_to_best_model.pt>
    python test_trained_47_class_model.py --model <path_to_best_model.pt> --file test_audio.wav
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

        # Load model architecture and info from the training results
        model_dir = os.path.dirname(model_path)
        self.load_model_config(model_dir)

        logger.info(f"Model config: input_dim={self.input_dim}, num_classes={self.num_classes}, "
                    f"hidden_dim={self.hidden_dim}, num_layers={self.num_layers}, dropout={self.dropout}, "
                    f"num_slot_classes={self.num_slot_classes}")

        # Load label map
        self.label_map = self.load_label_map(label_map_path or os.path.join(model_dir, '../../../processed/label_map.json'))

        # Create and load model
        self.model = self.load_model(model_path)

        # Audio processor
        self.audio_processor = AudioProcessor()

    def load_model_config(self, model_dir):
        """Load model configuration from training results file."""
        results_path = os.path.join(model_dir, 'final_results.json')

        # Default values from training script
        self.input_dim = 39
        self.num_classes = 47
        self.hidden_dim = 128
        self.num_layers = 2
        self.dropout = 0.3
        self.num_slot_classes = 0
        self.slot_value_maps = {}

        if os.path.exists(results_path):
            logger.info(f"Loading model configuration from {results_path}")
            with open(results_path, 'r') as f:
                results = json.load(f)
                model_info = results.get('model_info', {})

                self.input_dim = model_info.get('input_dim', self.input_dim)
                self.num_classes = model_info.get('num_classes', self.num_classes)
                self.hidden_dim = model_info.get('hidden_dim', self.hidden_dim)
                self.num_layers = model_info.get('num_layers', self.num_layers)
                self.dropout = model_info.get('dropout', self.dropout)
                self.num_slot_classes = model_info.get('num_slot_classes', self.num_slot_classes)
                self.slot_value_maps = model_info.get('slot_value_maps', self.slot_value_maps)
        else:
            logger.warning(f"Could not find {results_path}. Using default model parameters.")

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
        loaded_classes = len(label_map)
        logger.info(f"Loaded {loaded_classes} classes")

        # If the loaded label map size doesn't match the configured num_classes, adjust dynamically
        if loaded_classes != self.num_classes:
            logger.warning(
                f"num_classes mismatch: config/model_info has {self.num_classes}, "
                f"label_map has {loaded_classes}. Updating num_classes to {loaded_classes}."
            )
            self.num_classes = loaded_classes

        return label_map

    def load_model(self, model_path):
        """Load the trained model"""
        logger.info(f"Loading model from: {model_path}")

        # Create model with dynamically loaded architecture
        model = ImprovedTwiSpeechModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            dropout=self.dropout,
            num_slot_classes=self.num_slot_classes,
            slot_value_maps=self.slot_value_maps
        )

        # Load state dict
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # Defensive loading: handle classifier dimension mismatch gracefully
            classifier_weight_key = 'intent_classifier.weight'
            classifier_bias_key = 'intent_classifier.bias'
            if (classifier_weight_key in state_dict and
                state_dict[classifier_weight_key].shape[0] != self.num_classes):
                ckpt_classes = state_dict[classifier_weight_key].shape[0]
                logger.warning(
                    f"Classifier size mismatch (checkpoint={ckpt_classes}, current={self.num_classes}). "
                    "Removing incompatible classifier weights and reinitializing."
                )
                state_dict.pop(classifier_weight_key, None)
                state_dict.pop(classifier_bias_key, None)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded with missing keys (reinitialized): {missing}; unexpected keys: {unexpected}")
            else:
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing or unexpected:
                    logger.info(f"Loaded with missing keys: {missing}; unexpected keys: {unexpected}")
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
                pad_width = 50 - features.shape[1]
                features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
            elif features.shape[1] > 200:
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

                # Handle tuple output from model (intent_logits, slot_logits)
                if isinstance(outputs, tuple):
                    intent_logits = outputs[0]
                else:
                    intent_logits = outputs

                probabilities = torch.softmax(intent_logits, dim=1)
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

        features = self.extract_features_from_audio(audio_path)
        print(f"Extracted features shape: {features.shape}")

        predicted_class, confidence, probabilities = self.predict(features)
        predicted_intent = self.idx_to_label.get(predicted_class, f"unknown_{predicted_class}")

        print(f"\nPredicted intent: {predicted_intent}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.1f}%)")

        self.show_top_predictions(probabilities)
        return predicted_intent, confidence

    def test_live_audio(self, duration=3):
        """Test prediction on live recorded audio"""
        print(f"\nRecording live audio for {duration} seconds...")

        audio, sr = self.record_audio(duration)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio, sr)
            temp_path = tmp_file.name

        try:
            return self.test_audio_file(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def main():
    parser = argparse.ArgumentParser(description="Test a trained 47-class speech intent model")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model file (e.g., best_model.pt)")
    parser.add_argument("--label-map", type=str,
                       help="Path to label map JSON file (optional)")
    parser.add_argument("--file", type=str,
                       help="Audio file to test (if not provided, will record live)")
    parser.add_argument("--duration", type=int, default=3,
                       help="Recording duration in seconds for live audio")
    parser.add_argument("--loop", action="store_true",
                       help="Keep testing in a loop for live audio mode")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        sys.exit(1)

    try:
        print("🔧 Initializing model tester...")
        tester = Model47ClassTester(args.model, args.label_map)
        print("✅ Model loaded successfully!")

        print(f"📊 Model Info:")
        print(f"  - Input dimension: {tester.input_dim}")
        print(f"  - Hidden dimension: {tester.hidden_dim}")
        print(f"  - Number of layers: {tester.num_layers}")
        print(f"  - Dropout: {tester.dropout}")
        print(f"  - Number of classes: {tester.num_classes}")
        print(f"  - Number of slot classes: {tester.num_slot_classes}")
        print(f"  - Device: {tester.device}")
        print(f"  - Classes loaded: {len(tester.label_map)}")

        if args.file:
            if not os.path.exists(args.file):
                print(f"❌ Audio file not found: {args.file}")
                return
            tester.test_audio_file(args.file)
        else:
            print("\n🎤 Live Audio Testing Mode")
            print("Press Ctrl+C to exit")
            try:
                while True:
                    print("\n" + "="*50)
                    input("Press Enter to start recording (or Ctrl+C to exit)...")
                    tester.test_live_audio(args.duration)
                    if not args.loop:
                        break
            except KeyboardInterrupt:
                print("\n👋 Exiting...")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
