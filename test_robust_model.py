#!/usr/bin/env python3
"""
Test script for the Robust Speech Model

This script loads a trained RobustModel and allows testing with live audio
or pre-recorded files.

Usage:
    python test_robust_model.py --model data/models/robust/robust_best_model.pt
    python test_robust_model.py --model data/models/robust/robust_best_model.pt --file test_audio.wav
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
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_robust_model import RobustModel
from src.preprocessing.audio_processor import AudioProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustModelTester:
    def __init__(self, model_path, label_map_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        model_dir = os.path.dirname(model_path)

        # Load label map
        if label_map_path is None:
            label_map_path = Path(model_dir) / '../../processed/label_map.json'

        self.label_map = self.load_label_map(label_map_path)
        self.num_classes = len(self.label_map)

        # Load model config to get input_dim
        config_path = Path(model_dir) / 'robust_training_results.json'
        input_dim = 39 # default
        if config_path.exists():
            with open(config_path, 'r') as f:
                results = json.load(f)
                # This part is tricky as input_dim is not saved. We have to assume it from training context.
                # The training scripts use features that result in 39 dimensions (13 MFCC + deltas)
                # Let's stick to a default unless specified.
                logger.info("Assuming input_dim=39 based on training scripts.")

        self.input_dim = input_dim
        logger.info(f"Model config: input_dim={self.input_dim}, num_classes={self.num_classes}")

        self.model = self.load_model(model_path)
        self.audio_processor = AudioProcessor()

    def load_label_map(self, label_map_path):
        """Load label mapping from file."""
        if not Path(label_map_path).exists():
            raise FileNotFoundError(f"Label map not found at {label_map_path}")

        with open(label_map_path, 'r') as f:
            label_map = json.load(f)

        self.idx_to_label = {v: k for k, v in label_map.items()}
        logger.info(f"Loaded {len(label_map)} classes")
        return label_map

    def load_model(self, model_path):
        """Load the trained RobustModel."""
        logger.info(f"Loading model from: {model_path}")
        model = RobustModel(input_dim=self.input_dim, num_classes=self.num_classes)

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.to(self.device)
            model.eval()
            logger.info("Model loaded successfully!")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def extract_features(self, audio_path):
        """Extract features from an audio file."""
        return self.audio_processor.preprocess(audio_path)

    def predict(self, features):
        """Make a prediction on the given features."""
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(features_tensor, apply_temperature=False)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        return predicted_idx.item(), confidence.item(), probabilities.cpu().numpy()[0]

    def test_file(self, file_path):
        """Test a single audio file."""
        print(f"\nTesting audio file: {file_path}")
        features = self.extract_features(file_path)
        predicted_class, confidence, probabilities = self.predict(features)
        predicted_intent = self.idx_to_label.get(predicted_class, "Unknown")

        print(f"\nPredicted Intent: {predicted_intent}")
        print(f"Confidence: {confidence:.4f}")
        self.show_top_predictions(probabilities)

    def record_and_test(self, duration=3):
        """Record audio from the microphone and test it."""
        print(f"\nRecording for {duration} seconds...")
        sample_rate = 16000
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        print("Recording finished.")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            self.test_file(tmp.name)

    def show_top_predictions(self, probabilities, top_k=5):
        """Display the top K predictions."""
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        print(f"\nTop {top_k} predictions:")
        for i, idx in enumerate(top_indices):
            intent = self.idx_to_label.get(idx, f"unknown_{idx}")
            confidence = probabilities[idx]
            print(f"  {i+1}. {intent}: {confidence:.4f} ({confidence*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Test a trained Robust Speech Model.")
    parser.add_argument('--model', required=True, help='Path to the trained model file (.pt)')
    parser.add_argument('--file', help='Path to an audio file to test. If not provided, will record live.')
    parser.add_argument('--duration', type=int, default=3, help='Duration for live recording in seconds.')
    parser.add_argument('--loop', action='store_true', help='Loop for continuous live testing.')

    args = parser.parse_args()

    try:
        tester = RobustModelTester(model_path=args.model)

        if args.file:
            if not os.path.exists(args.file):
                print(f"Error: File not found at {args.file}")
                return
            tester.test_file(args.file)
        else:
            if args.loop:
                while True:
                    input("Press Enter to start recording...")
                    tester.record_and_test(args.duration)
            else:
                tester.record_and_test(args.duration)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
