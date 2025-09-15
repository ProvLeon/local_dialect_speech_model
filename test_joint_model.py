#!/usr/bin/env python3
"""
Test script for the jointly trained intent and slot model.

This script loads the multi-head model and tests it with live or pre-recorded audio,
showing the predicted intent, slot types, and slot values.
"""

import os
import sys
import torch
import numpy as np
import argparse
import sounddevice as sd
import soundfile as sf
import tempfile
import logging
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__name__), 'src'))

from src.models.speech_model import ImprovedTwiSpeechModel
from src.preprocessing.audio_processor import AudioProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JointModelTester:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model_dir = os.path.dirname(model_path)
        self.data_dir = os.path.abspath(os.path.join(self.model_dir, '../../processed'))

        # Load all necessary maps
        self.label_map, self.slot_map, self.slot_value_maps = self.load_maps()
        self.idx_to_label = {v: k for k, v in self.label_map.items()}
        self.idx_to_slot = {v: k for k, v in self.slot_map.items()}
        self.idx_to_slot_value = {st: {v: k for k, v in sm.items()} for st, sm in self.slot_value_maps.items()}

        self.model = self.load_model()
        self.audio_processor = AudioProcessor()

    def load_maps(self):
        label_map = self._load_json(os.path.join(self.data_dir, 'label_map.json'))
        slot_map = self._load_json(os.path.join(self.data_dir, 'slot_map.json'))
        slot_value_maps = {}
        for slot_type in slot_map.keys():
            path = os.path.join(self.data_dir, f'{slot_type}_map.json')
            if os.path.exists(path):
                slot_value_maps[slot_type] = self._load_json(path)
        return label_map, slot_map, slot_value_maps

    def _load_json(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")
        with open(path, 'r') as f:
            return json.load(f)

    def load_model(self):
        logger.info(f"Loading model from: {self.model_path}")
        input_dim = 39  # Assuming enriched features
        model = ImprovedTwiSpeechModel(
            input_dim=input_dim,
            hidden_dim=128,
            num_classes=len(self.label_map),
            num_slot_classes=len(self.slot_map),
            slot_value_maps=self.slot_value_maps
        )
        try:
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            logger.info("Model loaded successfully!")
            return model
        except Exception as e:
            logger.error(f"Error loading model state_dict: {e}")
            raise

    def predict(self, audio_path):
        features = self.audio_processor.preprocess(audio_path)
        features = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            intent_logits, slot_type_logits, slot_value_logits = self.model(features)

            intent_probs = torch.softmax(intent_logits, dim=1)
            intent_conf, intent_idx = torch.max(intent_probs, 1)
            predicted_intent = self.idx_to_label.get(intent_idx.item(), "Unknown")

            slot_type_probs = torch.sigmoid(slot_type_logits)
            predicted_slot_types = {self.idx_to_slot[i]: prob.item() for i, prob in enumerate(slot_type_probs[0]) if prob > 0.5}

            predicted_slot_values = {}
            for slot_type, value_logits in slot_value_logits.items():
                if slot_type in predicted_slot_types:
                    value_probs = torch.softmax(value_logits, dim=1)
                    val_conf, val_idx = torch.max(value_probs, 1)
                    predicted_value = self.idx_to_slot_value[slot_type].get(val_idx.item(), "Unknown")
                    if predicted_value != '__none__':
                        predicted_slot_values[slot_type] = {'value': predicted_value, 'confidence': val_conf.item()}

        return predicted_intent, intent_conf.item(), predicted_slot_values

    def test_file(self, file_path):
        print(f"\n--- Testing: {file_path} ---")
        intent, confidence, slots = self.predict(file_path)
        print(f"\nðŸŽ¯ Predicted Intent: {intent} (Confidence: {confidence:.2%})")
        print("ðŸŽ¯ Predicted Slots:")
        if slots:
            for slot, info in slots.items():
                print(f"  - {slot}: {info['value']} (Confidence: {info['confidence']:.2%})")
        else:
            print("  (No slots detected)")

    def record_and_test(self, duration=3):
        print(f"\nRecording for {duration} seconds...")
        sample_rate = 16000
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        print("Recording finished.")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            self.test_file(tmp.name)

def main():
    parser = argparse.ArgumentParser(description="Test the jointly trained intent and slot model.")
    parser.add_argument('--model', required=True, help='Path to the trained model file (.pt)')
    parser.add_argument('--file', help='Path to an audio file to test. If not provided, will record live.')
    parser.add_argument('--duration', type=int, default=3, help='Duration for live recording in seconds.')
    parser.add_argument('--loop', action='store_true', help='Loop for continuous live testing.')

    args = parser.parse_args()

    try:
        tester = JointModelTester(model_path=args.model)
        if args.file:
            tester.test_file(args.file)
        else:
            while True:
                input("Press Enter to start recording...")
                tester.record_and_test(args.duration)
                if not args.loop:
                    break
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
