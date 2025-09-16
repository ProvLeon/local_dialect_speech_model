#!/usr/bin/env python3
"""
Easy-to-use inference interface for the packaged speech model.
"""

import os
import sys
import json
import torch
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# Add project root directory to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from src.models.speech_model import ImprovedTwiSpeechModel, IntentOnlyModel
    from src.preprocessing.audio_processor import AudioProcessor
except ImportError:
    logging.warning("Could not import model components. Please ensure src/ is available.")

logger = logging.getLogger(__name__)

class ModelInference:
    """Simple inference interface for the packaged model."""

    def __init__(self, package_path: str):
        self.package_path = Path(package_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load configuration
        self.config = self._load_config()

        # Load model
        self.model = self._load_model()

        # Load label maps
        self.label_map, self.idx_to_label = self._load_label_maps()

        # Initialize audio processor
        self.audio_processor = AudioProcessor()

        logger.info(f"Model loaded successfully on {self.device}")

    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration."""
        config_path = self.package_path / 'config' / 'config.json'
        with open(config_path, 'r') as f:
            return json.load(f)

    def _load_model(self):
        """Load the model based on configuration."""
        model_path = self.package_path / 'model' / 'model_state_dict.bin'

        # Create model based on type
        if self.config['model_type'] == 'full':
            model = ImprovedTwiSpeechModel(
                input_dim=self.config.get('input_dim', 39),
                hidden_dim=self.config.get('hidden_dim', 128),
                num_classes=self.config.get('num_classes', 47),
                num_layers=self.config.get('num_layers', 2),
                dropout=self.config.get('dropout', 0.3),
                num_slot_classes=self.config.get('num_slot_classes', 1),
                slot_value_maps=self.config.get('slot_value_maps', {})
            )
        else:  # intent_only or enhanced
            model = IntentOnlyModel(
                input_dim=self.config.get('input_dim', 39),
                hidden_dim=self.config.get('hidden_dim', 128),
                num_classes=self.config.get('num_classes', 47),
                num_layers=self.config.get('num_layers', 2),
                dropout=self.config.get('dropout', 0.3)
            )

        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()

        return model

    def _load_label_maps(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Load label mappings."""
        label_path = self.package_path / 'tokenizer' / 'label_map.json'

        if label_path.exists():
            with open(label_path, 'r') as f:
                label_map = json.load(f)
        else:
            # Create default mapping
            label_map = {f"intent_{i}": i for i in range(self.config['num_classes'])}

        idx_to_label = {v: k for k, v in label_map.items()}
        return label_map, idx_to_label

    def predict(self, audio_path: str) -> Tuple[str, float]:
        """
        Make prediction on audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (predicted_intent, confidence)
        """
        try:
            # Extract features
            features = self.audio_processor.preprocess(audio_path)

            # Ensure correct shape
            if features.shape[1] < 50:
                pad_width = 50 - features.shape[1]
                features = torch.nn.functional.pad(torch.tensor(features), (0, pad_width))
            elif features.shape[1] > 200:
                features = features[:, :200]

            # Convert to tensor
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

            # Make prediction
            with torch.no_grad():
                outputs = self.model(features_tensor)

                # Handle different output formats
                if isinstance(outputs, tuple):
                    intent_logits = outputs[0]
                else:
                    intent_logits = outputs

                probabilities = torch.softmax(intent_logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            # Map to label
            predicted_intent = self.idx_to_label.get(predicted_class, f"unknown_{predicted_class}")

            return predicted_intent, confidence

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def predict_topk(self, audio_path: str, top_k: int = 5) -> Tuple[str, float, list]:
        """
        Make prediction on audio file and return top-k predictions.

        Args:
            audio_path: Path to audio file
            top_k: Number of top predictions to return

        Returns:
            Tuple of (predicted_intent, confidence, top_predictions_list)
        """
        try:
            # Extract features
            features = self.audio_processor.preprocess(audio_path)

            # Ensure correct shape
            if features.shape[1] < 50:
                pad_width = 50 - features.shape[1]
                features = torch.nn.functional.pad(torch.tensor(features), (0, pad_width))
            elif features.shape[1] > 200:
                features = features[:, :200]

            # Convert to tensor
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

            # Make prediction
            with torch.no_grad():
                outputs = self.model(features_tensor)

                # Handle different output formats
                if isinstance(outputs, tuple):
                    intent_logits = outputs[0]
                else:
                    intent_logits = outputs

                probabilities = torch.softmax(intent_logits, dim=1)

                # Get top-k predictions
                top_k_probs, top_k_indices = torch.topk(probabilities[0], min(top_k, probabilities.shape[1]))

                # Create top predictions list
                top_predictions = []
                for i in range(len(top_k_indices)):
                    idx = top_k_indices[i].item()
                    prob = top_k_probs[i].item()
                    intent = self.idx_to_label.get(idx, f"unknown_{idx}")
                    top_predictions.append({"intent": intent, "confidence": prob})

                # Get the top prediction
                predicted_intent = top_predictions[0]["intent"]
                confidence = top_predictions[0]["confidence"]

            return predicted_intent, confidence, top_predictions

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def predict_batch(self, audio_paths: list) -> list:
        """Make predictions on multiple audio files."""
        results = []
        for audio_path in audio_paths:
            try:
                intent, confidence = self.predict(audio_path)
                results.append({'audio_path': audio_path, 'intent': intent, 'confidence': confidence})
            except Exception as e:
                results.append({'audio_path': audio_path, 'error': str(e)})
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_name': self.config.get('model_name'),
            'version': self.config.get('version'),
            'model_type': self.config.get('model_type'),
            'num_classes': self.config.get('num_classes'),
            'input_dim': self.config.get('input_dim'),
            'device': str(self.device),
            'available_intents': list(self.label_map.keys())
        }

def load_model(package_path: str) -> ModelInference:
    """Convenience function to load model."""
    return ModelInference(package_path)
