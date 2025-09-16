#!/usr/bin/env python3
"""
Self-contained inference interface for the packaged speech model.
This module does not depend on the main src/ directory and contains all necessary components.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import logging
import signal
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import warnings

# Suppress librosa warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

# Version info to verify correct file is being used
INFERENCE_VERSION = "2.0.0-fixed"
logger.info(f"🚀 Loading inference.py version {INFERENCE_VERSION} (self-contained)")

class AudioProcessor:
    """Self-contained audio processing for inference."""

    def __init__(self, sr=16000, n_mfcc=13, n_mels=26, duration=3.0):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.duration = duration
        self.target_length = int(sr * duration)

    @contextmanager
    def _timeout_handler(self, seconds):
        """Context manager for timeout handling"""
        def signal_handler(signum, frame):
            raise TimeoutError(f"Audio processing timed out after {seconds} seconds")

        # Set the signal handler and a timeout alarm
        old_handler = signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)

        try:
            yield
        finally:
            # Reset the alarm and handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    def load_audio(self, audio_path: str, timeout_seconds: int = 30) -> np.ndarray:
        """Load and preprocess audio file with timeout handling."""
        try:
            logger.info(f"Loading audio file: {audio_path}")
            start_time = time.time()

            # Load audio with timeout
            with self._timeout_handler(timeout_seconds):
                audio, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration)

            load_time = time.time() - start_time
            logger.info(f"Audio loaded in {load_time:.2f}s, length: {len(audio)} samples")

            # Ensure consistent length
            if len(audio) < self.target_length:
                # Pad with zeros
                audio = np.pad(audio, (0, self.target_length - len(audio)), mode='constant')
                logger.debug(f"Padded audio to {len(audio)} samples")
            else:
                # Truncate
                audio = audio[:self.target_length]
                logger.debug(f"Truncated audio to {len(audio)} samples")

            return audio
        except TimeoutError:
            logger.error(f"Audio loading timed out after {timeout_seconds}s: {audio_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            raise

    def extract_features(self, audio: np.ndarray, timeout_seconds: int = 15) -> np.ndarray:
        """Extract MFCC features from audio with timeout handling."""
        try:
            logger.debug("Extracting MFCC features")
            start_time = time.time()

            with self._timeout_handler(timeout_seconds):
                # Extract MFCCs
                mfccs = librosa.feature.mfcc(
                    y=audio,
                    sr=self.sr,
                    n_mfcc=self.n_mfcc,
                    n_fft=512,
                    hop_length=256
                )

                # Extract delta and delta-delta features
                delta = librosa.feature.delta(mfccs)
                delta2 = librosa.feature.delta(mfccs, order=2)

                # Combine features
                features = np.vstack([mfccs, delta, delta2])

                # Transpose to (time_steps, features)
                features = features.T

            extract_time = time.time() - start_time
            logger.info(f"Features extracted in {extract_time:.2f}s, shape: {features.shape}")

            return features
        except TimeoutError:
            logger.error(f"Feature extraction timed out after {timeout_seconds}s")
            raise
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise

    def process_audio_file(self, audio_path: str, timeout_seconds: int = 45) -> torch.Tensor:
        """Process audio file and return tensor ready for model with timeout handling."""
        logger.info(f"Processing audio file: {audio_path}")
        start_time = time.time()

        try:
            # Load audio with timeout
            audio = self.load_audio(audio_path, timeout_seconds // 2)

            # Extract features with timeout
            features = self.extract_features(audio, timeout_seconds // 3)

            # Convert to tensor and add batch dimension
            tensor = torch.FloatTensor(features).unsqueeze(0)

            process_time = time.time() - start_time
            logger.info(f"Audio processing completed in {process_time:.2f}s, tensor shape: {tensor.shape}")

            return tensor
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"Audio processing failed after {process_time:.2f}s: {e}")
            raise


class CustomAttention(nn.Module):
    """Custom attention mechanism matching the saved model."""

    def __init__(self, hidden_dim):
        super(CustomAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Simple attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        return self.out(attn_output)


class IntentOnlyModel(nn.Module):
    """Self-contained intent classification model matching saved architecture."""

    def __init__(self, input_dim=39, hidden_dim=128, num_classes=49, dropout=0.3):
        super(IntentOnlyModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Convolutional layers (as Sequential modules to match saved structure)
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),  # kernel_size=5 based on weights
            nn.BatchNorm1d(64)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128)
        )

        # LSTM layers (bidirectional, 2 layers)
        self.lstm = nn.LSTM(
            input_size=128,  # After conv2
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # Custom attention mechanism
        self.attention = CustomAttention(hidden_dim * 2)  # bidirectional

        # Shared layer
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Intent classifier
        self.intent_classifier = nn.Linear(hidden_dim, num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape

        # Transpose for conv1d (batch, channels, seq_len)
        x = x.transpose(1, 2)

        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)

        # Transpose back for LSTM (batch, seq_len, channels)
        x = x.transpose(1, 2)

        # LSTM layers
        lstm_out, _ = self.lstm(x)

        # Apply attention
        attn_out = self.attention(lstm_out)

        # Global average pooling over time dimension
        pooled = torch.mean(attn_out, dim=1)

        # Shared layer
        x = F.relu(self.shared_layer(pooled))
        x = self.dropout(x)

        # Intent classification
        logits = self.intent_classifier(x)

        return logits


class ModelInference:
    """Self-contained inference interface for the packaged model."""

    def __init__(self, package_path: str):
        self.package_path = Path(package_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load configuration
        self.config = self._load_config()

        # Initialize audio processor
        self.audio_processor = AudioProcessor()

        # Load label maps
        self.label_map, self.idx_to_label = self._load_label_maps()

        # Load model
        self.model = self._load_model()

        logger.info(f"Model loaded successfully on {self.device}")
        logger.info(f"✅ Using inference.py version {INFERENCE_VERSION}")

    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration."""
        config_path = self.package_path / 'config' / 'config.json'

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        logger.info(f"Loaded config: {config.get('model_name', 'Unknown')} v{config.get('version', 'Unknown')}")
        return config

    def _load_label_maps(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Load label mapping files."""
        label_map_path = self.package_path / 'tokenizer' / 'label_map.json'

        if not label_map_path.exists():
            # Create default label map based on config
            num_classes = self.config.get('num_classes', 49)
            label_map = {f"intent_{i}": i for i in range(num_classes)}
            idx_to_label = {i: f"intent_{i}" for i in range(num_classes)}
            logger.warning(f"Label map not found, created default with {num_classes} classes")
        else:
            with open(label_map_path, 'r') as f:
                label_map = json.load(f)
            idx_to_label = {v: k for k, v in label_map.items()}
            logger.info(f"Loaded label map with {len(label_map)} classes")

        return label_map, idx_to_label

    def _load_model(self) -> nn.Module:
        """Load the trained model."""
        # Try different model file names
        model_files = [
            'model_state_dict.bin',
            'pytorch_model.bin',
            'model.pt',
            'best_model.pt'
        ]

        model_path = None
        for model_file in model_files:
            candidate_path = self.package_path / 'model' / model_file
            if candidate_path.exists():
                model_path = candidate_path
                break

        if model_path is None:
            raise FileNotFoundError(f"No model file found in {self.package_path / 'model'}")

        # Initialize model with config parameters
        model = IntentOnlyModel(
            input_dim=self.config.get('input_dim', 39),
            hidden_dim=self.config.get('hidden_dim', 128),
            num_classes=self.config.get('num_classes', 49)
        )

        try:
            # Load state dict
            state_dict = torch.load(model_path, map_location=self.device)

            # Handle different state dict formats
            if 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            elif 'state_dict' in state_dict:
                model.load_state_dict(state_dict['state_dict'])
            else:
                model.load_state_dict(state_dict)

            model.to(self.device)
            model.eval()

            logger.info(f"Loaded model from {model_path}")
            return model

        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise

    def predict(self, audio_path: str, timeout_seconds: int = 60) -> Tuple[str, float]:
        """Predict intent from audio file with timeout handling."""
        try:
            logger.info(f"Starting prediction for: {audio_path}")
            start_time = time.time()

            # Process audio with timeout
            features = self.audio_processor.process_audio_file(audio_path, timeout_seconds // 2)
            features = features.to(self.device)

            # Make prediction with timeout
            logger.info("Running model inference")
            inference_start = time.time()

            with torch.no_grad():
                logits = self.model(features)
                probabilities = F.softmax(logits, dim=1)
                predicted_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_idx].item()

            inference_time = time.time() - inference_start
            total_time = time.time() - start_time

            # Get label
            predicted_label = self.idx_to_label.get(predicted_idx, f"unknown_{predicted_idx}")

            logger.info(f"Prediction completed in {total_time:.2f}s (inference: {inference_time:.2f}s)")
            logger.info(f"Result: {predicted_label} (confidence: {confidence:.3f})")

            return predicted_label, confidence

        except Exception as e:
            total_time = time.time() - start_time if 'start_time' in locals() else 0
            logger.error(f"Error during prediction after {total_time:.2f}s: {e}")
            raise

    def predict_topk(self, audio_path: str, top_k: int = 5, timeout_seconds: int = 60) -> Tuple[str, float, List[Dict[str, Any]]]:
        """Predict top-k intents from audio file with timeout handling."""
        try:
            logger.info(f"Starting top-k prediction for: {audio_path} (k={top_k})")
            start_time = time.time()

            # Process audio with timeout
            features = self.audio_processor.process_audio_file(audio_path, timeout_seconds // 2)
            features = features.to(self.device)

            # Make prediction with timeout
            logger.info("Running model inference")
            inference_start = time.time()

            with torch.no_grad():
                logits = self.model(features)
                probabilities = F.softmax(logits, dim=1)

                # Get top-k predictions
                top_probs, top_indices = torch.topk(probabilities[0], min(top_k, len(self.idx_to_label)))

                # Format results
                top_predictions = []
                for prob, idx in zip(top_probs, top_indices):
                    label = self.idx_to_label.get(idx.item(), f"unknown_{idx.item()}")
                    top_predictions.append({
                        'intent': label,
                        'confidence': prob.item(),
                        'index': idx.item()
                    })

                # Best prediction
                best_label = top_predictions[0]['intent']
                best_confidence = top_predictions[0]['confidence']

            inference_time = time.time() - inference_start
            total_time = time.time() - start_time

            logger.info(f"Top-k prediction completed in {total_time:.2f}s (inference: {inference_time:.2f}s)")
            logger.info(f"Top result: {best_label} (confidence: {best_confidence:.3f})")

            return best_label, best_confidence, top_predictions

        except Exception as e:
            total_time = time.time() - start_time if 'start_time' in locals() else 0
            logger.error(f"Error during top-k prediction after {total_time:.2f}s: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_name': self.config.get('model_name', 'Unknown'),
            'version': self.config.get('version', 'Unknown'),
            'description': self.config.get('description', ''),
            'model_type': self.config.get('model_type', 'intent_only'),
            'num_classes': self.config.get('num_classes', len(self.idx_to_label)),
            'available_intents': list(self.label_map.keys()),
            'device': str(self.device),
            'architecture': self.config.get('architecture', {}),
            'input_dim': self.config.get('input_dim', 39),
            'hidden_dim': self.config.get('hidden_dim', 128)
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check."""
        try:
            # Create dummy input
            dummy_input = torch.randn(1, 188, self.config.get('input_dim', 39)).to(self.device)

            with torch.no_grad():
                output = self.model(dummy_input)

            return {
                'status': 'healthy',
                'model_loaded': True,
                'device': str(self.device),
                'output_shape': list(output.shape),
                'num_classes': self.config.get('num_classes', len(self.idx_to_label))
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'model_loaded': hasattr(self, 'model'),
                'device': str(self.device) if hasattr(self, 'device') else 'unknown'
            }


# For backwards compatibility
if __name__ == "__main__":
    # Simple test
    import sys
    if len(sys.argv) > 1:
        package_path = sys.argv[1]
        model = ModelInference(package_path)
        print("Model loaded successfully!")
        print(json.dumps(model.get_model_info(), indent=2))
    else:
        print("Usage: python inference.py <package_path>")
