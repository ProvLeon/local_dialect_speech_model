#!/usr/bin/env python3
"""
Optimized Speech Recognition Engine for Twi Language
===================================================

This module implements an optimized speech recognition system that uses a
multi-task Whisper model for both speech-to-text and intent classification.

Author: AI Assistant
Date: 2025-11-07
"""

import os
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download

import torch
import numpy as np
import librosa
import soundfile as sf
from transformers import WhisperProcessor

# Import the multi-task model and config from the training script
from train_whisper_twi import WhisperForMultiTask, TwiWhisperConfig

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import configuration
from config.config import OptimizedConfig

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio preprocessing and format conversion."""

    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.sample_rate = config.AUDIO["sample_rate"]
        self.max_duration = config.AUDIO["max_duration"]
        self.min_duration = config.AUDIO["min_duration"]

    def load_audio(self, audio_path: str) -> np.ndarray:
        """Load and preprocess audio file."""
        try:
            audio, sr = librosa.load(
                audio_path, sr=self.sample_rate, mono=True, duration=self.max_duration
            )
            duration = len(audio) / sr
            if duration < self.min_duration:
                logger.warning(f"Audio too short: {duration:.2f}s < {self.min_duration}s")
                return None
            return audio
        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            return None

    def convert_webm_to_wav(self, input_path: str, output_path: str) -> bool:
        """Convert WebM audio to WAV format."""
        try:
            audio, sr = librosa.load(input_path, sr=self.sample_rate, mono=True)
            sf.write(output_path, audio, sr)
            return True
        except Exception as e:
            logger.error(f"Failed to convert {input_path} to WAV: {e}")
            return False


class MultiTaskWhisperRecognizer:
    """Handles speech-to-text and intent classification using the multi-task model."""

    def __init__(self, config: OptimizedConfig, huggingface_repo_id: Optional[str] = None):
        self.config = config
        self.model = None
        self.processor = None
        self.device = config.get_device()
        self.id_to_label = {}
        self.huggingface_repo_id = huggingface_repo_id
        self._load_model()

    def _load_model(self):
        """Load the fine-tuned multi-task Whisper model."""
        if self.huggingface_repo_id:
            logger.info(f"Loading multi-task Whisper model from Hugging Face Hub: {self.huggingface_repo_id}")
            self.model = WhisperForMultiTask.from_pretrained(self.huggingface_repo_id)
            self.processor = WhisperProcessor.from_pretrained(self.huggingface_repo_id)

            # Download intent_labels.json from Hugging Face Hub
            try:
                label_file = hf_hub_download(repo_id=self.huggingface_repo_id, filename="intent_labels.json")
                with open(label_file, "r") as f:
                    label_to_id = json.load(f)
                    self.id_to_label = {int(i): label for label, i in label_to_id.items()} # Keys might be strings
            except Exception as e:
                logger.warning(f"Could not download intent_labels.json from Hugging Face Hub: {e}. Intent classification might be limited.")
                # Fallback: try to get labels from model config if available
                if hasattr(self.model.config, "id2label") and self.model.config.id2label:
                    self.id_to_label = {int(k): v for k, v in self.model.config.id2label.items()}
                else:
                    logger.warning("No intent labels found in model config either.")

        else:
            custom_model_path = self.config.WHISPER.get("custom_model_path")
            if not custom_model_path or not Path(custom_model_path).exists():
                # Default to a local whisper_twi model if custom_model_path is not set or doesn't exist
                local_default_model_path = Path(__file__).parent.parent / "models" / "whisper_twi"
                if local_default_model_path.exists():
                    custom_model_path = str(local_default_model_path)
                    logger.info(f"Custom model path not found, defaulting to local: {custom_model_path}")
                else:
                    raise ValueError(f"Custom multi-task model not found at {custom_model_path} and no local default 'whisper_twi' model found at {local_default_model_path}")

            logger.info(f"Loading multi-task Whisper model from local path: {custom_model_path}")
            self.model = WhisperForMultiTask.from_pretrained(custom_model_path)
            self.processor = WhisperProcessor.from_pretrained(custom_model_path)

            # Load intent labels from local path
            label_path = Path(custom_model_path) / "intent_labels.json"
            if label_path.exists():
                with open(label_path, "r") as f:
                    label_to_id = json.load(f)
                    self.id_to_label = {int(i): label for label, i in label_to_id.items()}
            else:
                logger.warning("intent_labels.json not found locally. Intent classification will be limited.")
                # Fallback: try to get labels from model config if available
                if hasattr(self.model.config, "id2label") and self.model.config.id2label:
                    self.id_to_label = {int(k): v for k, v in self.model.config.id2label.items()}
                else:
                    logger.warning("No intent labels found in model config either.")

        self.model.to(self.device)
        logger.info(f"Multi-task Whisper model loaded successfully on {self.device}")

    def recognize(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio and classify intent."""
        start_time = time.time()
        try:
            audio, sr = librosa.load(audio_path, sr=self.config.AUDIO["sample_rate"])
            inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt").to(self.device)

            with torch.no_grad():
                # Generate transcription
                predicted_ids = self.model.transcription_model.generate(
                    inputs.input_features,
                    max_length=448,
                    num_beams=5,
                    early_stopping=True,
                )
                # Get intent logits
                classification_output = self.model.classification_model(inputs.input_features)
                intent_logits = classification_output.logits

            # Decode transcription
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            # Process intent
            intent_probs = torch.softmax(intent_logits, dim=-1).squeeze()
            best_intent_id = torch.argmax(intent_probs).item()
            best_intent_prob = intent_probs[best_intent_id].item()
            best_intent_label = self.id_to_label.get(best_intent_id, "unknown")

            processing_time = time.time() - start_time
            return {
                "transcription": {
                    "text": transcription.strip(),
                    "confidence": 0.9, # Placeholder confidence
                },
                "intent": {
                    "intent": best_intent_label,
                    "confidence": best_intent_prob,
                },
                "processing_time": processing_time,
            }
        except Exception as e:
            logger.error(f"Recognition failed: {e}")
            return {"error": str(e), "status": "failed"}


class OptimizedSpeechRecognizer:
    """Main speech recognition engine."""

    def __init__(self, config: OptimizedConfig = None, huggingface_repo_id: Optional[str] = None):
        self.config = config or OptimizedConfig()
        self.audio_processor = AudioProcessor(self.config)
        self.recognizer = MultiTaskWhisperRecognizer(self.config, huggingface_repo_id=huggingface_repo_id)
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_processing_time": 0.0,
        }
        logger.info("OptimizedSpeechRecognizer initialized successfully")

    def recognize(self, audio_path: str) -> Dict[str, Any]:
        """Complete speech recognition pipeline."""
        start_time = time.time()
        self.stats["total_requests"] += 1

        try:
            audio_data = self.audio_processor.load_audio(audio_path)
            if audio_data is None:
                raise ValueError("Failed to load audio file")

            result = self.recognizer.recognize(audio_path)
            if "error" in result:
                raise ValueError(result["error"])

            total_time = time.time() - start_time
            result["processing_time"] = total_time
            result["status"] = "success"
            
            self.stats["successful_requests"] += 1
            self._update_avg_processing_time(total_time)

            return result

        except Exception as e:
            logger.error(f"Recognition failed for {audio_path}: {e}")
            self.stats["failed_requests"] += 1
            return {
                "error": str(e),
                "status": "failed",
                "processing_time": time.time() - start_time,
            }

    async def recognize_async(self, audio_path: str) -> Dict[str, Any]:
        """Asynchronous version of recognize method."""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, self.recognize, audio_path)

    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        success_rate = (self.stats["successful_requests"] / max(1, self.stats["total_requests"])) * 100
        return {**self.stats, "success_rate": success_rate}

    def _update_avg_processing_time(self, processing_time: float):
        """Update average processing time."""
        alpha = 0.1
        if self.stats["avg_processing_time"] == 0.0:
            self.stats["avg_processing_time"] = processing_time
        else:
            self.stats["avg_processing_time"] = (
                alpha * processing_time + (1 - alpha) * self.stats["avg_processing_time"]
            )

def create_speech_recognizer(
    config_overrides: Dict[str, Any] = None,
    huggingface_repo_id: Optional[str] = None,
) -> OptimizedSpeechRecognizer:
    """Factory function to create speech recognizer."""
    config = OptimizedConfig()
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return OptimizedSpeechRecognizer(config, huggingface_repo_id=huggingface_repo_id)
