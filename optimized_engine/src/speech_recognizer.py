#!/usr/bin/env python3
"""
Optimized Speech Recognition Engine for Twi Language
===================================================

This module implements an optimized speech recognition system that uses:
1. OpenAI Whisper for speech-to-text conversion
2. Custom intent classification for Twi language commands
3. Efficient processing pipeline for real-time applications

The system is designed to overcome the limitations of training custom
speech recognition models with limited data by leveraging pre-trained
models and focusing on intent classification.

Author: AI Assistant
Date: 2025-11-05
"""

import os
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import whisper
import numpy as np
import librosa
import soundfile as sf
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import json

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
            # Load audio with librosa
            audio, sr = librosa.load(
                audio_path, sr=self.sample_rate, mono=True, duration=self.max_duration
            )

            # Validate audio duration
            duration = len(audio) / sr
            if duration < self.min_duration:
                logger.warning(
                    f"Audio too short: {duration:.2f}s < {self.min_duration}s"
                )
                return None

            # Apply preprocessing
            if self.config.AUDIO["noise_reduction"]:
                audio = self._reduce_noise(audio)

            if self.config.AUDIO["volume_normalization"]:
                audio = self._normalize_volume(audio)

            return audio

        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            return None

    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """Apply basic noise reduction."""
        # Simple high-pass filter to remove low-frequency noise
        from scipy.signal import butter, filtfilt

        nyquist = self.sample_rate / 2
        low = 80 / nyquist  # Remove frequencies below 80Hz
        b, a = butter(4, low, btype="high")

        return filtfilt(b, a, audio)

    def _normalize_volume(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio volume."""
        # RMS normalization
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            audio = audio / rms * 0.1  # Target RMS of 0.1
        return audio

    def convert_webm_to_wav(self, input_path: str, output_path: str) -> bool:
        """Convert WebM audio to WAV format."""
        try:
            # Load with librosa (handles various formats)
            audio, sr = librosa.load(input_path, sr=self.sample_rate, mono=True)

            # Save as WAV
            sf.write(output_path, audio, sr)
            return True

        except Exception as e:
            logger.error(f"Failed to convert {input_path} to WAV: {e}")
            return False


class WhisperTranscriber:
    """Handles speech-to-text using OpenAI Whisper."""

    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.is_custom_model = False
        self.device = config.get_device()
        self._load_model()

    def _load_model(self):
        """Load Whisper model (either pre-trained or fine-tuned)."""
        try:
            model_size = self.config.WHISPER["model_size"]
            custom_model_path = self.config.WHISPER.get("custom_model_path")

            if custom_model_path and Path(custom_model_path).exists():
                # Load fine-tuned model
                logger.info(
                    f"Loading fine-tuned Whisper model from: {custom_model_path}"
                )

                try:
                    from transformers import (
                        WhisperForConditionalGeneration,
                        WhisperProcessor,
                    )

                    self.model = WhisperForConditionalGeneration.from_pretrained(
                        custom_model_path
                    )
                    self.processor = WhisperProcessor.from_pretrained(custom_model_path)
                    self.model.to(self.device)
                    self.is_custom_model = True

                    logger.info(
                        f"Fine-tuned Whisper model loaded successfully on {self.device}"
                    )

                except Exception as e:
                    logger.warning(f"Failed to load fine-tuned model: {e}")
                    logger.info("Falling back to pre-trained model")
                    self._load_pretrained_model(model_size)
            else:
                # Load pre-trained model
                self._load_pretrained_model(model_size)

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def _load_pretrained_model(self, model_size):
        """Load pre-trained Whisper model."""
        logger.info(f"Loading pre-trained Whisper model: {model_size}")
        self.model = whisper.load_model(model_size, device=self.device)
        self.is_custom_model = False
        logger.info(f"Pre-trained Whisper model loaded successfully on {self.device}")

    def transcribe(self, audio_path: str, language: str = None) -> Dict[str, Any]:
        """Transcribe audio to text using Whisper."""
        start_time = time.time()

        try:
            if self.is_custom_model:
                # Use fine-tuned model with transformers
                result = self._transcribe_with_custom_model(audio_path, language)
            else:
                # Use pre-trained model with whisper library
                result = self._transcribe_with_pretrained_model(audio_path, language)

            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            result["model_size"] = self.config.WHISPER["model_size"]

            return result

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                "text": "",
                "language": language,
                "confidence": 0.0,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    def _transcribe_with_custom_model(
        self, audio_path: str, language: str
    ) -> Dict[str, Any]:
        """Transcribe using fine-tuned Whisper model."""
        import librosa
        import torch

        # Load and preprocess audio
        audio, sr = librosa.load(audio_path, sr=16000)

        # Process with fine-tuned model
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        inputs = inputs.to(self.device)

        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.model.generate(
                inputs.input_features,
                max_length=448,
                num_beams=5,
                early_stopping=True,
                language=None,  # Auto-detect language
            )

        # Decode transcription
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]

        return {
            "text": transcription.strip(),
            "language": language,
            "confidence": 0.9,  # Fine-tuned models typically have high confidence
            "segments": [],
        }

    def _transcribe_with_pretrained_model(
        self, audio_path: str, language: str
    ) -> Dict[str, Any]:
        """Transcribe using pre-trained Whisper model."""
        # Use auto-detection since Whisper doesn't officially support 'tw' language code
        result = self.model.transcribe(
            audio_path,
            language=None,  # Auto-detect language - Whisper may detect as similar language
            task=self.config.WHISPER["task"],
            beam_size=self.config.WHISPER["beam_size"],
            best_of=self.config.WHISPER["best_of"],
            temperature=self.config.WHISPER["temperature"],
            compression_ratio_threshold=self.config.WHISPER[
                "compression_ratio_threshold"
            ],
            logprob_threshold=self.config.WHISPER["logprob_threshold"],
            no_speech_threshold=self.config.WHISPER["no_speech_threshold"],
            condition_on_previous_text=self.config.WHISPER[
                "condition_on_previous_text"
            ],
            fp16=self.config.PERFORMANCE["mixed_precision"] and self.device == "cuda",
            verbose=False,
        )

        # Extract confidence from segments
        confidence = self._calculate_confidence(result)

        return {
            "text": result["text"].strip(),
            "language": result.get("language", "detected"),
            "confidence": confidence,
            "segments": result.get("segments", []),
        }

    def _calculate_confidence(self, whisper_result: Dict) -> float:
        """Calculate confidence score from Whisper segments."""
        segments = whisper_result.get("segments", [])
        if not segments:
            return 0.5  # Default confidence

        # Average log probability of all segments
        total_logprob = sum(segment.get("avg_logprob", -1.0) for segment in segments)
        avg_logprob = total_logprob / len(segments)

        # Convert to confidence (0-1 range)
        confidence = max(0.0, min(1.0, (avg_logprob + 1.0)))
        return confidence


class TwiIntentClassifier:
    """Handles intent classification for Twi language."""

    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.intent_to_id = {}
        self.id_to_intent = {}
        self.device = config.get_device()

        self._load_or_create_model()

    def _load_or_create_model(self):
        """Load existing model or create new one."""
        custom_path = self.config.INTENT_CLASSIFIER["custom_model_path"]

        if custom_path and Path(custom_path).exists():
            self._load_custom_model(custom_path)
        else:
            self._create_baseline_model()

    def _load_custom_model(self, model_path: str):
        """Load custom trained intent classification model."""
        try:
            logger.info(f"Loading custom intent model from {model_path}")

            self.pipeline = pipeline(
                "text-classification",
                model=model_path,
                tokenizer=model_path,
                device=0 if self.device == "cuda" else -1,
            )

            # Load label mappings
            label_path = Path(model_path) / "intent_labels.json"
            if label_path.exists():
                with open(label_path, "r") as f:
                    label_data = json.load(f)
                    self.intent_to_id = label_data["intent_to_id"]
                    self.id_to_intent = label_data["id_to_intent"]

            logger.info("Custom intent model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load custom model: {e}")
            self._create_baseline_model()

    def _create_baseline_model(self):
        """Create baseline intent classification using pre-trained model."""
        try:
            logger.info("Creating baseline intent classification model")

            # Create intent mappings
            intents = list(self.config.INTENTS.keys())
            self.intent_to_id = {intent: idx for idx, intent in enumerate(intents)}
            self.id_to_intent = {
                idx: intent for intent, idx in self.intent_to_id.items()
            }

            # Use a simple similarity-based approach for baseline
            self.pipeline = None  # Will use similarity matching

            logger.info(f"Baseline model created with {len(intents)} intents")

        except Exception as e:
            logger.error(f"Failed to create baseline model: {e}")
            raise

    def classify_intent(self, text: str) -> Dict[str, Any]:
        """Classify intent from text."""
        start_time = time.time()

        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)

            if self.pipeline:
                # Use trained model
                results = self.pipeline(processed_text)

                # Apply confidence boost for high-priority intents
                boosted_results = self._apply_confidence_boost(results)

                return {
                    "intent": boosted_results[0]["label"],
                    "confidence": boosted_results[0]["score"],
                    "alternatives": boosted_results[
                        : self.config.INTENT_CLASSIFIER["top_k"]
                    ],
                    "processed_text": processed_text,
                    "processing_time": time.time() - start_time,
                    "method": "trained_model",
                }
            else:
                # Use similarity-based classification
                result = self._similarity_classification(processed_text)
                result["processing_time"] = time.time() - start_time
                result["method"] = "similarity_based"
                return result

        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    def _preprocess_text(self, text: str) -> str:
        """Preprocess Twi text for classification."""
        # Basic cleaning
        text = text.lower().strip()

        # Remove punctuation
        import string

        text = text.translate(str.maketrans("", "", string.punctuation))

        # Normalize common Twi variations
        twi_normalizations = {
            "kɔ": ["ko", "go"],
            "me": ["my", "my"],
            "wo": ["you", "your"],
            "hwehwɛ": ["search", "find"],
            "cart": ["shopping cart", "basket"],
        }

        for standard, variations in twi_normalizations.items():
            for variation in variations:
                text = text.replace(variation, standard)

        return text

    def _similarity_classification(self, text: str) -> Dict[str, Any]:
        """Classify intent using similarity to example phrases."""
        best_intent = "unknown"
        best_score = 0.0
        alternatives = []

        for intent, intent_config in self.config.INTENTS.items():
            examples = intent_config.get("examples", [])
            if not examples:
                continue

            # Calculate similarity to examples
            similarities = []
            for example in examples:
                similarity = self._calculate_similarity(text, example.lower())
                similarities.append(similarity)

            # Use maximum similarity
            max_similarity = max(similarities) if similarities else 0.0

            # Apply confidence boost
            confidence_boost = intent_config.get("confidence_boost", 0.0)
            boosted_score = min(1.0, max_similarity + confidence_boost)

            alternatives.append({"label": intent, "score": boosted_score})

            if boosted_score > best_score:
                best_score = boosted_score
                best_intent = intent

        # Sort alternatives by score
        alternatives.sort(key=lambda x: x["score"], reverse=True)

        return {
            "intent": best_intent,
            "confidence": best_score,
            "alternatives": alternatives[: self.config.INTENT_CLASSIFIER["top_k"]],
            "processed_text": text,
        }

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using Jaccard similarity."""
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def _apply_confidence_boost(self, results: List[Dict]) -> List[Dict]:
        """Apply confidence boost to high-priority intents."""
        boosted_results = []

        for result in results:
            intent = result["label"]
            confidence = result["score"]

            # Apply boost if configured
            if intent in self.config.INTENTS:
                boost = self.config.INTENTS[intent].get("confidence_boost", 0.0)
                confidence = min(1.0, confidence + boost)

            boosted_results.append({"label": intent, "score": confidence})

        # Re-sort by boosted confidence
        boosted_results.sort(key=lambda x: x["score"], reverse=True)
        return boosted_results


class OptimizedSpeechRecognizer:
    """Main speech recognition engine combining Whisper and intent classification."""

    def __init__(self, config: OptimizedConfig = None):
        self.config = config or OptimizedConfig()

        # Initialize components
        self.audio_processor = AudioProcessor(self.config)
        self.transcriber = WhisperTranscriber(self.config)
        self.intent_classifier = TwiIntentClassifier(self.config)

        # Performance tracking
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_processing_time": 0.0,
            "transcription_accuracy": 0.0,
            "intent_accuracy": 0.0,
        }

        logger.info("OptimizedSpeechRecognizer initialized successfully")

    def recognize(self, audio_path: str, language: str = None) -> Dict[str, Any]:
        """
        Complete speech recognition pipeline.

        Args:
            audio_path: Path to audio file
            language: Language code (default: "tw" for Twi)

        Returns:
            Dictionary with recognition results
        """
        start_time = time.time()
        self.stats["total_requests"] += 1

        try:
            # Step 1: Load and preprocess audio
            audio_data = self.audio_processor.load_audio(audio_path)
            if audio_data is None:
                raise ValueError("Failed to load audio file")

            # Step 2: Speech-to-text with Whisper
            transcription_result = self.transcriber.transcribe(audio_path, language)

            if not transcription_result.get("text"):
                raise ValueError("No text transcribed from audio")

            # Step 3: Intent classification
            intent_result = self.intent_classifier.classify_intent(
                transcription_result["text"]
            )

            # Step 4: Combine results
            total_time = time.time() - start_time

            result = {
                "transcription": transcription_result,
                "intent": intent_result,
                "audio_info": {
                    "file_path": audio_path,
                    "duration": len(audio_data) / self.config.AUDIO["sample_rate"],
                    "sample_rate": self.config.AUDIO["sample_rate"],
                },
                "processing_time": total_time,
                "timestamp": time.time(),
                "status": "success",
            }

            # Update statistics
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
                "timestamp": time.time(),
            }

    async def recognize_async(
        self, audio_path: str, language: str = None
    ) -> Dict[str, Any]:
        """Asynchronous version of recognize method."""
        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(
                executor, self.recognize, audio_path, language
            )

        return result

    def recognize_stream(
        self, audio_chunks: List[str], language: str = None
    ) -> List[Dict[str, Any]]:
        """Process multiple audio chunks in parallel."""
        results = []

        if self.config.PERFORMANCE["parallel_processing"]["enabled"]:
            # Parallel processing
            max_workers = self.config.PERFORMANCE["parallel_processing"]["max_workers"]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_chunk = {
                    executor.submit(self.recognize, chunk, language): chunk
                    for chunk in audio_chunks
                }

                # Collect results as they complete
                for future in as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    try:
                        result = future.result()
                        result["chunk_id"] = chunk
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Failed to process chunk {chunk}: {e}")
                        results.append(
                            {"chunk_id": chunk, "error": str(e), "status": "failed"}
                        )
        else:
            # Sequential processing
            for chunk in audio_chunks:
                result = self.recognize(chunk, language)
                result["chunk_id"] = chunk
                results.append(result)

        return results

    def get_supported_intents(self) -> List[Dict[str, Any]]:
        """Get list of supported intents with descriptions."""
        intents = []

        for intent, config in self.config.INTENTS.items():
            intents.append(
                {
                    "intent": intent,
                    "description": config.get("description", ""),
                    "examples": config.get("examples", []),
                    "priority": "high"
                    if config.get("confidence_boost", 0) > 0.1
                    else "normal",
                }
            )

        return intents

    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        success_rate = (
            self.stats["successful_requests"] / max(1, self.stats["total_requests"])
        ) * 100

        return {
            **self.stats,
            "success_rate": success_rate,
            "error_rate": 100 - success_rate,
            "model_info": {
                "whisper_model": self.config.WHISPER["model_size"],
                "device": self.config.get_device(),
                "supported_intents": len(self.config.INTENTS),
            },
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        health_status = {
            "status": "healthy",
            "components": {},
            "timestamp": time.time(),
        }

        try:
            # Check Whisper model
            if self.transcriber.model is not None:
                health_status["components"]["whisper"] = "healthy"
            else:
                health_status["components"]["whisper"] = "unhealthy"
                health_status["status"] = "degraded"

            # Check intent classifier
            if self.intent_classifier.intent_to_id:
                health_status["components"]["intent_classifier"] = "healthy"
            else:
                health_status["components"]["intent_classifier"] = "unhealthy"
                health_status["status"] = "degraded"

            # Check device availability
            device_info = {
                "device": self.config.get_device(),
                "cuda_available": torch.cuda.is_available() if torch else False,
            }
            health_status["device_info"] = device_info

            # Overall system status
            if all(
                status == "healthy" for status in health_status["components"].values()
            ):
                health_status["status"] = "healthy"

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)

        return health_status

    def _update_avg_processing_time(self, processing_time: float):
        """Update average processing time with exponential moving average."""
        alpha = 0.1  # Smoothing factor
        if self.stats["avg_processing_time"] == 0.0:
            self.stats["avg_processing_time"] = processing_time
        else:
            self.stats["avg_processing_time"] = (
                alpha * processing_time
                + (1 - alpha) * self.stats["avg_processing_time"]
            )


# Factory function for easy initialization
def create_speech_recognizer(
    config_overrides: Dict[str, Any] = None,
) -> OptimizedSpeechRecognizer:
    """
    Factory function to create speech recognizer with optional config overrides.

    Args:
        config_overrides: Dictionary of configuration overrides

    Returns:
        Configured OptimizedSpeechRecognizer instance
    """
    config = OptimizedConfig()

    # Apply overrides if provided
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning(f"Unknown config parameter: {key}")

    return OptimizedSpeechRecognizer(config)
