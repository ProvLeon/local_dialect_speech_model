#!/usr/bin/env python3
"""
Recommended Speech Recognition Solution for Twi Language
======================================================

This is a production-ready implementation using OpenAI Whisper as the base
with custom intent classification. This approach is recommended for low-resource
languages like Twi where you have limited training data.

Architecture:
Audio → Whisper (Speech-to-Text) → Intent Classifier → Response

Author: AI Assistant
Date: 2025-11-05
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
import whisper
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import librosa
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TwiSpeechRecognizer:
    """
    Production-ready Twi speech recognition system using Whisper + Custom Intent Classification
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.whisper_model = None
        self.intent_classifier = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize models
        self._load_whisper_model()
        self._load_intent_classifier()

        logger.info(f"TwiSpeechRecognizer initialized on {self.device}")

    def _load_whisper_model(self):
        """Load Whisper model for speech-to-text."""
        model_size = self.config.get("whisper_model", "large-v3")
        logger.info(f"Loading Whisper model: {model_size}")

        try:
            self.whisper_model = whisper.load_model(model_size, device=self.device)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def _load_intent_classifier(self):
        """Load or initialize intent classification model."""
        intent_model_path = self.config.get("intent_model_path")

        if intent_model_path and Path(intent_model_path).exists():
            # Load custom trained model
            logger.info(f"Loading custom intent classifier from {intent_model_path}")
            self.intent_classifier = pipeline(
                "text-classification",
                model=intent_model_path,
                device=0 if torch.cuda.is_available() else -1,
            )
        else:
            # Use multilingual BERT as baseline
            logger.info("Using multilingual BERT for intent classification")
            self.intent_classifier = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",
                device=0 if torch.cuda.is_available() else -1,
            )

    def transcribe_audio(self, audio_path: str, language: str = "tw") -> Dict[str, Any]:
        """
        Transcribe audio file to text using Whisper.

        Args:
            audio_path: Path to audio file
            language: Language code (tw for Twi)

        Returns:
            Dictionary with transcription results
        """
        logger.info(f"Transcribing audio: {audio_path}")
        start_time = time.time()

        try:
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(
                audio_path,
                language=language,
                task="transcribe",
                fp16=torch.cuda.is_available(),
                verbose=False,
            )

            transcription_time = time.time() - start_time
            logger.info(f"Transcription completed in {transcription_time:.2f}s")

            return {
                "text": result["text"].strip(),
                "language": result.get("language", language),
                "segments": result.get("segments", []),
                "processing_time": transcription_time,
                "confidence": self._calculate_confidence(result),
            }

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def classify_intent(self, text: str) -> Dict[str, Any]:
        """
        Classify intent from transcribed text.

        Args:
            text: Transcribed text in Twi

        Returns:
            Dictionary with intent classification results
        """
        logger.info(f"Classifying intent for text: {text[:50]}...")

        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)

            # Classify intent
            result = self.intent_classifier(processed_text)

            return {
                "intent": result[0]["label"],
                "confidence": result[0]["score"],
                "all_predictions": result,
                "processed_text": processed_text,
            }

        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return {"intent": "unknown", "confidence": 0.0, "error": str(e)}

    def process_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """
        Complete pipeline: Audio → Transcription → Intent Classification

        Args:
            audio_path: Path to audio file

        Returns:
            Complete processing results
        """
        logger.info(f"Processing audio file: {audio_path}")
        start_time = time.time()

        try:
            # Step 1: Transcribe audio
            transcription_result = self.transcribe_audio(audio_path)

            # Step 2: Classify intent
            intent_result = self.classify_intent(transcription_result["text"])

            total_time = time.time() - start_time

            return {
                "transcription": transcription_result,
                "intent": intent_result,
                "total_processing_time": total_time,
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return {
                "error": str(e),
                "status": "failed",
                "total_processing_time": time.time() - start_time,
            }

    def _preprocess_text(self, text: str) -> str:
        """Preprocess Twi text for intent classification."""
        # Basic text cleaning
        text = text.lower().strip()

        # Remove common Twi stop words if needed
        twi_stopwords = ["na", "ne", "no", "a", "wo", "me"]
        words = text.split()
        filtered_words = [w for w in words if w not in twi_stopwords]

        return " ".join(filtered_words) if filtered_words else text

    def _calculate_confidence(self, whisper_result: Dict) -> float:
        """Calculate confidence score from Whisper segments."""
        if "segments" not in whisper_result:
            return 0.5  # Default confidence

        segments = whisper_result["segments"]
        if not segments:
            return 0.5

        # Average confidence from all segments
        total_confidence = sum(segment.get("avg_logprob", -1.0) for segment in segments)
        avg_confidence = total_confidence / len(segments)

        # Convert log probability to confidence (0-1)
        confidence = max(0.0, min(1.0, (avg_confidence + 1.0)))
        return confidence


class TwiDataCollector:
    """Data collection and augmentation utilities for Twi speech data."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def augment_audio(self, audio_path: str, output_prefix: str) -> List[str]:
        """
        Create augmented versions of audio file.

        Args:
            audio_path: Original audio file
            output_prefix: Prefix for output files

        Returns:
            List of augmented file paths
        """
        logger.info(f"Augmenting audio: {audio_path}")

        # Load original audio
        audio, sr = librosa.load(audio_path, sr=16000)
        augmented_files = []

        # Speed variations
        for speed in [0.8, 0.9, 1.1, 1.2]:
            augmented_audio = librosa.effects.time_stretch(audio, rate=speed)
            output_path = self.output_dir / f"{output_prefix}_speed_{speed}.wav"
            sf.write(output_path, augmented_audio, sr)
            augmented_files.append(str(output_path))

        # Pitch variations
        for pitch_shift in [-2, -1, 1, 2]:
            augmented_audio = librosa.effects.pitch_shift(
                audio, sr=sr, n_steps=pitch_shift
            )
            output_path = self.output_dir / f"{output_prefix}_pitch_{pitch_shift}.wav"
            sf.write(output_path, augmented_audio, sr)
            augmented_files.append(str(output_path))

        # Add noise
        noise_levels = [0.005, 0.01]
        for noise_level in noise_levels:
            noise = np.random.normal(0, noise_level, audio.shape)
            augmented_audio = audio + noise
            output_path = self.output_dir / f"{output_prefix}_noise_{noise_level}.wav"
            sf.write(output_path, augmented_audio, sr)
            augmented_files.append(str(output_path))

        logger.info(f"Created {len(augmented_files)} augmented files")
        return augmented_files


class TwiIntentTrainer:
    """Fine-tune intent classification model on Twi data."""

    def __init__(self, base_model: str = "microsoft/DialoGPT-medium"):
        self.base_model = base_model
        self.tokenizer = None
        self.model = None

    def prepare_dataset(self, data: List[Dict[str, str]]) -> Dataset:
        """
        Prepare dataset for training.

        Args:
            data: List of {"text": "...", "intent": "..."} dictionaries

        Returns:
            HuggingFace Dataset
        """
        # Create intent labels
        unique_intents = list(set(item["intent"] for item in data))
        intent_to_id = {intent: idx for idx, intent in enumerate(unique_intents)}

        # Prepare data
        texts = [item["text"] for item in data]
        labels = [intent_to_id[item["intent"]] for item in data]

        dataset = Dataset.from_dict({"text": texts, "labels": labels})

        return dataset, intent_to_id

    def fine_tune(self, dataset: Dataset, output_dir: str, num_epochs: int = 3):
        """Fine-tune model on Twi intent data."""
        from transformers import Trainer, TrainingArguments

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model, num_labels=len(dataset.unique("labels"))
        )

        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], truncation=True, padding=True, max_length=512
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
        )

        # Train
        trainer.train()
        trainer.save_model(output_dir)


def create_sample_config() -> Dict[str, Any]:
    """Create sample configuration for the Twi speech recognizer."""
    return {
        "whisper_model": "large-v3",  # or "medium", "small" for faster inference
        "intent_model_path": None,  # Path to custom intent model
        "supported_intents": [
            "greeting",
            "question",
            "request",
            "complaint",
            "compliment",
            "goodbye",
        ],
        "audio_settings": {"sample_rate": 16000, "channels": 1, "format": "wav"},
        "processing": {
            "max_audio_length": 30,  # seconds
            "confidence_threshold": 0.5,
        },
    }


def main():
    """Example usage of the Twi Speech Recognizer."""

    # Create configuration
    config = create_sample_config()

    # Initialize recognizer
    recognizer = TwiSpeechRecognizer(config)

    # Example: Process an audio file
    audio_file = "sample_twi_audio.wav"  # Replace with actual file

    if Path(audio_file).exists():
        result = recognizer.process_audio_file(audio_file)

        print("=== Processing Results ===")
        print(f"Status: {result['status']}")

        if result["status"] == "success":
            print(f"Transcription: {result['transcription']['text']}")
            print(f"Intent: {result['intent']['intent']}")
            print(f"Confidence: {result['intent']['confidence']:.2f}")
            print(f"Processing Time: {result['total_processing_time']:.2f}s")
        else:
            print(f"Error: {result['error']}")
    else:
        print("Sample audio file not found. Please provide a valid audio file.")

        # Show configuration example
        print("\n=== Sample Configuration ===")
        print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
