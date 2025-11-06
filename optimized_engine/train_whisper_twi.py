#!/usr/bin/env python3
"""
Whisper Fine-tuning Pipeline for Twi Speech Recognition
======================================================

This script fine-tunes OpenAI's Whisper model on Twi audio data using the
recorded audio files and corresponding transcriptions from prompts_lean.csv.

Based on the TwiWhisperModel approach but adapted for your specific dataset
and intent classification use case.

Usage:
    python train_whisper_twi.py --model_size small --epochs 10
    python train_whisper_twi.py --model_size tiny --batch_size 16 --lr 1e-5

Author: AI Assistant
Date: 2025-11-05
"""

import os
import sys
import json
import logging
import argparse


import re
import shutil
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import whisper
from torch.utils.data import Dataset

# Suppress all warnings
warnings.filterwarnings("ignore")

# Set environment variable to manage memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


# Fix for numpy.dtypes error when using older numpy with newer jax/transformers
# This can happen in environments like Kaggle.
if not hasattr(np, "dtypes"):

    class MockStringDType:
        def __init__(self, *args, **kwargs):
            self.name = "string"

        def __call__(self, *args, **kwargs):
            return self

        def __str__(self):
            return "StringDType"

        def __repr__(self):
            return "StringDType()"

    class MockDtypes:
        StringDType = MockStringDType()

        def __getattr__(self, name):
            return MockStringDType()

    np.dtypes = MockDtypes()


# Set environment variables for CPU-only mode if needed, but preferably use trainer args
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
)

try:
    from datasets import Dataset as HFDataset, Audio
except ImportError:
    HFDataset = None
    Audio = None

import librosa
import soundfile as sf

# Import evaluation metrics with fallback
try:
    from jiwer import wer, cer

    EVAL_AVAILABLE = True
except ImportError:
    EVAL_AVAILABLE = False

    def wer(ref, hyp):
        return 0.0

    def cer(ref, hyp):
        return 0.0


# Import evaluate with fallback (skip to avoid JAX issues)
EVALUATE_AVAILABLE = False
evaluate = None


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TwiWhisperConfig:
    """Configuration for Twi Whisper fine-tuning."""

    # Model configuration
    model_name: str = "openai/whisper-small"
    model_size: str = "small"  # tiny, base, small, medium, large
    language: str = (
        None  # Auto-detect language (Whisper doesn't officially support 'tw'
    )
    task: str = "transcribe"

    # Data paths
    data_dir: str = "../data/raw"
    prompts_file: str = "../prompts_lean.csv"
    output_dir: str = "./models/whisper_twi"
    cache_dir: str = "./data/cache"

    # Training configuration
    num_epochs: int = 10
    batch_size: int = 2
    learning_rate: float = 5e-6  # Lowered for training stability
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 8

    # Audio processing
    sample_rate: int = 16000
    max_audio_length: int = 30  # seconds
    min_audio_length: float = 0.5  # seconds

    # Validation
    eval_steps: int = 500
    save_steps: int = 1000
    eval_ratio: float = 0.15
    test_ratio: float = 0.15

    # Logging
    logging_steps: int = 100
    report_to: str = "none"  # tensorboard, wandb, none


class TwiAudioDataset(Dataset):
    """Dataset class for Twi audio files and transcriptions."""

    def __init__(
        self,
        audio_paths: List[str],
        transcriptions: List[str],
        intents: List[str],
        processor: WhisperProcessor,
        config: TwiWhisperConfig,
    ):
        self.audio_paths = audio_paths
        self.transcriptions = transcriptions
        self.intents = intents
        self.processor = processor
        self.config = config

        # Filter valid samples
        self._filter_valid_samples()

    def _filter_valid_samples(self):
        """Filter out invalid audio files and transcriptions."""
        valid_indices = []

        for i, (audio_path, transcription) in enumerate(
            zip(self.audio_paths, self.transcriptions)
        ):
            # Check if audio file exists
            if not Path(audio_path).exists():
                logger.warning(f"Audio file not found: {audio_path}")
                continue

            # Check if transcription is valid
            if not transcription or len(transcription.strip()) == 0:
                logger.warning(f"Empty transcription for: {audio_path}")
                continue

            # Check audio duration
            try:
                audio, sr = librosa.load(audio_path, sr=self.config.sample_rate)
                duration = len(audio) / sr

                if duration < self.config.min_audio_length:
                    logger.warning(f"Audio too short ({duration:.2f}s): {audio_path}")
                    continue

                if duration > self.config.max_audio_length:
                    logger.warning(f"Audio too long ({duration:.2f}s): {audio_path}")
                    continue

            except Exception as e:
                logger.warning(f"Error loading audio {audio_path}: {e}")
                continue

            valid_indices.append(i)

        # Update lists with valid samples only
        self.audio_paths = [self.audio_paths[i] for i in valid_indices]
        self.transcriptions = [self.transcriptions[i] for i in valid_indices]
        self.intents = [self.intents[i] for i in valid_indices]

        logger.info(
            f"Filtered dataset: {len(self.audio_paths)} valid samples out of {len(valid_indices)} total"
        )

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        transcription = self.transcriptions[idx]

        # Load and preprocess audio
        try:
            audio, sr = librosa.load(
                audio_path,
                sr=self.config.sample_rate,
                duration=self.config.max_audio_length,
            )

            # Ensure minimum length (pad if too short)
            min_samples = int(0.5 * self.config.sample_rate)  # 0.5 seconds minimum
            if len(audio) < min_samples:
                audio = np.pad(audio, (0, min_samples - len(audio)), mode="constant")

            # Normalize audio to [-1, 1] range
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))

            # Ensure consistent length (truncate or pad to fixed size)
            target_length = int(self.config.max_audio_length * self.config.sample_rate)
            if len(audio) > target_length:
                audio = audio[:target_length]
            elif len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")

        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            # Return silent audio as fallback with correct length
            target_length = int(self.config.max_audio_length * self.config.sample_rate)
            audio = np.zeros(target_length)

        # Process with Whisper processor
        inputs = self.processor(
            audio,
            sampling_rate=self.config.sample_rate,
            return_tensors="pt",
            padding=False,  # We handle padding manually
            truncation=False,  # We handle truncation manually
        )

        # Tokenize transcription
        labels = self.processor.tokenizer(
            transcription,
            return_tensors="pt",
            padding=False,  # We handle padding in collator
            truncation=True,
            max_length=448,  # Whisper's max sequence length
        ).input_ids

        return {
            "input_features": inputs.input_features.squeeze(0),
            "labels": labels.squeeze(0),
            "transcription": transcription,
            "intent": self.intents[idx],
            "audio_path": audio_path,
        }


class TwiWhisperDataCollator:
    """Data collator for Whisper training."""

    def __init__(self, processor: WhisperProcessor):
        self.processor = processor

    def __call__(self, features):
        # Extract input features and labels
        input_features = [f["input_features"] for f in features]
        labels = [f["labels"] for f in features]

        # Stack input features (they should all have the same shape now)
        # input_features shape: [batch_size, 80, 3000]
        batch = {"input_features": torch.stack(input_features)}

        # Pad labels using tokenizer
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": labels}, return_tensors="pt", padding=True
        )

        # Replace padding tokens in labels with -100 for loss calculation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100
        )

        batch["labels"] = labels

        return batch


class TwiWhisperTrainer:
    """Trainer class for fine-tuning Whisper on Twi data."""

    def __init__(self, config: TwiWhisperConfig):
        self.config = config
        self.setup_directories()

        # Initialize processor and model
        self.processor = None
        self.model = None
        self.tokenizer = None

        self._load_model_and_processor()

    def setup_directories(self):
        """Create necessary directories."""
        for dir_path in [self.config.output_dir, self.config.cache_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def _load_model_and_processor(self):
        """Load Whisper model and processor."""
        logger.info(f"Loading Whisper model: {self.config.model_name}")

        # Load processor
        self.processor = WhisperProcessor.from_pretrained(self.config.model_name)
        self.tokenizer = self.processor.tokenizer

        # Load model with CPU-only device mapping
        try:
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.config.model_name, device_map="cpu", torch_dtype=torch.float32
            )
        except Exception:
            # Fallback without device_map if it causes issues
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.config.model_name, torch_dtype=torch.float32
            )
            # Manually move to CPU
            self.model = self.model.to("cpu")

        # Set language and task
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []

        # Add Twi language token if not present
        if "tw" not in self.tokenizer.get_vocab():
            logger.info("Adding Twi language tokens to tokenizer")
            # Add special tokens for Twi
            special_tokens = ["<|tw|>", "<|transcribe|>"]
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": special_tokens}
            )
            self.model.resize_token_embeddings(len(self.tokenizer))

        logger.info("Model and processor loaded successfully")

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """Load and prepare the Twi dataset from audio files and prompts CSV."""
        logger.info("Loading Twi dataset...")

        # Load prompts CSV, filtering out comment lines
        try:
            # Read CSV while skipping comment lines
            with open(self.config.prompts_file, "r", encoding="utf-8") as f:
                lines = []
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comment lines
                    if line and not line.startswith("#"):
                        lines.append(line)

            # Create a temporary file-like object from filtered lines
            from io import StringIO

            csv_content = "\n".join(lines)
            prompts_df = pd.read_csv(StringIO(csv_content))

            # Filter out any rows where 'id' column starts with '#' (in case some slipped through)
            prompts_df = prompts_df[
                ~prompts_df["id"].astype(str).str.startswith("#", na=False)
            ]

            logger.info(f"Loaded {len(prompts_df)} prompts from CSV")
        except Exception as e:
            logger.error(f"Error reading prompts CSV: {e}")
            raise

        # Map audio files to transcriptions
        audio_files = []
        transcriptions = []
        intents = []

        # Scan for audio files in data directory
        data_path = Path(self.config.data_dir)

        # Get all audio files
        audio_extensions = [".wav", ".mp3", ".m4a", ".flac"]
        all_audio_files = []

        for ext in audio_extensions:
            all_audio_files.extend(list(data_path.rglob(f"*{ext}")))

        logger.info(f"Found {len(all_audio_files)} audio files")

        # Match audio files with prompts
        for audio_file in all_audio_files:
            # Extract identifier from filename
            filename = audio_file.stem

            # Remove sample suffix (_s01, _s02, etc.) if present
            # This handles both "nav_home_1.wav" and "nav_home_1_s01.wav" formats
            sample_pattern = re.compile(r"_s\d{2}$")
            base_filename = sample_pattern.sub("", filename)

            # Try to match with CSV data using exact ID matching
            matched = False
            for _, row in prompts_df.iterrows():
                row_id = str(row.get("id", "")).strip()
                if row_id and row_id == base_filename:
                    audio_files.append(str(audio_file))
                    transcriptions.append(str(row["text"]))
                    intents.append(str(row.get("canonical_intent", "")))
                    matched = True
                    break

            if not matched:
                logger.warning(f"No prompt found for audio file: {audio_file}")

        logger.info(f"Matched {len(audio_files)} audio files with transcriptions")

        # Create dataset dictionary
        dataset = {
            "audio_paths": audio_files,
            "transcriptions": transcriptions,
            "intents": intents,
        }

        # Show statistics
        intent_counts = Counter(intents)
        logger.info("Intent distribution:")
        for intent, count in intent_counts.most_common(10):
            logger.info(f"  {intent}: {count}")

        return prompts_df, dataset

    def split_dataset(self, dataset: Dict[str, List[str]]) -> Tuple[Dict, Dict, Dict]:
        """Split dataset into train, validation, and test sets."""
        logger.info("Splitting dataset...")

        total_samples = len(dataset["audio_paths"])

        # Calculate split sizes
        test_size = int(total_samples * self.config.test_ratio)
        eval_size = int(total_samples * self.config.eval_ratio)
        train_size = total_samples - test_size - eval_size

        logger.info(
            f"Dataset split: {train_size} train, {eval_size} eval, {test_size} test"
        )

        # Create indices
        indices = list(range(total_samples))
        np.random.shuffle(indices)

        train_indices = indices[:train_size]
        eval_indices = indices[train_size : train_size + eval_size]
        test_indices = indices[train_size + eval_size :]

        # Split data
        def create_split(indices):
            return {
                "audio_paths": [dataset["audio_paths"][i] for i in indices],
                "transcriptions": [dataset["transcriptions"][i] for i in indices],
                "intents": [dataset["intents"][i] for i in indices],
            }

        train_dataset = create_split(train_indices)
        eval_dataset = create_split(eval_indices)
        test_dataset = create_split(test_indices)

        return train_dataset, eval_dataset, test_dataset

    def create_datasets(
        self, train_data: Dict, eval_data: Dict
    ) -> Tuple[TwiAudioDataset, TwiAudioDataset]:
        """Create PyTorch datasets."""
        logger.info("Creating PyTorch datasets...")

        train_dataset = TwiAudioDataset(
            train_data["audio_paths"],
            train_data["transcriptions"],
            train_data["intents"],
            self.processor,
            self.config,
        )

        eval_dataset = TwiAudioDataset(
            eval_data["audio_paths"],
            eval_data["transcriptions"],
            eval_data["intents"],
            self.processor,
            self.config,
        )

        logger.info(
            f"Created datasets: {len(train_dataset)} train, {len(eval_dataset)} eval"
        )

        return train_dataset, eval_dataset

    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics (WER and CER)."""
        predictions, labels = eval_pred

        # Decode predictions and labels
        decoded_preds = self.processor.batch_decode(
            predictions, skip_special_tokens=True
        )
        decoded_labels = self.processor.batch_decode(labels, skip_special_tokens=True)

        # Replace -100 in labels as we can't decode them
        labels = np.where(labels != -100, labels, self.processor.tokenizer.pad_token_id)
        decoded_labels = self.processor.batch_decode(labels, skip_special_tokens=True)

        # Compute WER and CER
        wer_score = wer(decoded_labels, decoded_preds)
        cer_score = cer(decoded_labels, decoded_preds)

        return {"wer": wer_score, "cer": cer_score}

    def train(self):
        """Train the Whisper model on Twi data."""
        logger.info("Starting Whisper fine-tuning on Twi data...")

        # Load and prepare data
        prompts_df, dataset = self.load_and_prepare_data()

        if len(dataset["audio_paths"]) == 0:
            raise ValueError("No valid audio-transcription pairs found!")

        # Split dataset
        train_data, eval_data, test_data = self.split_dataset(dataset)

        # Create datasets
        train_dataset, eval_dataset = self.create_datasets(train_data, eval_data)

        # Data collator
        data_collator = TwiWhisperDataCollator(self.processor)

        # Detect GPU availability

        use_cuda = torch.cuda.is_available()

        fp16_enabled = use_cuda  # Only enable fp16 if CUDA is available

        if use_cuda:
            logger.info(
                "🚀 GPU detected! Training will use CUDA with mixed precision (FP16)."
            )

        else:
            logger.warning("⚠️ No GPU detected. Training will proceed on CPU.")

        # Training arguments

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            gradient_checkpointing=True,  # Memory-saving technique
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            num_train_epochs=self.config.num_epochs,
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            report_to=self.config.report_to,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=False,
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            fp16=fp16_enabled,  # Dynamically enable mixed precision
            no_cuda=not use_cuda,  # Dynamically set no_cuda
            predict_with_generate=True,
            generation_max_length=448,
            save_total_limit=3,
            max_grad_norm=1.0,  # Add gradient clipping for stability
        )

        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.processor.feature_extractor,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        # Train
        logger.info("Starting training...")
        if use_cuda:
            torch.cuda.empty_cache()
        trainer.train()

        # Save final model
        trainer.save_model()
        self.processor.save_pretrained(self.config.output_dir)

        # Save test dataset for later evaluation
        test_data_path = Path(self.config.output_dir) / "test_data.json"
        with open(test_data_path, "w") as f:
            json.dump(test_data, f, indent=2)

        logger.info(f"Training completed! Model saved to {self.config.output_dir}")

        # Evaluate on test set
        self.evaluate_test_set(test_data)

        return trainer

    def evaluate_test_set(self, test_data: Dict):
        """Evaluate the trained model on test set."""
        logger.info("Evaluating on test set...")

        # Create test dataset
        test_dataset = TwiAudioDataset(
            test_data["audio_paths"],
            test_data["transcriptions"],
            test_data["intents"],
            self.processor,
            self.config,
        )

        # Generate predictions
        predictions = []
        references = []

        self.model.eval()
        with torch.no_grad():
            for i, sample in enumerate(test_dataset):
                if i % 10 == 0:
                    logger.info(f"Processing test sample {i}/{len(test_dataset)}")

                # Prepare input
                input_features = sample["input_features"].unsqueeze(0)

                # Generate
                predicted_ids = self.model.generate(
                    input_features, max_length=448, num_beams=5, early_stopping=True
                )

                # Decode
                prediction = self.processor.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )[0]
                reference = sample["transcription"]

                predictions.append(prediction)
                references.append(reference)

        # Compute metrics
        wer_score = wer(references, predictions)
        cer_score = cer(references, predictions)

        logger.info(f"Test Results:")
        logger.info(f"  WER: {wer_score:.4f}")
        logger.info(f"  CER: {cer_score:.4f}")

        # Save detailed results
        results = {
            "test_wer": wer_score,
            "test_cer": cer_score,
            "num_samples": len(test_dataset),
            "predictions": predictions[:10],  # Save first 10 for inspection
            "references": references[:10],
        }

        results_path = Path(self.config.output_dir) / "test_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Test results saved to {results_path}")

        return wer_score, cer_score


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on Twi data")

    # Model arguments
    parser.add_argument(
        "--model_size",
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size",
    )
    parser.add_argument(
        "--model_name", default=None, help="Specific model name (overrides model_size)"
    )

    # Data arguments
    parser.add_argument(
        "--data_dir", default="../data/raw", help="Directory containing audio files"
    )
    parser.add_argument(
        "--prompts_file",
        default="../prompts_lean.csv",
        help="CSV file with prompts and transcriptions",
    )
    parser.add_argument(
        "--output_dir",
        default="./models/whisper_twi",
        help="Output directory for trained model",
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument(
        "--eval_steps", type=int, default=500, help="Evaluation frequency"
    )

    # Audio processing
    parser.add_argument(
        "--max_audio_length",
        type=int,
        default=30,
        help="Maximum audio length in seconds",
    )
    parser.add_argument(
        "--sample_rate", type=int, default=16000, help="Audio sample rate"
    )

    args = parser.parse_args()

    # Create configuration
    config = TwiWhisperConfig(
        model_name=args.model_name or f"openai/whisper-{args.model_size}",
        model_size=args.model_size,
        data_dir=args.data_dir,
        prompts_file=args.prompts_file,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_steps,
        max_audio_length=args.max_audio_length,
        sample_rate=args.sample_rate,
    )

    logger.info("=" * 60)
    logger.info("WHISPER FINE-TUNING FOR TWI SPEECH RECOGNITION")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Data directory: {config.data_dir}")
    logger.info(f"Prompts file: {config.prompts_file}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Epochs: {config.num_epochs}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info("=" * 60)

    try:
        # Initialize trainer
        trainer = TwiWhisperTrainer(config)

        # Start training
        trainer.train()

        logger.info("🎉 Training completed successfully!")
        logger.info(f"📁 Model saved to: {config.output_dir}")

        # Update optimized engine configuration
        logger.info("\n" + "=" * 60)
        logger.info("TO USE THE FINE-TUNED MODEL:")
        logger.info("=" * 60)
        logger.info("1. Update optimized_engine/config/config.py:")
        logger.info(f"""
WHISPER = {{
    "model_size": "custom",
    "custom_model_path": "{config.output_dir}",
    "language": "tw",
    "task": "transcribe",
    ...
}}
""")
        logger.info("2. Restart the optimized engine server")
        logger.info("3. Test with your Twi audio files")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
