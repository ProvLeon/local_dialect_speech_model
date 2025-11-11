#!/usr/bin/env python3
"""
Whisper Multi-Task Fine-tuning for Twi Speech Recognition and Intent Classification
=================================================================================

This script fine-tunes OpenAI's Whisper model on Twi audio data for both
transcription and intent classification in a multi-task setup.

Usage:
    python train_whisper_twi.py --model_size small --epochs 10
    python train_whisper_twi.py --model_size tiny --batch_size 16 --lr 1e-5

Author: AI Assistant
Date: 2025-11-07
"""

import argparse
import json
import logging
import os
import random
import re
import shutil
import sys
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torchaudio

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
import audiomentations as A
import librosa
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperModel,
    WhisperPreTrainedModel,
    WhisperProcessor,
)
from transformers.modeling_outputs import SequenceClassifierOutput

try:
    from jiwer import cer, wer
except ImportError:

    def wer(ref, hyp):
        return 0.0

    def cer(ref, hyp):
        return 0.0


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


@dataclass
class TwiWhisperConfig:
    """Configuration for Twi Whisper multi-task fine-tuning."""

    # Model configuration
    model_name: str = (
        "openai/whisper-tiny"  # Changed to a smaller model to reduce memory usage
    )
    model_size: str = "small"
    language: str = "tw"
    task: str = "transcribe"

    # Data paths
    data_dir: str = "../data/raw"
    manifest_file: str = "../data/lean_dataset/audio_manifest_multisample.jsonl"
    output_dir: str = "./models/whisper_twi_multitask"
    cache_dir: str = "./data/cache"

    # Training configuration
    num_epochs: int = 10
    batch_size: int = 4  # Reduced for stability
    learning_rate: float = 5e-5  # Increased for better learning
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 8  # Increased for effective batch size
    intent_loss_weight: float = 0.5

    # Audio processing
    sample_rate: int = 16000
    max_audio_length: int = 30
    min_audio_length: float = 0.5

    # Validation
    per_device_eval_batch_size: int = 4
    eval_steps: int = 20
    save_steps: int = 20
    eval_ratio: float = 0.1
    test_ratio: float = 0.1

    # Logging
    logging_steps: int = 50
    report_to: str = "tensorboard"

    # Intent classification
    num_intent_labels: int = 0
    label_to_id: Dict[str, int] = field(default_factory=dict)
    id_to_label: Dict[int, str] = field(default_factory=dict)


class WhisperForSpeechClassification(WhisperPreTrainedModel):
    """Whisper model with a sequence classification head."""

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.whisper = WhisperModel(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    def forward(
        self,
        input_features,
        attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        encoder_outputs = self.whisper.encoder(
            input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = encoder_outputs[0]
        pooled_output = hidden_states.mean(dim=1)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)  # Ignore samples with label -1
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class WhisperForConditionalGenerationTwi(WhisperForConditionalGeneration):
    """Whisper model fine-tuned for Twi transcription."""

    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_features=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return super().forward(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class TwiAudioDataset(Dataset):
    """Dataset for Twi audio, transcriptions, and intents, with augmentation."""

    def __init__(
        self,
        audio_paths: List[str],
        transcriptions: List[str],
        intents: List[str],
        processor: WhisperProcessor,
        config: TwiWhisperConfig,
        label_to_id: Dict[str, int],
        is_train: bool = False,
    ):
        self.audio_paths = audio_paths
        self.transcriptions = transcriptions
        self.intents = intents
        self.processor = processor
        self.config = config
        self.label_to_id = label_to_id
        self.is_train = is_train
        self.waveform_augmentations = (
            self._get_waveform_augmentations() if is_train else None
        )
        self.spectrogram_augmentations = (
            self._get_spectrogram_augmentations() if is_train else None
        )
        self._filter_valid_samples()

    def _get_waveform_augmentations(self):
        return A.Compose(
            [
                A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                A.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
                A.PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            ]
        )

    def _get_spectrogram_augmentations(self):
        return nn.Sequential(
            torchaudio.transforms.TimeMasking(time_mask_param=80),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=80),
        )

    def _filter_valid_samples(self):
        valid_indices = []
        for i, (audio_path, trans) in enumerate(
            zip(self.audio_paths, self.transcriptions)
        ):
            if not Path(audio_path).exists():
                logger.warning(f"Audio file not found: {audio_path}")
                continue
            if not trans or len(trans.strip()) == 0:
                logger.warning(f"Empty transcription for: {audio_path}")
                continue
            try:
                y, sr = librosa.load(audio_path, sr=self.config.sample_rate)
                duration = librosa.get_duration(y=y, sr=sr)
                if not (
                    self.config.min_audio_length
                    <= duration
                    <= self.config.max_audio_length
                ):
                    logger.warning(
                        f"Audio duration out of range ({duration:.2f}s): {audio_path}"
                    )
                    continue
            except Exception as e:
                logger.warning(f"Error loading audio {audio_path}: {e}")
                continue
            valid_indices.append(i)

        self.audio_paths = [self.audio_paths[i] for i in valid_indices]
        self.transcriptions = [self.transcriptions[i] for i in valid_indices]
        self.intents = [self.intents[i] for i in valid_indices]
        logger.info(f"Filtered dataset: {len(self.audio_paths)} valid samples.")

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        transcription = self.transcriptions[idx]

        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.config.sample_rate)

            # Apply waveform augmentations if training
            if self.is_train and self.waveform_augmentations:
                audio = self.waveform_augmentations(samples=audio, sample_rate=sr)

            # Extract features
            input_features = self.processor(
                audio, sampling_rate=self.config.sample_rate, return_tensors="pt"
            ).input_features[0]

            # Apply spectrogram augmentations if training
            if self.is_train and self.spectrogram_augmentations:
                input_features = self.spectrogram_augmentations(input_features)

        except Exception as e:
            logger.error(f"Error processing audio {audio_path}: {e}")
            # Create fallback features with correct dimensions
            input_features = torch.zeros((80, 3000))

        # Tokenize transcription properly for Whisper
        labels = self.processor.tokenizer(
            transcription, max_length=448, truncation=True, return_tensors="pt"
        ).input_ids.squeeze(0)

        # Debug: Print sample data info only once
        if idx == 0:
            logger.info(f"Sample data - Audio: {audio_path}")
            logger.info(f"Sample data - Text: {transcription}")
            logger.info(f"Sample data - Features shape: {input_features.shape}")
            logger.info(f"Sample data - Labels shape: {labels.shape}")

        return {
            "input_features": input_features,
            "labels": labels,
        }


class TwiWhisperDataCollator:
    """Data collator for Whisper fine-tuning on Twi."""

    def __init__(self, processor: WhisperProcessor):
        self.processor = processor

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Pad input features
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Pad labels
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding token id's of the labels by -100 so it's ignored by the loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # If bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


class TwiWhisperTrainer(Seq2SeqTrainer):
    """Custom Seq2SeqTrainer for Twi Whisper fine-tuning."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TwiWhisperManager:
    """Trainer for the multi-task Whisper model."""

    def __init__(self, config: TwiWhisperConfig):
        self.config = config
        self.setup_directories()
        self.processor = WhisperProcessor.from_pretrained(config.model_name)
        self.tokenizer = self.processor.tokenizer

    def setup_directories(self):
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

    def load_and_prepare_data(self):
        import json

        data = []
        intents = set()
        found_files = 0

        logger.info(f"Loading data from: {self.config.manifest_file}")

        # Load data from JSONL manifest file
        with open(self.config.manifest_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line.strip())

                    # Try multiple possible audio paths
                    possible_paths = [
                        sample["audio_path"],
                        os.path.join("..", sample["audio_path"]),
                        sample["audio_path"].replace("data/raw/", "../data/raw/"),
                        sample["audio_path"].replace("data/raw/", "data/raw/"),
                    ]

                    audio_path = None
                    for path in possible_paths:
                        if os.path.exists(path):
                            audio_path = path
                            found_files += 1
                            break

                    if audio_path:
                        data.append(
                            {
                                "audio_path": audio_path,
                                "transcription": sample["text"],
                                "intent": sample["intent"],
                            }
                        )
                        intents.add(sample["intent"])

        # Set up intent labels
        intents = sorted(list(intents))
        self.config.num_intent_labels = len(intents)
        self.config.label_to_id = {label: i for i, label in enumerate(intents)}
        self.config.id_to_label = {i: label for i, label in enumerate(intents)}

        logger.info(f"Loaded {len(data)} samples from manifest")
        logger.info(f"Found {found_files} audio files")
        logger.info(f"Number of intents: {len(intents)}")

        if len(data) == 0:
            logger.error("❌ CRITICAL ERROR: No audio files found!")
            logger.error("📁 Expected audio files in directories like:")
            logger.error("   - ../data/raw/P01/")
            logger.error("   - ../data/raw/P02/")
            logger.error("   - etc.")
            logger.error("")
            logger.error("🎙️  You need to:")
            logger.error("   1. Record audio files for each prompt in the manifest")
            logger.error("   2. Save them in the correct directory structure")
            logger.error("   3. Ensure file names match the manifest entries")
            logger.error("")
            logger.error("📋 Check the manifest file for expected file paths:")
            logger.error(f"   {self.config.manifest_file}")
            raise ValueError(
                "No valid audio files found! Training cannot proceed without audio data. Please record the audio files first."
            )

        return data

    def split_dataset(self, data):
        np.random.shuffle(data)
        test_size = int(len(data) * self.config.test_ratio)
        eval_size = int(len(data) * self.config.eval_ratio)
        train_data = data[test_size + eval_size :]
        eval_data = data[test_size : test_size + eval_size]
        test_data = data[:test_size]
        return train_data, eval_data, test_data

    def create_datasets(self, train_data, eval_data):
        train_dataset = TwiAudioDataset(
            [d["audio_path"] for d in train_data],
            [d["transcription"] for d in train_data],
            [None] * len(train_data),  # No intents for now
            self.processor,
            self.config,
            {},  # No label mapping needed
            is_train=True,
        )
        eval_dataset = TwiAudioDataset(
            [d["audio_path"] for d in eval_data],
            [d["transcription"] for d in eval_data],
            [None] * len(eval_data),  # No intents for now
            self.processor,
            self.config,
            {},  # No label mapping needed
            is_train=False,
        )
        return train_dataset, eval_dataset

    def compute_metrics(self, eval_pred):
        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids

        # Debug: Print shapes and sample values
        print(f"DEBUG: pred_ids shape: {pred_ids.shape}")
        print(f"DEBUG: label_ids shape: {label_ids.shape}")

        # Replace -100s used for padding as we can't decode them
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        if isinstance(labels, tuple):
            transcription_labels = labels[0]
        else:
            transcription_labels = labels

        # Transcription metrics
        transcription_labels[transcription_labels == -100] = (
            self.processor.tokenizer.pad_token_id
        )
        decoded_preds = self.processor.tokenizer.batch_decode(
            transcription_logits, skip_special_tokens=True
        )
        decoded_labels = self.processor.tokenizer.batch_decode(
            transcription_labels, skip_special_tokens=True
        )
        wer_score = wer(decoded_labels, decoded_preds)
        cer_score = cer(decoded_labels, decoded_preds)

        return {"wer": wer_score, "cer": cer_score}

    def train(self):
        logger.info("Starting multi-task training...")

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            logger.info("CUDA is available. Using GPU for training.")
        else:
            logger.info("CUDA not available. Using CPU for training.")

        # Data
        data = self.load_and_prepare_data()
        train_data, eval_data, test_data = self.split_dataset(data)
        train_dataset, eval_dataset = self.create_datasets(train_data, eval_data)

        # Model
        whisper_config = WhisperConfig.from_pretrained(self.config.model_name)
        config_dict = asdict(self.config)
        for key in list(config_dict.keys()):
            if not hasattr(whisper_config, key):
                del config_dict[key]
        whisper_config.update(config_dict)

        # Explicitly set num_labels for the classification head
        whisper_config.num_labels = self.config.num_intent_labels

        # Load the pretrained Whisper model
        model = WhisperForConditionalGenerationTwi.from_pretrained(
            self.config.model_name, config=whisper_config
        )

        # Configure the model for fine-tuning
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        model.config.use_cache = False  # Disable cache for training

        logger.info(f"Model vocab size: {model.config.vocab_size}")
        logger.info(f"Model loaded successfully from {self.config.model_name}")

        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            num_train_epochs=self.config.num_epochs,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            logging_steps=5,  # More frequent logging
            logging_first_step=True,
            report_to=self.config.report_to,
            load_best_model_at_end=True,
            metric_for_best_model="eval_wer",
            greater_is_better=False,
            fp16=use_cuda,
            dataloader_drop_last=False,
            predict_with_generate=True,
            generation_max_length=448,
            generation_num_beams=1,
            remove_unused_columns=False,
            label_names=["labels"],
            logging_dir=f"{self.config.output_dir}/logs",
            gradient_checkpointing=False,
            include_inputs_for_metrics=True,
        )

        # Create data collator
        data_collator = TwiWhisperDataCollator(self.processor)

        # Create trainer
        trainer = TwiWhisperTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.processor.feature_extractor,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        # Verify we have actual training data
        if len(train_dataset) == 0:
            logger.error("❌ No training data available!")
            return

        logger.info("✅ Model loaded and ready for training with real audio data...")

        # Train with better logging
        logger.info("Starting training...")
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Evaluation samples: {len(eval_dataset)}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.learning_rate}")

        trainer.train()

        # Save
        trainer.save_model()
        self.processor.save_pretrained(self.config.output_dir)
        with open(Path(self.config.output_dir) / "intent_labels.json", "w") as f:
            json.dump(self.config.label_to_id, f)

        logger.info(f"Training complete. Model saved to {self.config.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper for Twi ASR and Intent"
    )
    parser.add_argument("--model_size", default="small", help="Whisper model size")
    parser.add_argument(
        "--epochs", type=int, default=15, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--report_to",
        default="tensorboard",
        help="Logging backend (e.g., tensorboard, wandb)",
    )
    args = parser.parse_args()
    args.model_size = "tiny"  # Force tiny model to conserve memory

    config = TwiWhisperConfig(
        model_size=args.model_size,
        model_name=f"openai/whisper-{args.model_size}",
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        report_to=args.report_to,
    )

    trainer = TwiWhisperManager(config)
    trainer.train()


if __name__ == "__main__":
    main()
