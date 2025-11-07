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

import os
import sys
import json
import logging
import argparse
import re
import shutil
import warnings
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

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
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset

import librosa
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    WhisperPreTrainedModel,
    WhisperModel,
    WhisperConfig,
)
from transformers.modeling_outputs import SequenceClassifierOutput

try:
    from jiwer import wer, cer
except ImportError:
    def wer(ref, hyp): return 0.0
    def cer(ref, hyp): return 0.0

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
    model_name: str = "openai/whisper-small"
    model_size: str = "small"
    language: str = "tw"
    task: str = "transcribe"

    # Data paths
    data_dir: str = "../data/raw"
    prompts_file: str = "../prompts_lean.csv"
    output_dir: str = "./models/whisper_twi_multitask"
    cache_dir: str = "./data/cache"

    # Training configuration
    num_epochs: int = 15
    batch_size: int = 8
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 2
    intent_loss_weight: float = 0.5

    # Audio processing
    sample_rate: int = 16000
    max_audio_length: int = 30
    min_audio_length: float = 0.5

    # Validation
    eval_steps: int = 200
    save_steps: int = 400
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
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
            loss_fct = CrossEntropyLoss()
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

class WhisperForMultiTask(WhisperPreTrainedModel):
    """A multi-task model combining transcription and classification."""
    def __init__(self, config):
        super().__init__(config)
        self.transcription_model = WhisperForConditionalGeneration(config)
        self.classification_model = WhisperForSpeechClassification(config)

    def forward(
        self,
        input_features,
        labels=None,
        intent_labels=None,
        **kwargs,
    ):
        transcription_output = self.transcription_model(
            input_features=input_features,
            labels=labels,
            **kwargs,
        )
        classification_output = self.classification_model(
            input_features=input_features,
            labels=intent_labels,
        )
        return {
            "transcription_loss": transcription_output.loss,
            "transcription_logits": transcription_output.logits,
            "classification_loss": classification_output.loss,
            "classification_logits": classification_output.logits,
        }


class TwiAudioDataset(Dataset):
    """Dataset for Twi audio, transcriptions, and intents."""
    def __init__(
        self,
        audio_paths: List[str],
        transcriptions: List[str],
        intents: List[str],
        processor: WhisperProcessor,
        config: TwiWhisperConfig,
        label_to_id: Dict[str, int],
    ):
        self.audio_paths = audio_paths
        self.transcriptions = transcriptions
        self.intents = intents
        self.processor = processor
        self.config = config
        self.label_to_id = label_to_id
        self._filter_valid_samples()

    def _filter_valid_samples(self):
        valid_indices = []
        for i, (audio_path, trans) in enumerate(zip(self.audio_paths, self.transcriptions)):
            if not Path(audio_path).exists():
                logger.warning(f"Audio file not found: {audio_path}")
                continue
            if not trans or len(trans.strip()) == 0:
                logger.warning(f"Empty transcription for: {audio_path}")
                continue
            try:
                y, sr = librosa.load(audio_path, sr=self.config.sample_rate)
                duration = librosa.get_duration(y=y, sr=sr)
                if not (self.config.min_audio_length <= duration <= self.config.max_audio_length):
                    logger.warning(f"Audio duration out of range ({duration:.2f}s): {audio_path}")
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
        intent = self.intents[idx]

        try:
            audio, sr = librosa.load(audio_path, sr=self.config.sample_rate)
            audio = self.processor(audio, sampling_rate=self.config.sample_rate).input_features[0]
        except Exception as e:
            logger.error(f"Error processing audio {audio_path}: {e}")
            audio = np.zeros(self.config.sample_rate * 1) # Fallback

        labels = self.processor.tokenizer(transcription).input_ids
        intent_label = self.label_to_id.get(intent, -1)

        return {
            "input_features": audio,
            "labels": labels,
            "intent_labels": intent_label,
        }


class MultiTaskDataCollator:
    """Data collator for multi-task training."""
    def __init__(self, processor: WhisperProcessor):
        self.processor = processor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        if "intent_labels" in features[0]:
            intent_labels = [feature["intent_labels"] for feature in features]
            batch["intent_labels"] = torch.tensor(intent_labels, dtype=torch.long)

        return batch


class MultiTaskTrainer(Seq2SeqTrainer):
    """A Seq2SeqTrainer for multi-task learning."""
    def __init__(self, *args, intent_loss_weight=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.intent_loss_weight = intent_loss_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        transcription_loss = outputs.get("transcription_loss")
        classification_loss = outputs.get("classification_loss")

        total_loss = transcription_loss + self.intent_loss_weight * classification_loss
        return (total_loss, outputs) if return_outputs else total_loss


class TwiWhisperTrainer:
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
        df = pd.read_csv(self.config.prompts_file, on_bad_lines='skip')
        df = df.dropna(subset=["id", "text", "canonical_intent"])
        
        intents = list(df["canonical_intent"].unique())
        self.config.num_intent_labels = len(intents)
        self.config.label_to_id = {label: i for i, label in enumerate(intents)}
        self.config.id_to_label = {i: label for i, label in enumerate(intents)}

        data = []
        data_dir = Path(self.config.data_dir)
        for _, row in df.iterrows():
            audio_id = row['id']
            # Search for the audio file in the subdirectories
            audio_files = list(data_dir.rglob(f"**/{audio_id}.wav")) + list(data_dir.rglob(f"**/{audio_id}.mp3"))
            
            if audio_files:
                audio_path = audio_files[0]
                data.append({
                    "audio_path": str(audio_path),
                    "transcription": row["text"],
                    "intent": row["canonical_intent"],
                })
            else:
                logger.warning(f"Audio file not found for id: {audio_id}")

        return data

    def split_dataset(self, data):
        np.random.shuffle(data)
        test_size = int(len(data) * self.config.test_ratio)
        eval_size = int(len(data) * self.config.eval_ratio)
        train_data = data[test_size + eval_size:]
        eval_data = data[test_size : test_size + eval_size]
        test_data = data[:test_size]
        return train_data, eval_data, test_data

    def create_datasets(self, train_data, eval_data):
        train_dataset = TwiAudioDataset(
            [d["audio_path"] for d in train_data],
            [d["transcription"] for d in train_data],
            [d["intent"] for d in train_data],
            self.processor, self.config, self.config.label_to_id
        )
        eval_dataset = TwiAudioDataset(
            [d["audio_path"] for d in eval_data],
            [d["transcription"] for d in eval_data],
            [d["intent"] for d in eval_data],
            self.processor, self.config, self.config.label_to_id
        )
        return train_dataset, eval_dataset

    def compute_metrics(self, eval_pred):
        transcription_logits, intent_logits = eval_pred.predictions
        transcription_labels, intent_labels = eval_pred.label_ids

        # Transcription metrics
        transcription_labels[transcription_labels == -100] = self.tokenizer.pad_token_id
        decoded_preds = self.tokenizer.batch_decode(transcription_logits, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(transcription_labels, skip_special_tokens=True)
        wer_score = wer(decoded_labels, decoded_preds)
        cer_score = cer(decoded_labels, decoded_preds)

        # Intent metrics
        intent_preds = np.argmax(intent_logits, axis=-1)
        accuracy = (intent_preds == intent_labels).astype(np.float32).mean().item()

        return {"wer": wer_score, "cer": cer_score, "intent_accuracy": accuracy}

    def train(self):
        logger.info("Starting multi-task training...")
        
        # Data
        data = self.load_and_prepare_data()
        train_data, eval_data, test_data = self.split_dataset(data)
        train_dataset, eval_dataset = self.create_datasets(train_data, eval_data)
        data_collator = MultiTaskDataCollator(self.processor)

        # Model
        whisper_config = WhisperConfig.from_pretrained(self.config.model_name)
        config_dict = asdict(self.config)
        for key in list(config_dict.keys()):
            if not hasattr(whisper_config, key):
                del config_dict[key]
        whisper_config.update(config_dict)

        model = WhisperForMultiTask.from_pretrained(
            self.config.model_name,
            config=whisper_config,
            ignore_mismatched_sizes=True,
        )
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []

        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            num_train_epochs=self.config.num_epochs,
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            report_to=self.config.report_to,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            predict_with_generate=True,
            logging_dir=f"{self.config.output_dir}/logs",
        )

        # Trainer
        trainer = MultiTaskTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.processor.feature_extractor,
            compute_metrics=self.compute_metrics,
            intent_loss_weight=self.config.intent_loss_weight,
        )

        # Train
        trainer.train()

        # Save
        trainer.save_model()
        self.processor.save_pretrained(self.config.output_dir)
        with open(Path(self.config.output_dir) / "intent_labels.json", "w") as f:
            json.dump(self.config.label_to_id, f)

        logger.info(f"Training complete. Model saved to {self.config.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper for Twi ASR and Intent")
    parser.add_argument("--model_size", default="small", help="Whisper model size")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--report_to", default="tensorboard", help="Logging backend (e.g., tensorboard, wandb)")
    args = parser.parse_args()

    config = TwiWhisperConfig(
        model_size=args.model_size,
        model_name=f"openai/whisper-{args.model_size}",
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        report_to=args.report_to,
    )

    trainer = TwiWhisperTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
