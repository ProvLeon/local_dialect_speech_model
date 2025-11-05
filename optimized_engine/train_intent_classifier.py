#!/usr/bin/env python3
"""
Intent Classifier Training Script for Twi Speech Recognition
===========================================================

This script trains a custom intent classification model using the Twi prompts
data from your CSV file. It creates a fine-tuned transformer model specifically
for your e-commerce intents.

Author: AI Assistant
Date: 2025-11-05
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import Counter
import argparse

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Add project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(project_root))

from config.config import OptimizedConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TwiIntentTrainer:
    """Trainer for Twi intent classification model."""

    def __init__(self, config: OptimizedConfig = None):
        self.config = config or OptimizedConfig()
        self.tokenizer = None
        self.model = None
        self.label_to_id = {}
        self.id_to_label = {}
        self.device = self.config.get_device()

    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load and preprocess the Twi prompts data."""
        logger.info(f"Loading data from {csv_path}")

        # Read CSV
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from CSV")

        # Filter out rows without intent or text
        df = df.dropna(subset=["text", "intent"])
        df = df[df["intent"].str.strip() != ""]
        df = df[df["text"].str.strip() != ""]

        logger.info(f"After filtering: {len(df)} rows with valid text and intent")

        return df

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Prepare training data from CSV."""
        texts = []
        intents = []

        # Extract text and intent pairs
        for _, row in df.iterrows():
            text = str(row["text"]).strip()
            intent = str(row["intent"]).strip()

            # Skip empty or invalid entries
            if not text or not intent or intent.lower() in ["intent", "nan"]:
                continue

            texts.append(text)
            intents.append(intent)

        logger.info(f"Prepared {len(texts)} training examples")

        # Show intent distribution
        intent_counts = Counter(intents)
        logger.info("Intent distribution:")
        for intent, count in intent_counts.most_common():
            logger.info(f"  {intent}: {count}")

        return texts, intents

    def augment_data(
        self, texts: List[str], intents: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Augment training data with variations."""
        augmented_texts = texts.copy()
        augmented_intents = intents.copy()

        # Simple augmentation strategies for Twi
        augmentations = [
            # Add common Twi variations
            lambda x: x.replace("kɔ", "ko"),  # Alternate spelling
            lambda x: x.replace("ɛ", "e"),  # Accent removal
            lambda x: x.replace("ɔ", "o"),  # Accent removal
            # Add common prefixes/suffixes
            lambda x: f"me pɛ sɛ {x}",  # "I want to..."
            lambda x: f"{x} yi",  # Add particle
            lambda x: f"boa me {x}",  # "Help me..."
        ]

        original_count = len(texts)

        for i, (text, intent) in enumerate(zip(texts, intents)):
            # Apply augmentations randomly
            for aug_func in augmentations:
                try:
                    augmented_text = aug_func(text)
                    if augmented_text != text and len(augmented_text) > 0:
                        augmented_texts.append(augmented_text)
                        augmented_intents.append(intent)
                except:
                    continue

        logger.info(
            f"Augmented data from {original_count} to {len(augmented_texts)} examples"
        )
        return augmented_texts, augmented_intents

    def create_label_mappings(self, intents: List[str]):
        """Create label to ID mappings."""
        unique_intents = sorted(set(intents))
        self.label_to_id = {intent: idx for idx, intent in enumerate(unique_intents)}
        self.id_to_label = {idx: intent for intent, idx in self.label_to_id.items()}

        logger.info(f"Created mappings for {len(unique_intents)} unique intents")
        return unique_intents

    def prepare_datasets(self, texts: List[str], intents: List[str], test_size=0.2):
        """Prepare train/validation datasets."""
        # Convert intents to IDs
        intent_ids = [self.label_to_id[intent] for intent in intents]

        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, intent_ids, test_size=test_size, random_state=42, stratify=intent_ids
        )

        logger.info(
            f"Split data: {len(train_texts)} train, {len(val_texts)} validation"
        )

        # Tokenize data
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], truncation=True, padding=True, max_length=512
            )

        # Create datasets
        train_dataset = Dataset.from_dict({"text": train_texts, "labels": train_labels})

        val_dataset = Dataset.from_dict({"text": val_texts, "labels": val_labels})

        # Tokenize
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)

        return train_dataset, val_dataset

    def initialize_model(self, num_labels: int):
        """Initialize tokenizer and model."""
        model_name = "microsoft/DialoGPT-medium"  # Good for conversational AI

        logger.info(f"Initializing model: {model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, ignore_mismatched_sizes=True
        )

        # Move to device
        self.model.to(self.device)

        logger.info(f"Model initialized with {num_labels} labels on {self.device}")

    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(labels, predictions)

        return {
            "accuracy": accuracy,
            "f1": accuracy,  # Simplified for now
        }

    def train_model(self, train_dataset, val_dataset, output_dir: str):
        """Train the intent classification model."""
        logger.info("Starting model training...")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=10,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=100,
            save_steps=500,
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
        )

        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True)

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        # Train
        trainer.train()

        # Save model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save label mappings
        label_path = Path(output_dir) / "intent_labels.json"
        with open(label_path, "w") as f:
            json.dump(
                {"label_to_id": self.label_to_id, "id_to_label": self.id_to_label},
                f,
                indent=2,
            )

        logger.info(f"Model saved to {output_dir}")

        return trainer

    def evaluate_model(self, trainer, val_dataset):
        """Evaluate the trained model."""
        logger.info("Evaluating model...")

        # Evaluate
        eval_results = trainer.evaluate()

        logger.info("Evaluation results:")
        for key, value in eval_results.items():
            logger.info(f"  {key}: {value:.4f}")

        # Predictions for detailed analysis
        predictions = trainer.predict(val_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids

        # Convert back to labels
        pred_labels = [self.id_to_label[idx] for idx in y_pred]
        true_labels = [self.id_to_label[idx] for idx in y_true]

        # Classification report
        report = classification_report(true_labels, pred_labels)
        logger.info(f"Classification Report:\n{report}")

        return eval_results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Twi Intent Classifier")
    parser.add_argument(
        "--data", default="../twi_prompts.csv", help="Path to Twi prompts CSV file"
    )
    parser.add_argument(
        "--output",
        default="./models/intent_classifier",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--augment", action="store_true", help="Enable data augmentation"
    )

    args = parser.parse_args()

    # Initialize trainer
    config = OptimizedConfig()
    trainer = TwiIntentTrainer(config)

    # Check if data file exists
    data_path = Path(args.data)
    if not data_path.exists():
        # Try alternative paths
        alternative_paths = [
            Path("../twi_prompts.csv"),
            Path("../../twi_prompts.csv"),
            project_root / "twi_prompts.csv",
        ]

        for alt_path in alternative_paths:
            if alt_path.exists():
                data_path = alt_path
                break
        else:
            logger.error(f"Could not find data file. Tried: {args.data}")
            logger.error(f"Alternative paths: {[str(p) for p in alternative_paths]}")
            return

    logger.info(f"Using data file: {data_path}")

    try:
        # Load data
        df = trainer.load_data(str(data_path))

        # Prepare training data
        texts, intents = trainer.prepare_training_data(df)

        if len(texts) == 0:
            logger.error("No training data found!")
            return

        # Augment data if requested
        if args.augment:
            texts, intents = trainer.augment_data(texts, intents)

        # Create label mappings
        unique_intents = trainer.create_label_mappings(intents)

        # Initialize model
        trainer.initialize_model(len(unique_intents))

        # Prepare datasets
        train_dataset, val_dataset = trainer.prepare_datasets(texts, intents)

        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Train model
        model_trainer = trainer.train_model(train_dataset, val_dataset, str(output_dir))

        # Evaluate model
        trainer.evaluate_model(model_trainer, val_dataset)

        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {output_dir}")

        # Update config to use trained model
        config_update = f"""
# Update your config.py with:
INTENT_CLASSIFIER = {{
    "custom_model_path": "{output_dir}",
    "confidence_threshold": 0.5,
    "top_k": 3,
}}
"""
        logger.info(config_update)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
