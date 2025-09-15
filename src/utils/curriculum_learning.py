#!/usr/bin/env python3
"""
Curriculum Learning Strategy for Gradual Class Introduction in Speech Intent Classification

This module implements a sophisticated curriculum learning approach that gradually introduces
all 47 classes while preventing overfitting and ensuring no classes are lost during training.

Key Features:
- Progressive class introduction based on difficulty/rarity
- Adaptive learning rate scheduling per curriculum stage
- Class-specific loss weighting that evolves over training
- Memory replay to prevent catastrophic forgetting
- Validation-driven curriculum progression
- Multi-stage training with increasing complexity

Strategy:
1. Stage 1: Train on most common classes (easy curriculum)
2. Stage 2: Gradually add moderate classes
3. Stage 3: Introduce rare classes with memory replay
4. Stage 4: Fine-tune on full dataset with class balancing
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Any, Set
import logging
import json
import os
from copy import deepcopy
import random
from sklearn.metrics import f1_score, accuracy_score

logger = logging.getLogger(__name__)


class CurriculumStage:
    """Represents a single stage in the curriculum learning process."""

    def __init__(self,
                 stage_id: int,
                 classes: List[str],
                 max_epochs: int,
                 base_lr: float,
                 class_weights: Optional[Dict[str, float]] = None,
                 memory_replay_ratio: float = 0.0,
                 validation_threshold: float = 0.7):
        """
        Initialize a curriculum stage.

        Args:
            stage_id: Unique identifier for this stage
            classes: List of classes to include in this stage
            max_epochs: Maximum epochs for this stage
            base_lr: Base learning rate for this stage
            class_weights: Optional class weights for loss computation
            memory_replay_ratio: Ratio of previous stage samples to replay
            validation_threshold: Accuracy threshold to advance to next stage
        """
        self.stage_id = stage_id
        self.classes = set(classes)
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.class_weights = class_weights or {}
        self.memory_replay_ratio = memory_replay_ratio
        self.validation_threshold = validation_threshold

        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        self.completed = False


class MemoryBank:
    """Memory bank for storing samples from previous curriculum stages."""

    def __init__(self, max_samples_per_class: int = 50):
        """
        Initialize memory bank.

        Args:
            max_samples_per_class: Maximum samples to store per class
        """
        self.max_samples_per_class = max_samples_per_class
        self.samples = defaultdict(list)
        self.labels = defaultdict(list)
        self.slots = defaultdict(list)

    def add_samples(self,
                   features: List[np.ndarray],
                   labels: List[str],
                   slots: List[Dict[str, Any]],
                   class_subset: Set[str]):
        """
        Add samples to memory bank for specified classes.

        Args:
            features: List of feature arrays
            labels: List of class labels
            slots: List of slot dictionaries
            class_subset: Set of classes to store in memory
        """
        for feat, label, slot in zip(features, labels, slots):
            if label in class_subset:
                if len(self.samples[label]) < self.max_samples_per_class:
                    self.samples[label].append(feat)
                    self.labels[label].append(label)
                    self.slots[label].append(slot)
                else:
                    # Replace random sample to maintain diversity
                    idx = random.randint(0, self.max_samples_per_class - 1)
                    self.samples[label][idx] = feat
                    self.labels[label][idx] = label
                    self.slots[label][idx] = slot

    def get_replay_samples(self,
                          current_classes: Set[str],
                          replay_ratio: float) -> Tuple[List[np.ndarray], List[str], List[Dict[str, Any]]]:
        """
        Get samples for memory replay.

        Args:
            current_classes: Current stage classes
            replay_ratio: Ratio of replay samples to current samples

        Returns:
            Tuple of (replay_features, replay_labels, replay_slots)
        """
        replay_features = []
        replay_labels = []
        replay_slots = []

        # Get all stored classes except current ones
        memory_classes = set(self.samples.keys()) - current_classes

        if not memory_classes or replay_ratio <= 0:
            return replay_features, replay_labels, replay_slots

        # Calculate number of replay samples needed
        total_current_samples = sum(len(self.samples[cls]) for cls in current_classes if cls in self.samples)
        num_replay_samples = int(total_current_samples * replay_ratio)

        if num_replay_samples == 0:
            return replay_features, replay_labels, replay_slots

        # Sample proportionally from memory classes
        samples_per_class = max(1, num_replay_samples // len(memory_classes))

        for cls in memory_classes:
            if cls in self.samples:
                # Sample up to samples_per_class from this class
                available_samples = len(self.samples[cls])
                num_to_sample = min(samples_per_class, available_samples)

                indices = random.sample(range(available_samples), num_to_sample)

                for idx in indices:
                    replay_features.append(self.samples[cls][idx])
                    replay_labels.append(self.labels[cls][idx])
                    replay_slots.append(self.slots[cls][idx])

        logger.info(f"Memory replay: {len(replay_features)} samples from {len(memory_classes)} classes")
        return replay_features, replay_labels, replay_slots


class CurriculumPlanner:
    """Plans the curriculum stages based on class difficulty and frequency."""

    def __init__(self,
                 features: List[np.ndarray],
                 labels: List[str],
                 num_stages: int = 4,
                 base_epochs_per_stage: int = 20,
                 base_lr: float = 0.001):
        """
        Initialize curriculum planner.

        Args:
            features: List of feature arrays
            labels: List of class labels
            num_stages: Number of curriculum stages
            base_epochs_per_stage: Base epochs per stage
            base_lr: Base learning rate
        """
        self.features = features
        self.labels = labels
        self.num_stages = num_stages
        self.base_epochs_per_stage = base_epochs_per_stage
        self.base_lr = base_lr

        self.class_counts = Counter(labels)
        self.unique_classes = list(self.class_counts.keys())
        self.stages = []

    def compute_class_difficulty(self) -> Dict[str, float]:
        """
        Compute difficulty score for each class based on frequency and other factors.

        Returns:
            Dictionary mapping class names to difficulty scores (0-1, higher = more difficult)
        """
        difficulties = {}

        # Frequency-based difficulty (rare classes are harder)
        min_count = min(self.class_counts.values())
        max_count = max(self.class_counts.values())

        for class_name, count in self.class_counts.items():
            # Inverse frequency normalized to 0-1
            freq_difficulty = 1.0 - (count - min_count) / (max_count - min_count) if max_count > min_count else 0.0

            # Add slight randomness to break ties
            noise = random.uniform(-0.05, 0.05)

            difficulties[class_name] = max(0.0, min(1.0, freq_difficulty + noise))

        return difficulties

    def create_curriculum_stages(self) -> List[CurriculumStage]:
        """
        Create curriculum stages with progressive difficulty.

        Returns:
            List of curriculum stages
        """
        difficulties = self.compute_class_difficulty()

        # Sort classes by difficulty (easy to hard)
        sorted_classes = sorted(self.unique_classes, key=lambda x: difficulties[x])

        # Divide classes into stages
        classes_per_stage = len(sorted_classes) // self.num_stages
        remainder = len(sorted_classes) % self.num_stages

        stages = []
        current_pos = 0

        for stage_id in range(self.num_stages):
            # Cumulative class assignment (each stage includes previous classes)
            if stage_id == 0:
                # Stage 1: Start with easiest classes
                stage_size = classes_per_stage + (1 if stage_id < remainder else 0)
                stage_classes = sorted_classes[:stage_size]
            else:
                # Subsequent stages: Add more classes cumulatively
                prev_stage_size = len(stages[-1].classes)
                additional_classes = classes_per_stage + (1 if stage_id < remainder else 0)

                # Include all previous classes plus new ones
                end_pos = min(len(sorted_classes), prev_stage_size + additional_classes)
                stage_classes = sorted_classes[:end_pos]

            # Compute adaptive parameters for this stage
            stage_epochs = self.base_epochs_per_stage
            stage_lr = self.base_lr * (0.8 ** stage_id)  # Decay learning rate

            # Memory replay increases with stage complexity
            memory_replay_ratio = 0.0 if stage_id == 0 else min(0.3, 0.1 * stage_id)

            # Validation threshold decreases for harder stages
            val_threshold = max(0.5, 0.8 - 0.1 * stage_id)

            # Compute class weights based on frequency
            stage_class_weights = {}
            for cls in stage_classes:
                # Inverse frequency weighting
                weight = 1.0 / self.class_counts[cls]
                stage_class_weights[cls] = weight

            # Normalize weights
            total_weight = sum(stage_class_weights.values())
            stage_class_weights = {k: v/total_weight * len(stage_class_weights)
                                 for k, v in stage_class_weights.items()}

            stage = CurriculumStage(
                stage_id=stage_id,
                classes=stage_classes,
                max_epochs=stage_epochs,
                base_lr=stage_lr,
                class_weights=stage_class_weights,
                memory_replay_ratio=memory_replay_ratio,
                validation_threshold=val_threshold
            )

            stages.append(stage)

        # Log curriculum plan
        logger.info("Curriculum Learning Plan:")
        for i, stage in enumerate(stages):
            logger.info(f"  Stage {i}: {len(stage.classes)} classes, "
                       f"{stage.max_epochs} epochs, lr={stage.base_lr:.4f}, "
                       f"replay={stage.memory_replay_ratio:.2f}")
            logger.info(f"    Classes: {sorted(list(stage.classes))}")

        return stages


class CurriculumDataLoader:
    """Custom data loader for curriculum learning with memory replay."""

    def __init__(self,
                 features: List[np.ndarray],
                 labels: List[str],
                 slots: List[Dict[str, Any]],
                 label_to_idx: Dict[str, int],
                 batch_size: int = 32):
        """
        Initialize curriculum data loader.

        Args:
            features: List of feature arrays
            labels: List of class labels
            slots: List of slot dictionaries
            label_to_idx: Mapping from labels to indices
            batch_size: Batch size for data loading
        """
        self.features = features
        self.labels = labels
        self.slots = slots
        self.label_to_idx = label_to_idx
        self.batch_size = batch_size

    def create_stage_dataloader(self,
                              stage: CurriculumStage,
                              memory_bank: MemoryBank,
                              split: str = 'train') -> DataLoader:
        """
        Create data loader for a specific curriculum stage.

        Args:
            stage: Curriculum stage
            memory_bank: Memory bank for replay samples
            split: Data split ('train' or 'val')

        Returns:
            DataLoader for the stage
        """
        # Filter samples for current stage classes
        stage_indices = []
        for i, label in enumerate(self.labels):
            if label in stage.classes:
                stage_indices.append(i)

        stage_features = [self.features[i] for i in stage_indices]
        stage_labels = [self.labels[i] for i in stage_indices]
        stage_slots = [self.slots[i] for i in stage_indices]

        # Add memory replay samples if needed
        if stage.memory_replay_ratio > 0:
            replay_features, replay_labels, replay_slots = memory_bank.get_replay_samples(
                stage.classes, stage.memory_replay_ratio
            )
            stage_features.extend(replay_features)
            stage_labels.extend(replay_labels)
            stage_slots.extend(replay_slots)

        # Create dataset
        from ..features.feature_extractor import TwiDataset
        dataset = TwiDataset(stage_features, stage_labels, self.label_to_idx, slots=stage_slots)

        # Create weighted sampler for training
        if split == 'train' and len(stage_features) > 0:
            # Compute sample weights based on class weights
            sample_weights = []
            for label in stage_labels:
                weight = stage.class_weights.get(label, 1.0)
                sample_weights.append(weight)

            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )

            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=0,  # Avoid multiprocessing issues
                collate_fn=self._collate_with_padding
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=(split == 'train'),
                num_workers=0,
                collate_fn=self._collate_with_padding
            )

    def _collate_with_padding(self, batch):
        """Collate function with padding for variable-length sequences."""
        import torch.nn.functional as F

        features = []
        labels = []

        for sample in batch:
            if len(sample) >= 2:
                feat, label = sample[0], sample[1]
                features.append(feat)
                labels.append(label)

        if not features:
            return torch.empty(0), torch.empty(0, dtype=torch.long)

        # Pad features to max length in batch
        max_time = max(f.shape[1] for f in features)
        padded_features = []

        for feat in features:
            if feat.shape[1] < max_time:
                pad_amount = max_time - feat.shape[1]
                padded = F.pad(feat, (0, pad_amount))
            else:
                padded = feat
            padded_features.append(padded)

        return torch.stack(padded_features), torch.stack(labels)


class CurriculumTrainer:
    """Main trainer for curriculum learning."""

    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 model_dir: str,
                 patience: int = 10):
        """
        Initialize curriculum trainer.

        Args:
            model: Neural network model
            device: Training device
            model_dir: Directory to save models
            patience: Early stopping patience
        """
        self.model = model
        self.device = device
        self.model_dir = model_dir
        self.patience = patience

        os.makedirs(model_dir, exist_ok=True)

        self.memory_bank = MemoryBank()
        self.training_history = {
            'stages': [],
            'global_metrics': []
        }

    def train_stage(self,
                   stage: CurriculumStage,
                   train_loader: DataLoader,
                   val_loader: DataLoader) -> Dict[str, Any]:
        """
        Train model for a single curriculum stage.

        Args:
            stage: Curriculum stage to train
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            Training statistics for the stage
        """
        logger.info(f"Starting curriculum stage {stage.stage_id}")
        logger.info(f"  Classes: {sorted(list(stage.classes))}")
        logger.info(f"  Max epochs: {stage.max_epochs}")
        logger.info(f"  Learning rate: {stage.base_lr}")

        # Setup optimizer and scheduler for this stage
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=stage.base_lr,
            weight_decay=0.01
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Setup loss function with class weights
        if stage.class_weights:
            # Convert class weights to tensor
            num_classes = self.model.fc3.out_features  # Assuming final layer is fc3
            weight_tensor = torch.ones(num_classes)

            for class_name, weight in stage.class_weights.items():
                if class_name in train_loader.dataset.label_to_idx:
                    class_idx = train_loader.dataset.label_to_idx[class_name]
                    weight_tensor[class_idx] = weight

            criterion = nn.CrossEntropyLoss(weight=weight_tensor.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()

        # Training loop
        stage_history = {
            'stage_id': stage.stage_id,
            'epochs': [],
            'train_losses': [],
            'train_accs': [],
            'val_losses': [],
            'val_accs': [],
            'val_f1s': []
        }

        best_val_acc = 0.0
        epochs_without_improvement = 0

        for epoch in range(stage.max_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()

            train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0.0

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())

            val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0.0
            val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0) if all_targets else 0.0

            # Update learning rate
            scheduler.step(val_acc)

            # Record metrics
            stage_history['epochs'].append(epoch)
            stage_history['train_losses'].append(train_loss / len(train_loader) if len(train_loader) > 0 else 0)
            stage_history['train_accs'].append(train_acc)
            stage_history['val_losses'].append(val_loss / len(val_loader) if len(val_loader) > 0 else 0)
            stage_history['val_accs'].append(val_acc)
            stage_history['val_f1s'].append(val_f1)

            logger.info(f"Epoch {epoch}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%, Val F1={val_f1:.4f}")

            # Check for improvement
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0

                # Save best model for this stage
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'stage': stage.stage_id,
                    'epoch': epoch,
                    'val_acc': val_acc
                }, os.path.join(self.model_dir, f'best_model_stage_{stage.stage_id}.pt'))
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

            # Check stage completion threshold
            if val_acc >= stage.validation_threshold * 100:
                logger.info(f"Stage threshold reached: {val_acc:.2f}% >= {stage.validation_threshold*100:.2f}%")
                break

        stage.best_val_acc = best_val_acc
        stage.completed = True

        logger.info(f"Stage {stage.stage_id} completed. Best val acc: {best_val_acc:.2f}%")

        return stage_history

    def run_curriculum(self,
                      features: List[np.ndarray],
                      labels: List[str],
                      slots: List[Dict[str, Any]],
                      label_to_idx: Dict[str, int],
                      val_split: float = 0.2) -> Dict[str, Any]:
        """
        Run the complete curriculum learning process.

        Args:
            features: List of feature arrays
            labels: List of class labels
            slots: List of slot dictionaries
            label_to_idx: Mapping from labels to indices
            val_split: Validation split ratio

        Returns:
            Complete training history
        """
        logger.info("Starting curriculum learning training")

        # Create curriculum plan
        planner = CurriculumPlanner(features, labels)
        stages = planner.create_curriculum_stages()

        # Split data into train and validation
        from sklearn.model_selection import train_test_split
        train_indices, val_indices = train_test_split(
            range(len(features)),
            test_size=val_split,
            stratify=labels,
            random_state=42
        )

        train_features = [features[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        train_slots = [slots[i] for i in train_indices]

        val_features = [features[i] for i in val_indices]
        val_labels = [labels[i] for i in val_indices]
        val_slots = [slots[i] for i in val_indices]

        # Create data loaders
        train_data_loader = CurriculumDataLoader(
            train_features, train_labels, train_slots, label_to_idx
        )
        val_data_loader = CurriculumDataLoader(
            val_features, val_labels, val_slots, label_to_idx
        )

        # Train each stage
        for stage in stages:
            # Create stage-specific data loaders
            stage_train_loader = train_data_loader.create_stage_dataloader(
                stage, self.memory_bank, 'train'
            )
            stage_val_loader = val_data_loader.create_stage_dataloader(
                stage, self.memory_bank, 'val'
            )

            # Train the stage
            stage_history = self.train_stage(stage, stage_train_loader, stage_val_loader)
            self.training_history['stages'].append(stage_history)

            # Add samples to memory bank for next stages
            if stage.stage_id < len(stages) - 1:  # Not the last stage
                self.memory_bank.add_samples(
                    train_features, train_labels, train_slots, stage.classes
                )

        # Save final model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history,
            'curriculum_completed': True
        }, os.path.join(self.model_dir, 'final_curriculum_model.pt'))

        # Save training history
        with open(os.path.join(self.model_dir, 'curriculum_history.json'), 'w') as f:
            # Convert numpy types for JSON serialization
            serializable_history = self._make_json_serializable(self.training_history)
            json.dump(serializable_history, f, indent=2)

        logger.info("Curriculum learning completed successfully!")

        return self.training_history

    def _make_json_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


def run_curriculum_training(features: List[np.ndarray],
                          labels: List[str],
                          slots: List[Dict[str, Any]],
                          label_to_idx: Dict[str, int],
                          model: nn.Module,
                          device: torch.device,
                          model_dir: str) -> Dict[str, Any]:
    """
    Convenience function to run curriculum training.

    Args:
        features: List of feature arrays
        labels: List of class labels
        slots: List of slot dictionaries
        label_to_idx: Mapping from labels to indices
        model: Neural network model
        device: Training device
        model_dir: Directory to save models

    Returns:
        Training history
    """
    trainer = CurriculumTrainer(model, device, model_dir)
    return trainer.run_curriculum(features, labels, slots, label_to_idx)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # This would be integrated with the main training pipeline
    print("Curriculum Learning module loaded successfully!")
    print("Use run_curriculum_training() to start curriculum learning.")
