#!/usr/bin/env python3
"""
train_enhanced_fixed.py

Enhanced training script for multi-sample audio dataset with:
- Data augmentation
- Class balancing
- Advanced model architecture
- Slot-aware training
- Fixed for small datasets and CPU training

This script avoids the issues with the original enhanced training:
- No stratified splitting (handles small classes)
- No multiprocessing (avoids pickle issues)
- Simplified data loading
- Better error handling
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.speech_model import ImprovedTwiSpeechModel
from src.features.feature_augmenter import FeatureAugmenter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedTwiDataset(Dataset):
    """Enhanced dataset with augmentation and slot support."""

    def __init__(self, features, labels, slots, label_to_idx, augment=False, augment_factor=2):
        self.features = features
        self.labels = labels
        self.slots = slots
        self.label_to_idx = label_to_idx
        self.augment = augment
        self.augment_factor = augment_factor

        # Convert labels to indices
        self.label_indices = [self.label_to_idx[label] for label in labels]

        # Initialize augmenter if needed
        if self.augment:
            self.augmenter = FeatureAugmenter()

        # Create augmented dataset for training
        if self.augment:
            self._create_augmented_data()

    def _create_augmented_data(self):
        """Create augmented versions of underrepresented classes."""
        # Count samples per class
        label_counts = Counter(self.labels)
        max_count = max(label_counts.values())

        aug_features = []
        aug_labels = []
        aug_slots = []
        aug_indices = []

        for i, (feature, label, slot) in enumerate(zip(self.features, self.labels, self.slots)):
            count = label_counts[label]

            # Calculate how many augmented copies we need
            target_count = min(max_count, count * self.augment_factor)
            needed = max(0, target_count - count)

            if needed > 0:
                # Create augmented copies
                for _ in range(needed):
                    try:
                        aug_feature = self.augmenter.augment(feature)
                        aug_features.append(aug_feature)
                        aug_labels.append(label)
                        aug_slots.append(slot.copy() if isinstance(slot, dict) else {})
                        aug_indices.append(len(self.features) + len(aug_features) - 1)
                    except Exception as e:
                        logger.warning(f"Failed to augment sample {i}: {e}")

        # Add augmented data
        if aug_features:
            self.features.extend(aug_features)
            self.labels.extend(aug_labels)
            self.slots.extend(aug_slots)
            self.label_indices = [self.label_to_idx[label] for label in self.labels]

            logger.info(f"Added {len(aug_features)} augmented samples")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.label_indices[idx], dtype=torch.long)
        slot = self.slots[idx] if idx < len(self.slots) else {}
        return feature, label, slot

    def get_num_classes(self):
        return len(self.label_to_idx)

    def get_class_weights(self):
        """Compute class weights for balanced training."""
        label_counts = Counter(self.label_indices)
        total_samples = len(self.label_indices)
        num_classes = len(self.label_to_idx)

        weights = torch.zeros(num_classes)
        for class_idx, count in label_counts.items():
            weights[class_idx] = total_samples / (num_classes * count)

        return weights


def simple_collate(batch):
    """Simple collate function that works without multiprocessing."""
    features = []
    labels = []
    slots = []

    for item in batch:
        if len(item) == 3:
            feat, lab, slot = item
        else:
            feat, lab = item
            slot = {}

        features.append(feat)
        labels.append(lab)
        slots.append(slot)

    return torch.stack(features), torch.stack(labels), slots


def load_data(data_dir):
    """Load processed features, labels, and slots."""
    logger.info(f"Loading data from {data_dir}...")

    # Load features and labels
    features = np.load(os.path.join(data_dir, "features.npy"), allow_pickle=True)
    labels = np.load(os.path.join(data_dir, "labels.npy"), allow_pickle=True)

    # Load slots if available
    slots_path = os.path.join(data_dir, "slots.json")
    if os.path.exists(slots_path):
        with open(slots_path, "r") as f:
            slots = json.load(f)
    else:
        slots = [{} for _ in range(len(features))]

    # Load label map
    with open(os.path.join(data_dir, "label_map.json"), "r") as f:
        label_map = json.load(f)

    # Convert to lists for easier manipulation
    features = list(features)
    labels = list(labels)

    # Ensure slots list matches features length
    while len(slots) < len(features):
        slots.append({})

    logger.info(f"Loaded {len(features)} samples with {len(label_map)} classes")
    logger.info(f"Feature shape: {features[0].shape}")

    return features, labels, slots, label_map


def create_datasets(features, labels, slots, label_map, augment=True):
    """Create train, validation, and test datasets."""
    # Create indices for splitting
    indices = np.arange(len(features))

    # Simple random split (no stratification to handle small classes)
    train_indices, temp_indices = train_test_split(
        indices, test_size=0.3, random_state=42
    )
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, random_state=42
    )

    # Create subset data
    train_features = [features[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    train_slots = [slots[i] for i in train_indices]

    val_features = [features[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    val_slots = [slots[i] for i in val_indices]

    test_features = [features[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    test_slots = [slots[i] for i in test_indices]

    # Create datasets
    train_dataset = EnhancedTwiDataset(
        train_features, train_labels, train_slots, label_map,
        augment=augment, augment_factor=2
    )

    val_dataset = EnhancedTwiDataset(
        val_features, val_labels, val_slots, label_map,
        augment=False
    )

    test_dataset = EnhancedTwiDataset(
        test_features, test_labels, test_slots, label_map,
        augment=False
    )

    logger.info(f"Train set: {len(train_dataset)} samples (after augmentation)")
    logger.info(f"Validation set: {len(val_dataset)} samples")
    logger.info(f"Test set: {len(test_dataset)} samples")

    return train_dataset, val_dataset, test_dataset


def train_model(model, train_loader, val_loader, epochs=20, learning_rate=0.001, device='cpu'):
    """Train the model with enhanced features."""
    # Get class weights for balanced training
    class_weights = None
    try:
        class_weights = train_loader.dataset.get_class_weights().to(device)
        logger.info("Using class weights for balanced training")
    except Exception as e:
        logger.warning(f"Could not compute class weights: {e}")

    # Setup training
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )

    # Training history
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_accuracies': [],
        'val_accuracies': [],
        'learning_rates': []
    }

    best_val_acc = 0.0
    patience = 10
    patience_counter = 0

    logger.info(f"Training for {epochs} epochs...")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_features, batch_labels, batch_slots in train_pbar:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()

            # Update progress bar
            train_acc = 100.0 * train_correct / train_total
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_acc:.1f}%'
            })

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for batch_features, batch_labels, batch_slots in val_pbar:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()

                # Update progress bar
                val_acc = 100.0 * val_correct / val_total
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{val_acc:.1f}%'
                })

        # Calculate epoch averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']

        # Update history
        history['train_losses'].append(avg_train_loss)
        history['val_losses'].append(avg_val_loss)
        history['train_accuracies'].append(train_acc)
        history['val_accuracies'].append(val_acc)
        history['learning_rates'].append(current_lr)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Print epoch summary
        logger.info(f"Epoch {epoch+1}/{epochs}:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        logger.info(f"  Learning Rate: {current_lr:.6f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break

    history['best_val_acc'] = best_val_acc
    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")

    return history


def evaluate_model(model, test_loader, label_map, device='cpu'):
    """Comprehensive model evaluation."""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_features, batch_labels, batch_slots in tqdm(test_loader, desc="Evaluating"):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_features)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    # Convert indices back to labels
    idx_to_label = {idx: label for label, idx in label_map.items()}
    pred_labels = [idx_to_label[idx] for idx in all_predictions]
    true_labels = [idx_to_label[idx] for idx in all_labels]

    # Print detailed metrics
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(true_labels, pred_labels))

    # Calculate accuracy
    accuracy = sum(p == t for p, t in zip(pred_labels, true_labels)) / len(true_labels)
    logger.info(f"Test Accuracy: {accuracy*100:.2f}%")

    return accuracy, pred_labels, true_labels


def save_model_and_results(model, label_map, history, model_dir):
    """Save model, configuration, and training results."""
    os.makedirs(model_dir, exist_ok=True)

    # Save model state
    model_path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to: {model_path}")

    # Save configuration
    config = {
        "input_dim": model.input_dim,
        "hidden_dim": 256,
        "num_classes": model.num_classes,
        "feature_type": "mfcc",
        "model_type": "ImprovedTwiSpeechModel",
        "best_val_acc": history.get('best_val_acc', 0.0)
    }
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Save label map
    label_map_path = os.path.join(model_dir, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)

    # Save training history
    history_path = os.path.join(model_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"All files saved to: {model_dir}")


def plot_training_curves(history, model_dir):
    """Plot and save training curves."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    epochs = range(1, len(history['train_losses']) + 1)

    # Loss curves
    ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy curves
    ax2.plot(epochs, history['train_accuracies'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_accuracies'], 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    # Learning rate
    ax3.plot(epochs, history['learning_rates'], 'g-')
    ax3.set_title('Learning Rate')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.grid(True)

    # Best accuracy indicator
    best_epoch = np.argmax(history['val_accuracies']) + 1
    best_acc = max(history['val_accuracies'])
    ax4.bar(['Best Validation Accuracy'], [best_acc])
    ax4.set_title(f'Best Performance (Epoch {best_epoch})')
    ax4.set_ylabel('Accuracy (%)')
    ax4.text(0, best_acc/2, f'{best_acc:.2f}%', ha='center', va='center', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(model_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Training curves saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced training for multi-sample audio data")
    parser.add_argument("--data-dir", type=str, default="data/processed_lean_fast",
                        help="Directory containing processed features")
    parser.add_argument("--model-dir", type=str, default="data/models/enhanced_multisample",
                        help="Directory to save trained model")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable data augmentation")

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    features, labels, slots, label_map = load_data(args.data_dir)

    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        features, labels, slots, label_map,
        augment=not args.no_augment
    )

    # Create data loaders (no multiprocessing)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=simple_collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=simple_collate
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=simple_collate
    )

    # Create model
    input_dim = features[0].shape[0]
    hidden_dim = 256
    num_classes = len(label_map)

    logger.info(f"Creating model: input_dim={input_dim}, hidden_dim={hidden_dim}, num_classes={num_classes}")
    model = ImprovedTwiSpeechModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=0.3
    )
    model = model.to(device)

    # Train model
    history = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device
    )

    # Evaluate on test set
    logger.info("\nFinal evaluation on test set:")
    test_accuracy, pred_labels, true_labels = evaluate_model(
        model, test_loader, label_map, device
    )

    # Save everything
    save_model_and_results(model, label_map, history, args.model_dir)
    plot_training_curves(history, args.model_dir)

    # Final summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Best Validation Accuracy: {history['best_val_acc']:.2f}%")
    print(f"Final Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Model saved to: {args.model_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
