#!/usr/bin/env python3
"""
train_simple.py

Simple training script for the multi-sample audio dataset.
Avoids complex augmentation and multiprocessing issues.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.speech_model import ImprovedTwiSpeechModel


class SimpleTwiDataset(Dataset):
    """Simple PyTorch dataset for Twi audio commands."""

    def __init__(self, features, labels, label_to_idx):
        self.features = features
        self.labels = labels
        self.label_to_idx = label_to_idx

        # Convert string labels to indices
        self.label_indices = [self.label_to_idx[label] for label in labels]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.label_indices[idx], dtype=torch.long)
        return feature, label

    def get_num_classes(self):
        return len(self.label_to_idx)


def load_data(data_dir):
    """Load processed features and labels."""
    print(f"Loading data from {data_dir}...")

    # Load features and labels
    features = np.load(os.path.join(data_dir, "features.npy"), allow_pickle=True)
    labels = np.load(os.path.join(data_dir, "labels.npy"), allow_pickle=True)

    # Load label map
    with open(os.path.join(data_dir, "label_map.json"), "r") as f:
        label_map = json.load(f)

    print(f"Loaded {len(features)} samples with {len(label_map)} classes")
    print(f"Feature shape: {features[0].shape}")

    return features, labels, label_map


def create_datasets(features, labels, label_map, train_ratio=0.8):
    """Create train and validation datasets."""
    dataset = SimpleTwiDataset(features, labels, label_map)

    # Split dataset
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")

    return train_dataset, val_dataset


def train_model(model, train_loader, val_loader, epochs=20, learning_rate=0.001, device='cpu'):
    """Train the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_acc = 0.0
    patience = 10
    patience_counter = 0

    print(f"Training for {epochs} epochs...")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_features, batch_labels in train_pbar:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()

            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * train_correct / train_total:.2f}%'
            })

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for batch_features, batch_labels in val_pbar:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()

                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * val_correct / val_total:.2f}%'
                })

        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        train_acc = 100.0 * train_correct / train_total

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc
    }


def evaluate_model(model, val_loader, label_map, device='cpu'):
    """Evaluate the model and generate detailed metrics."""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_features, batch_labels in tqdm(val_loader, desc="Evaluating"):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_features)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    # Convert indices back to label names
    idx_to_label = {idx: label for label, idx in label_map.items()}
    pred_labels = [idx_to_label[idx] for idx in all_predictions]
    true_labels = [idx_to_label[idx] for idx in all_labels]

    # Generate classification report
    print("\nDetailed Classification Report:")
    print(classification_report(true_labels, pred_labels))

    return all_predictions, all_labels, pred_labels, true_labels


def save_model(model, label_map, history, model_dir):
    """Save the trained model and metadata."""
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    # Save label map
    label_map_path = os.path.join(model_dir, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Label map saved to: {label_map_path}")

    # Save training history
    history_path = os.path.join(model_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")

    # Save model config
    config = {
        "input_dim": model.input_dim,
        "hidden_dim": 256,
        "num_classes": model.num_classes,
        "feature_type": "mfcc",
        "model_type": "ImprovedTwiSpeechModel"
    }
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Model config saved to: {config_path}")


def plot_training_history(history, model_dir):
    """Plot and save training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['val_losses'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy curve
    ax2.plot(history['val_accuracies'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(model_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Simple training script for multi-sample audio data")
    parser.add_argument("--data-dir", type=str, default="data/processed_lean_multisample",
                        help="Directory containing processed features")
    parser.add_argument("--model-dir", type=str, default="data/models/simple_multisample",
                        help="Directory to save trained model")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Ratio of data to use for training")

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    features, labels, label_map = load_data(args.data_dir)

    # Create datasets
    train_dataset, val_dataset = create_datasets(features, labels, label_map, args.train_ratio)

    # Create data loaders (no multiprocessing to avoid pickle issues)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Create model
    input_dim = features[0].shape[0]  # Feature dimension (e.g., 39 for MFCC)
    num_classes = len(label_map)

    hidden_dim = 256  # Default hidden dimension
    print(f"Creating model with input_dim={input_dim}, hidden_dim={hidden_dim}, num_classes={num_classes}")
    model = ImprovedTwiSpeechModel(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    model = model.to(device)

    # Train model
    history = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device
    )

    # Evaluate model
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    evaluate_model(model, val_loader, label_map, device)

    # Save model and results
    save_model(model, label_map, history, args.model_dir)
    plot_training_history(history, args.model_dir)

    print(f"\nTraining complete! Model saved to: {args.model_dir}")
    print(f"Best validation accuracy: {history['best_val_acc']:.2f}%")


if __name__ == "__main__":
    main()
