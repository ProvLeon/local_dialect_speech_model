#!/usr/bin/env python3
"""
train_realistic.py

Realistic training script for small multi-class audio datasets.

This script addresses the fundamental issues with training on small datasets:
1. Groups similar intents to reduce class count
2. Uses shorter feature sequences for faster training
3. Implements proper data handling for imbalanced classes
4. Provides realistic evaluation metrics

Designed for datasets with 300-500 samples across 40+ classes.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import argparse
from tqdm import tqdm
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


class RealisticTwiModel(nn.Module):
    """Lightweight model designed for small datasets."""

    def __init__(self, input_dim, num_classes, dropout=0.3):
        super(RealisticTwiModel, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        # Simple but effective architecture
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier with fewer parameters
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, features, time)
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.bn3(self.conv3(x)))

        # Global pooling
        x = self.global_pool(x)  # (batch, 128, 1)
        x = x.squeeze(-1)  # (batch, 128)

        # Classification
        return self.classifier(x)


class FocusedDataset(Dataset):
    """Dataset with realistic preprocessing."""

    def __init__(self, features, labels, label_to_idx, max_length=500):
        self.features = []
        self.labels = []
        self.label_to_idx = label_to_idx

        # Process features to be shorter and more manageable
        for feat, label in zip(features, labels):
            # Truncate or pad to max_length
            if feat.shape[1] > max_length:
                # Take middle section (often most informative)
                start = (feat.shape[1] - max_length) // 2
                feat = feat[:, start:start + max_length]
            else:
                # Pad if too short
                pad_length = max_length - feat.shape[1]
                feat = np.pad(feat, ((0, 0), (0, pad_length)), mode='constant')

            self.features.append(feat)
            self.labels.append(label)

        # Convert labels to indices
        self.label_indices = [self.label_to_idx[label] for label in self.labels]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.label_indices[idx], dtype=torch.long)
        return feature, label


def group_similar_intents(labels, min_samples_per_group=8):
    """Group similar intents to create more balanced classes."""

    # Define intent groupings based on semantic similarity
    intent_groups = {
        'navigation': ['go_home', 'go_back', 'continue'],
        'search_browse': ['search', 'show_description', 'show_reviews', 'show_similar_items'],
        'cart_operations': ['add_to_cart', 'remove_from_cart', 'show_cart', 'clear_cart'],
        'payment_checkout': ['checkout', 'make_payment', 'confirm_order'],
        'filtering_sorting': ['apply_filter', 'clear_filter', 'sort_items'],
        'account_management': ['open_account', 'show_addresses', 'add_address', 'remove_address', 'set_default_address'],
        'order_management': ['show_orders', 'show_order_status', 'track_order', 'cancel_order'],
        'product_selection': ['select_color', 'select_size', 'change_color', 'change_size', 'change_quantity'],
        'wishlist_operations': ['open_wishlist', 'save_for_later'],
        'customer_service': ['help', 'start_live_chat', 'show_faqs'],
        'notifications': ['enable_order_updates', 'disable_order_updates', 'enable_price_alert', 'disable_price_alert'],
        'other_operations': ['return_item', 'exchange_item', 'order_not_arrived', 'refund_status', 'apply_coupon', 'remove_coupon', 'show_price']
    }

    # Count samples per intent
    label_counts = Counter(labels)

    # Create mapping from original intent to group
    intent_to_group = {}
    group_counts = defaultdict(int)

    # First, assign intents to predefined groups
    for group_name, intents in intent_groups.items():
        for intent in intents:
            if intent in label_counts:
                intent_to_group[intent] = group_name
                group_counts[group_name] += label_counts[intent]

    # Handle intents not in predefined groups
    ungrouped_intents = [intent for intent in label_counts if intent not in intent_to_group]

    # For ungrouped intents, either add to existing groups or create individual classes
    for intent in ungrouped_intents:
        count = label_counts[intent]
        if count >= min_samples_per_group:
            # Keep as individual class
            intent_to_group[intent] = intent
            group_counts[intent] = count
        else:
            # Add to 'other_operations' group
            intent_to_group[intent] = 'other_operations'
            group_counts['other_operations'] += count

    # Create new labels
    new_labels = [intent_to_group[label] for label in labels]
    new_label_counts = Counter(new_labels)

    print(f"Original classes: {len(label_counts)}")
    print(f"Grouped classes: {len(new_label_counts)}")
    print(f"New class distribution:")
    for group, count in sorted(new_label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {group}: {count} samples")

    return new_labels, intent_to_group


def load_and_preprocess_data(data_dir, group_intents=True, max_length=500):
    """Load and preprocess data with realistic settings."""
    print(f"Loading data from {data_dir}...")

    # Load original data
    features = np.load(os.path.join(data_dir, "features.npy"), allow_pickle=True)
    labels = np.load(os.path.join(data_dir, "labels.npy"), allow_pickle=True)

    print(f"Original dataset: {len(features)} samples, {len(set(labels))} classes")
    print(f"Original feature shape: {features[0].shape}")

    # Group intents if requested
    if group_intents:
        labels, intent_mapping = group_similar_intents(labels)

        # Save intent mapping for later use
        os.makedirs("data/models", exist_ok=True)
        with open("data/models/intent_grouping.json", "w") as f:
            json.dump(intent_mapping, f, indent=2)

    # Create label mapping
    unique_labels = sorted(set(labels))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}

    print(f"Final dataset: {len(features)} samples, {len(unique_labels)} classes")
    print(f"Using max_length: {max_length}")

    return features, labels, label_to_idx


def create_balanced_split(features, labels, train_ratio=0.7, val_ratio=0.2):
    """Create train/val/test splits ensuring each class appears in each set."""
    label_to_indices = defaultdict(list)
    for i, label in enumerate(labels):
        label_to_indices[label].append(i)

    train_indices = []
    val_indices = []
    test_indices = []

    for label, indices in label_to_indices.items():
        n_samples = len(indices)
        n_train = max(1, int(n_samples * train_ratio))
        n_val = max(1, int(n_samples * val_ratio))

        # Shuffle indices for this label
        np.random.seed(42)
        shuffled = np.random.permutation(indices)

        train_indices.extend(shuffled[:n_train])
        val_indices.extend(shuffled[n_train:n_train + n_val])
        test_indices.extend(shuffled[n_train + n_val:])

    return train_indices, val_indices, test_indices


def train_model(model, train_loader, val_loader, epochs=25, learning_rate=0.002, device='cpu'):
    """Train model with realistic settings."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc = 0.0
    patience = 8
    patience_counter = 0

    history = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'val_loss': []
    }

    print(f"Training for up to {epochs} epochs...")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_features, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()

        # Calculate metrics
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        scheduler.step()

        print(f"Epoch {epoch+1:2d}: Train Acc={train_acc:5.1f}%, Val Acc={val_acc:5.1f}%, "
              f"Train Loss={avg_train_loss:.3f}, Val Loss={avg_val_loss:.3f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs (best val acc: {best_val_acc:.1f}%)")
            break

    return history, best_val_acc


def evaluate_model(model, test_loader, label_to_idx, device='cpu'):
    """Evaluate model on test set."""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_features)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    # Convert back to label names
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    pred_labels = [idx_to_label[idx] for idx in all_predictions]
    true_labels = [idx_to_label[idx] for idx in all_labels]

    # Calculate accuracy
    accuracy = sum(p == t for p, t in zip(pred_labels, true_labels)) / len(true_labels)

    print(f"\nTest Accuracy: {accuracy*100:.1f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(true_labels, pred_labels))

    return accuracy


def save_model(model, label_to_idx, history, best_val_acc, model_dir):
    """Save model and metadata."""
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))

    # Save metadata
    metadata = {
        "model_type": "RealisticTwiModel",
        "input_dim": model.input_dim,
        "num_classes": model.num_classes,
        "best_val_acc": best_val_acc,
        "label_to_idx": label_to_idx
    }

    with open(os.path.join(model_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(model_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"Model saved to {model_dir}")


def main():
    parser = argparse.ArgumentParser(description="Realistic training for small audio datasets")
    parser.add_argument("--data-dir", type=str, default="data/processed_lean_multisample",
                        help="Directory with processed features")
    parser.add_argument("--model-dir", type=str, default="data/models/realistic",
                        help="Directory to save model")
    parser.add_argument("--max-length", type=int, default=500,
                        help="Maximum sequence length for features")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=24,
                        help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.002,
                        help="Learning rate")
    parser.add_argument("--no-grouping", action="store_true",
                        help="Disable intent grouping")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess data
    features, labels, label_to_idx = load_and_preprocess_data(
        args.data_dir,
        group_intents=not args.no_grouping,
        max_length=args.max_length
    )

    # Create datasets
    train_indices, val_indices, test_indices = create_balanced_split(features, labels)

    train_features = [features[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_features = [features[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    test_features = [features[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]

    train_dataset = FocusedDataset(train_features, train_labels, label_to_idx, args.max_length)
    val_dataset = FocusedDataset(val_features, val_labels, label_to_idx, args.max_length)
    test_dataset = FocusedDataset(test_features, test_labels, label_to_idx, args.max_length)

    print(f"Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Create model
    input_dim = train_features[0].shape[0]
    num_classes = len(label_to_idx)

    print(f"Model: input_dim={input_dim}, num_classes={num_classes}, max_length={args.max_length}")
    model = RealisticTwiModel(input_dim=input_dim, num_classes=num_classes)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Train
    history, best_val_acc = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device
    )

    # Evaluate
    test_acc = evaluate_model(model, test_loader, label_to_idx, device)

    # Save
    save_model(model, label_to_idx, history, best_val_acc, args.model_dir)

    print(f"\nFinal Results:")
    print(f"Best Validation Accuracy: {best_val_acc:.1f}%")
    print(f"Test Accuracy: {test_acc*100:.1f}%")


if __name__ == "__main__":
    main()
