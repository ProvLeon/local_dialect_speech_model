import torch
from torch.utils.data import Dataset
# import numpy as np
from .feature_augmenter import FeatureAugmenter


class AugmentedTwiDataset(Dataset):
    """
    Enhanced PyTorch dataset for Twi audio commands with augmentation
    """
    def __init__(self, features, labels, label_to_idx=None, augment=False, transform=None):
        """
        Initialize dataset with optional augmentation

        Args:
            features: List or array of features
            labels: List of string labels
            label_to_idx: Mapping from string labels to indices
            augment: Whether to apply augmentation
            transform: Additional transforms to apply
        """
        super().__init__()
        self.features = features
        self.labels = labels
        self.transform = transform
        self.augment = augment

        if augment:
            self.augmenter = FeatureAugmenter(prob=0.7)

        if label_to_idx is None:
            unique_labels = sorted(set(labels))
            self.label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx

        # Convert string labels to indices
        self.label_indices = [self.label_to_idx[label] for label in labels]

        # Analyze class distribution for class balance weighting
        self._analyze_class_distribution()

    def _analyze_class_distribution(self):
        """Analyze class distribution for imbalance handling"""
        self.class_counts = {}
        for label in self.label_indices:
            self.class_counts[label] = self.class_counts.get(label, 0) + 1

        self.class_weights = {}
        total_samples = len(self.label_indices)
        num_classes = len(self.class_counts)

        for label, count in self.class_counts.items():
            self.class_weights[label] = total_samples / (count * num_classes)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx].copy()
        label = self.label_indices[idx]

        # Apply augmentation if enabled
        if self.augment:
            feature = self.augmenter.augment(feature)

        # Convert to torch tensors
        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Apply additional transforms if provided
        if self.transform:
            feature_tensor = self.transform(feature_tensor)

        return feature_tensor, label_tensor

    def get_num_classes(self):
        """Get the number of classes"""
        return len(self.label_to_idx)

    def get_class_weights(self):
        """Get class weights for imbalanced learning"""
        weights = torch.zeros(len(self.label_to_idx))
        for label, weight in self.class_weights.items():
            weights[label] = weight

        return weights
