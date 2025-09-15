#!/usr/bin/env python3
"""
Advanced Data Augmentation for Rare Class Preservation in Speech Intent Classification

This module provides sophisticated augmentation strategies specifically designed to maintain
all 47 classes while addressing extreme class imbalance and overfitting issues.

Key Features:
- Intelligent rare class upsampling
- Multiple augmentation techniques for audio features
- Class-aware augmentation intensity
- Temporal and spectral augmentation
- Synthetic minority oversampling (SMOTE) adapted for speech features
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter, defaultdict
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Tuple, Optional, Any
import random
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


class AdvancedAudioAugmenter:
    """
    Advanced augmentation strategies for speech features with class-aware intensity.
    """

    def __init__(self, target_samples_per_class: int = 20, min_samples_per_class: int = 10):
        """
        Initialize the advanced augmenter.

        Args:
            target_samples_per_class: Target number of samples for each class
            min_samples_per_class: Minimum samples to generate for very rare classes
        """
        self.target_samples_per_class = target_samples_per_class
        self.min_samples_per_class = min_samples_per_class

    def compute_augmentation_factors(self, labels: List[str]) -> Dict[str, int]:
        """
        Compute how many augmented samples each class needs.

        Args:
            labels: List of class labels

        Returns:
            Dictionary mapping class labels to required augmentation factors
        """
        label_counts = Counter(labels)
        augmentation_factors = {}

        for label, count in label_counts.items():
            if count < self.min_samples_per_class:
                # Very rare classes get minimum boost
                augmentation_factors[label] = max(1, self.min_samples_per_class - count)
            elif count < self.target_samples_per_class:
                # Moderately rare classes get proportional boost
                augmentation_factors[label] = max(1, self.target_samples_per_class - count)
            else:
                # Common classes get minimal augmentation
                augmentation_factors[label] = max(1, int(0.1 * count))

        logger.info("Augmentation factors computed:")
        for label, factor in sorted(augmentation_factors.items()):
            original_count = label_counts[label]
            logger.info(f"  {label}: {original_count} -> {original_count + factor} (+{factor})")

        return augmentation_factors

    def time_stretch(self, features: np.ndarray, stretch_factor: float = 1.2) -> np.ndarray:
        """
        Apply time stretching to audio features.

        Args:
            features: Input features of shape (channels, time)
            stretch_factor: Factor to stretch time dimension

        Returns:
            Time-stretched features
        """
        channels, time_steps = features.shape
        new_time_steps = int(time_steps * stretch_factor)

        # Use linear interpolation for time stretching
        indices = np.linspace(0, time_steps - 1, new_time_steps)
        stretched = np.zeros((channels, new_time_steps))

        for c in range(channels):
            stretched[c] = np.interp(indices, np.arange(time_steps), features[c])

        return stretched

    def pitch_shift(self, features: np.ndarray, semitones: float = 2.0) -> np.ndarray:
        """
        Simulate pitch shifting by frequency domain manipulation.

        Args:
            features: Input features of shape (channels, time)
            semitones: Number of semitones to shift

        Returns:
            Pitch-shifted features
        """
        # Simple pitch shift simulation by shifting mel-scale features
        channels, time_steps = features.shape
        shift_bins = int(semitones * 0.5)  # Approximate shift in feature bins

        shifted = np.zeros_like(features)

        if shift_bins > 0:
            # Shift up
            shifted[shift_bins:] = features[:-shift_bins]
        elif shift_bins < 0:
            # Shift down
            shifted[:shift_bins] = features[-shift_bins:]
        else:
            shifted = features.copy()

        return shifted

    def add_noise(self, features: np.ndarray, noise_factor: float = 0.05) -> np.ndarray:
        """
        Add gaussian noise to features.

        Args:
            features: Input features
            noise_factor: Standard deviation of noise relative to signal std

        Returns:
            Noisy features
        """
        signal_std = np.std(features)
        noise = np.random.normal(0, noise_factor * signal_std, features.shape)
        return features + noise

    def spectral_dropout(self, features: np.ndarray, dropout_prob: float = 0.1) -> np.ndarray:
        """
        Randomly zero out entire frequency channels (spectral dropout).

        Args:
            features: Input features of shape (channels, time)
            dropout_prob: Probability of dropping each channel

        Returns:
            Features with spectral dropout applied
        """
        channels, time_steps = features.shape
        mask = np.random.random(channels) > dropout_prob

        dropped = features.copy()
        dropped[~mask] = 0

        return dropped

    def temporal_masking(self, features: np.ndarray, max_mask_length: int = 10, num_masks: int = 2) -> np.ndarray:
        """
        Apply temporal masking (SpecAugment style).

        Args:
            features: Input features of shape (channels, time)
            max_mask_length: Maximum length of each mask
            num_masks: Number of masks to apply

        Returns:
            Temporally masked features
        """
        channels, time_steps = features.shape
        masked = features.copy()

        for _ in range(num_masks):
            if time_steps <= max_mask_length:
                continue

            mask_length = random.randint(1, min(max_mask_length, time_steps // 4))
            mask_start = random.randint(0, time_steps - mask_length)

            masked[:, mask_start:mask_start + mask_length] = 0

        return masked

    def feature_interpolation(self, features1: np.ndarray, features2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Interpolate between two feature vectors.

        Args:
            features1: First feature vector
            features2: Second feature vector
            alpha: Interpolation factor (0.5 = midpoint)

        Returns:
            Interpolated features
        """
        # Ensure same temporal length by padding/truncating
        min_time = min(features1.shape[1], features2.shape[1])
        f1_trimmed = features1[:, :min_time]
        f2_trimmed = features2[:, :min_time]

        return alpha * f1_trimmed + (1 - alpha) * f2_trimmed

    def smart_augment_sample(self, features: np.ndarray, intensity: str = 'medium') -> np.ndarray:
        """
        Apply smart augmentation with class-aware intensity.

        Args:
            features: Input features
            intensity: 'light', 'medium', or 'heavy'

        Returns:
            Augmented features
        """
        augmented = features.copy()

        if intensity == 'light':
            # Minimal augmentation for common classes
            if random.random() < 0.3:
                augmented = self.add_noise(augmented, noise_factor=0.02)
            if random.random() < 0.2:
                augmented = self.temporal_masking(augmented, max_mask_length=3, num_masks=1)

        elif intensity == 'medium':
            # Moderate augmentation for moderately rare classes
            if random.random() < 0.5:
                augmented = self.add_noise(augmented, noise_factor=0.05)
            if random.random() < 0.4:
                augmented = self.time_stretch(augmented, stretch_factor=random.uniform(0.9, 1.1))
            if random.random() < 0.3:
                augmented = self.temporal_masking(augmented, max_mask_length=5, num_masks=1)

        elif intensity == 'heavy':
            # Aggressive augmentation for very rare classes
            if random.random() < 0.7:
                augmented = self.add_noise(augmented, noise_factor=0.08)
            if random.random() < 0.6:
                augmented = self.time_stretch(augmented, stretch_factor=random.uniform(0.8, 1.3))
            if random.random() < 0.5:
                augmented = self.pitch_shift(augmented, semitones=random.uniform(-3, 3))
            if random.random() < 0.4:
                augmented = self.spectral_dropout(augmented, dropout_prob=0.15)
            if random.random() < 0.3:
                augmented = self.temporal_masking(augmented, max_mask_length=8, num_masks=2)

        return augmented


class SMOTEAudioAugmenter:
    """
    SMOTE (Synthetic Minority Oversampling Technique) adapted for audio features.
    """

    def __init__(self, k_neighbors: int = 3):
        """
        Initialize SMOTE augmenter.

        Args:
            k_neighbors: Number of nearest neighbors to use for interpolation
        """
        self.k_neighbors = k_neighbors

    def flatten_features(self, features_list: List[np.ndarray]) -> np.ndarray:
        """
        Flatten variable-length features for SMOTE processing.

        Args:
            features_list: List of feature arrays with shape (channels, time)

        Returns:
            Flattened features matrix
        """
        # Use statistical summary features for SMOTE
        summary_features = []

        for features in features_list:
            # Extract statistical features from each channel
            stats = []
            for channel in features:
                stats.extend([
                    np.mean(channel),
                    np.std(channel),
                    np.min(channel),
                    np.max(channel),
                    np.median(channel),
                    np.percentile(channel, 25),
                    np.percentile(channel, 75)
                ])
            summary_features.append(stats)

        return np.array(summary_features)

    def reconstruct_features(self, summary_features: np.ndarray, reference_features: List[np.ndarray]) -> np.ndarray:
        """
        Reconstruct time-series features from statistical summaries.

        Args:
            summary_features: Statistical summary features
            reference_features: Reference features for temporal structure

        Returns:
            Reconstructed time-series features
        """
        # Choose a random reference for temporal structure
        ref_idx = random.randint(0, len(reference_features) - 1)
        reference = reference_features[ref_idx]
        channels, time_steps = reference.shape

        # Reconstruct by scaling reference features to match summary statistics
        reconstructed = np.zeros_like(reference)

        stats_per_channel = 7  # Number of stats per channel
        for c in range(channels):
            start_idx = c * stats_per_channel
            target_mean = summary_features[start_idx]
            target_std = summary_features[start_idx + 1]

            # Normalize reference channel and scale to target statistics
            ref_channel = reference[c]
            ref_mean = np.mean(ref_channel)
            ref_std = np.std(ref_channel) + 1e-8

            normalized = (ref_channel - ref_mean) / ref_std
            reconstructed[c] = normalized * target_std + target_mean

        return reconstructed

    def generate_smote_samples(self, features_list: List[np.ndarray], num_samples: int) -> List[np.ndarray]:
        """
        Generate synthetic samples using SMOTE.

        Args:
            features_list: List of feature arrays for the class
            num_samples: Number of synthetic samples to generate

        Returns:
            List of synthetic feature arrays
        """
        if len(features_list) < 2:
            # Can't do SMOTE with less than 2 samples, use direct augmentation
            augmenter = AdvancedAudioAugmenter()
            synthetic = []
            for _ in range(num_samples):
                base_features = random.choice(features_list)
                augmented = augmenter.smart_augment_sample(base_features, intensity='heavy')
                synthetic.append(augmented)
            return synthetic

        # Convert to statistical summaries
        summary_matrix = self.flatten_features(features_list)

        if len(summary_matrix) < self.k_neighbors:
            k = len(summary_matrix) - 1
        else:
            k = self.k_neighbors

        # Fit nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(summary_matrix)  # +1 to exclude self

        synthetic_samples = []
        for _ in range(num_samples):
            # Choose random sample as base
            base_idx = random.randint(0, len(summary_matrix) - 1)
            base_summary = summary_matrix[base_idx]

            # Find neighbors
            distances, indices = nbrs.kneighbors([base_summary])
            neighbor_indices = indices[0][1:]  # Exclude self

            # Choose random neighbor
            neighbor_idx = random.choice(neighbor_indices)
            neighbor_summary = summary_matrix[neighbor_idx]

            # Interpolate
            alpha = random.random()
            synthetic_summary = alpha * base_summary + (1 - alpha) * neighbor_summary

            # Reconstruct time-series features
            synthetic_features = self.reconstruct_features(synthetic_summary, features_list)
            synthetic_samples.append(synthetic_features)

        return synthetic_samples


class ClassAwareAugmentationPipeline:
    """
    Main pipeline for class-aware augmentation to preserve all 47 classes.
    """

    def __init__(self,
                 target_samples_per_class: int = 20,
                 min_samples_per_class: int = 10,
                 use_smote: bool = True,
                 preserve_original: bool = True):
        """
        Initialize the augmentation pipeline.

        Args:
            target_samples_per_class: Target number of samples for each class
            min_samples_per_class: Minimum samples for very rare classes
            use_smote: Whether to use SMOTE for synthetic sample generation
            preserve_original: Whether to keep original samples
        """
        self.target_samples_per_class = target_samples_per_class
        self.min_samples_per_class = min_samples_per_class
        self.use_smote = use_smote
        self.preserve_original = preserve_original

        self.augmenter = AdvancedAudioAugmenter(target_samples_per_class, min_samples_per_class)
        self.smote_augmenter = SMOTEAudioAugmenter() if use_smote else None

    def categorize_classes(self, labels: List[str]) -> Dict[str, List[str]]:
        """
        Categorize classes by rarity for different augmentation strategies.

        Args:
            labels: List of class labels

        Returns:
            Dictionary with class categories
        """
        label_counts = Counter(labels)

        categories = {
            'very_rare': [],    # 1-3 samples
            'rare': [],         # 4-10 samples
            'moderate': [],     # 11-20 samples
            'common': []        # 21+ samples
        }

        for label, count in label_counts.items():
            if count <= 3:
                categories['very_rare'].append(label)
            elif count <= 10:
                categories['rare'].append(label)
            elif count <= 20:
                categories['moderate'].append(label)
            else:
                categories['common'].append(label)

        logger.info("Class categorization:")
        for category, class_list in categories.items():
            logger.info(f"  {category}: {len(class_list)} classes - {class_list}")

        return categories

    def augment_class_samples(self,
                             features_list: List[np.ndarray],
                             labels_list: List[str],
                             class_label: str,
                             num_augmentations: int,
                             intensity: str) -> Tuple[List[np.ndarray], List[str]]:
        """
        Augment samples for a specific class.

        Args:
            features_list: List of feature arrays for the class
            labels_list: List of labels for the class
            class_label: The class label being augmented
            num_augmentations: Number of augmented samples to generate
            intensity: Augmentation intensity level

        Returns:
            Tuple of (augmented_features, augmented_labels)
        """
        augmented_features = []
        augmented_labels = []

        # Keep original samples if requested
        if self.preserve_original:
            augmented_features.extend(features_list)
            augmented_labels.extend(labels_list)

        # Generate synthetic samples using SMOTE if available and applicable
        if self.use_smote and len(features_list) >= 2 and num_augmentations > 0:
            smote_count = min(num_augmentations // 2, len(features_list))
            if smote_count > 0:
                smote_samples = self.smote_augmenter.generate_smote_samples(features_list, smote_count)
                augmented_features.extend(smote_samples)
                augmented_labels.extend([class_label] * smote_count)
                num_augmentations -= smote_count

        # Generate remaining samples using traditional augmentation
        for _ in range(num_augmentations):
            base_features = random.choice(features_list)
            augmented = self.augmenter.smart_augment_sample(base_features, intensity=intensity)
            augmented_features.append(augmented)
            augmented_labels.append(class_label)

        return augmented_features, augmented_labels

    def process_full_dataset(self,
                           features: List[np.ndarray],
                           labels: List[str],
                           slots: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[np.ndarray], List[str], List[Dict[str, Any]]]:
        """
        Process the full dataset with class-aware augmentation.

        Args:
            features: List of feature arrays
            labels: List of class labels
            slots: Optional list of slot dictionaries

        Returns:
            Tuple of (augmented_features, augmented_labels, augmented_slots)
        """
        if slots is None:
            slots = [{} for _ in range(len(features))]

        # Group samples by class
        class_samples = defaultdict(list)
        class_labels = defaultdict(list)
        class_slots = defaultdict(list)

        for feat, label, slot in zip(features, labels, slots):
            class_samples[label].append(feat)
            class_labels[label].append(label)
            class_slots[label].append(slot)

        # Categorize classes
        categories = self.categorize_classes(labels)

        # Compute augmentation factors
        augmentation_factors = self.augmenter.compute_augmentation_factors(labels)

        # Process each class
        final_features = []
        final_labels = []
        final_slots = []

        for label in class_samples.keys():
            # Determine augmentation intensity based on category
            if label in categories['very_rare']:
                intensity = 'heavy'
            elif label in categories['rare']:
                intensity = 'medium'
            elif label in categories['moderate']:
                intensity = 'light'
            else:
                intensity = 'light'

            # Augment class samples
            aug_features, aug_labels = self.augment_class_samples(
                class_samples[label],
                class_labels[label],
                label,
                augmentation_factors[label],
                intensity
            )

            # Extend slots (replicate original slots for augmented samples)
            original_slots = class_slots[label]
            aug_slots = original_slots.copy()

            # Add slots for augmented samples (replicate randomly from originals)
            num_new_samples = len(aug_features) - len(original_slots)
            for _ in range(num_new_samples):
                random_slot = random.choice(original_slots) if original_slots else {}
                aug_slots.append(deepcopy(random_slot))

            final_features.extend(aug_features)
            final_labels.extend(aug_labels)
            final_slots.extend(aug_slots)

        logger.info(f"Augmentation complete:")
        logger.info(f"  Original samples: {len(features)}")
        logger.info(f"  Final samples: {len(final_features)}")
        logger.info(f"  Augmentation ratio: {len(final_features) / len(features):.2f}x")

        # Verify all classes are preserved
        original_classes = set(labels)
        final_classes = set(final_labels)
        assert original_classes == final_classes, f"Classes lost during augmentation: {original_classes - final_classes}"

        return final_features, final_labels, final_slots


def create_balanced_dataset(features: List[np.ndarray],
                          labels: List[str],
                          slots: Optional[List[Dict[str, Any]]] = None,
                          target_samples_per_class: int = 20,
                          min_samples_per_class: int = 10) -> Tuple[List[np.ndarray], List[str], List[Dict[str, Any]]]:
    """
    Convenience function to create a balanced dataset preserving all classes.

    Args:
        features: List of feature arrays
        labels: List of class labels
        slots: Optional list of slot dictionaries
        target_samples_per_class: Target number of samples for each class
        min_samples_per_class: Minimum samples for very rare classes

    Returns:
        Tuple of (balanced_features, balanced_labels, balanced_slots)
    """
    pipeline = ClassAwareAugmentationPipeline(
        target_samples_per_class=target_samples_per_class,
        min_samples_per_class=min_samples_per_class,
        use_smote=True,
        preserve_original=True
    )

    return pipeline.process_full_dataset(features, labels, slots)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Simulate some sample data
    np.random.seed(42)
    random.seed(42)

    # Create mock features with different temporal lengths
    features = []
    labels = []

    # Very rare class (1 sample)
    features.append(np.random.randn(39, 50))
    labels.append("very_rare_1")

    # Rare class (3 samples)
    for _ in range(3):
        features.append(np.random.randn(39, random.randint(40, 80)))
        labels.append("rare_1")

    # Moderate class (15 samples)
    for _ in range(15):
        features.append(np.random.randn(39, random.randint(60, 120)))
        labels.append("moderate_1")

    # Common class (50 samples)
    for _ in range(50):
        features.append(np.random.randn(39, random.randint(80, 150)))
        labels.append("common_1")

    print(f"Original dataset: {len(features)} samples, {len(set(labels))} classes")

    # Apply augmentation
    balanced_features, balanced_labels, balanced_slots = create_balanced_dataset(
        features, labels, target_samples_per_class=20
    )

    print(f"Balanced dataset: {len(balanced_features)} samples, {len(set(balanced_labels))} classes")

    # Verify class distribution
    from collections import Counter
    final_counts = Counter(balanced_labels)
    print("Final class distribution:")
    for label, count in sorted(final_counts.items()):
        print(f"  {label}: {count}")
