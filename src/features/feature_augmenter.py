import numpy as np
import torch
import random

class FeatureAugmenter:
    """
    Augmentation techniques specifically for audio features like MFCC
    to help the model generalize better
    """
    def __init__(self, prob=0.5):
        """
        Initialize augmenter with probability of applying each transform
        """
        self.prob = prob

    def time_masking(self, feature, max_time_mask=10):
        """Apply time masking to features (mask out time steps)"""
        if np.random.random() > self.prob:
            return feature

        feature = feature.copy()
        time_len = feature.shape[1]

        # Choose number of masks
        num_masks = np.random.randint(1, 3)

        for _ in range(num_masks):
            # Choose mask length
            t_mask = np.random.randint(1, max_time_mask)

            # Choose mask start point
            t_start = np.random.randint(0, time_len - t_mask)

            # Apply mask (set to zero)
            feature[:, t_start:t_start+t_mask] = 0

        return feature

    def freq_masking(self, feature, max_freq_mask=8):
        """Apply frequency masking to features (mask out frequency bands)"""
        if np.random.random() > self.prob:
            return feature

        feature = feature.copy()
        freq_len = feature.shape[0]

        # Choose number of masks
        num_masks = np.random.randint(1, 3)

        for _ in range(num_masks):
            # Choose mask length
            f_mask = np.random.randint(1, min(max_freq_mask, freq_len//3))

            # Choose mask start point
            f_start = np.random.randint(0, freq_len - f_mask)

            # Apply mask (set to zero)
            feature[f_start:f_start+f_mask, :] = 0

        return feature

    def feature_scaling(self, feature, scale_range=(0.8, 1.2)):
        """Scale feature values by a random factor"""
        if np.random.random() > self.prob:
            return feature

        # Choose scale factor
        scale = np.random.uniform(*scale_range)
        return feature * scale

    def noise_addition(self, feature, noise_level=0.005):
        """Add random noise to features"""
        if np.random.random() > self.prob:
            return feature

        noise = np.random.randn(*feature.shape) * noise_level
        return feature + noise

    def feature_dropout(self, feature, dropout_prob=0.05):
        """Randomly drop feature values"""
        if np.random.random() > self.prob:
            return feature

        feature = feature.copy()
        mask = np.random.random(feature.shape) > dropout_prob
        return feature * mask

    def augment(self, feature):
        """Apply a sequence of random augmentations"""
        # List of augmentation functions
        augmentation_functions = [
            self.time_masking,
            self.freq_masking,
            self.feature_scaling,
            self.noise_addition,
            self.feature_dropout
        ]

        # Choose a random number of augmentations to apply (1-3)
        num_augs = random.randint(1, 3)

        # Randomly select augmentation functions (without using np.random.choice)
        selected_augs = random.sample(augmentation_functions, num_augs)

        # Apply each augmentation
        for aug_fn in selected_augs:
            feature = aug_fn(feature)

        return feature
