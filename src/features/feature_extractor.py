# src/features/feature_extractor.py
import pandas as pd
import numpy as np
import os
import torch
from tqdm import tqdm
import json
from torch.utils.data import Dataset
# from ..preprocessing.enhanced_audio_processor import EnhancedAudioProcessor as AudioProcessor
from ..preprocessing.audio_processor import AudioProcessor

class FeatureExtractor:
    def __init__(self, metadata_path, output_dir="data/processed", max_length=None):
        """
        Initialize feature extractor

        Args:
            metadata_path: Path to metadata CSV
            output_dir: Directory to save processed features
            max_length: Max length for feature padding/truncation
        """
        self.metadata = pd.read_csv(metadata_path)
        self.processor = AudioProcessor()
        self.output_dir = output_dir
        self.max_length = max_length

        os.makedirs(self.output_dir, exist_ok=True)

    def extract_all_features(self):
        """Extract features for all audio files in metadata"""
        features_list = []
        labels = []

        for idx, row in tqdm(self.metadata.iterrows(), total=len(self.metadata), desc="Extracting features"):
            try:
                # Extract features
                features = self.processor.preprocess(f"{row['file']}", self.max_length)

                # Save individual feature file
                filename = os.path.basename(str(row['file'])).replace('.wav', '.npy')

                output_path = os.path.join(self.output_dir, filename)
                np.save(output_path, features)

                features_list.append(features)
                labels.append(row['intent'])

            except Exception as e:
                print(f"Error processing {row['file']}: {e}")

        # Create dataset dictionary
        dataset = {
            'features': features_list,
            'labels': labels
        }

        # Save dataset
        np.save(os.path.join(self.output_dir, 'features.npy'), features_list)
        np.save(os.path.join(self.output_dir, 'labels.npy'), labels)

        # Save mapping from label to index
        unique_labels = sorted(set(labels))
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        # np.save(os.path.join(self.output_dir, 'label_map.npy'), label_to_idx)
        with open(os.path.join(self.output_dir, 'label_map.json'), 'w') as f:
            json.dump(label_to_idx, f)

        print(f"Feature extraction complete. Saved to {self.output_dir}")
        return dataset, label_to_idx


class TwiDataset(Dataset):
    """PyTorch dataset for Twi audio commands"""

    def __init__(self, features, labels, label_to_idx=None, transform=None):
        """
        Initialize dataset

        Args:
            features: List or array of features
            labels: List of string labels
            label_to_idx: Mapping from string labels to indices
            transform: Optional transform to apply
        """
        super().__init__()
        self.features = features
        self.labels = labels
        self.transform = transform

        if label_to_idx is None:
            unique_labels  = sorted(set(labels))
            self.label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx

        # Convert string labels to indices
        self.label_indices = [self.label_to_idx[label] for label in labels]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.label_indices[idx]

        # Convert to torch tensors
        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.transform:
            feature_tensor = self.transform(feature_tensor)

        return feature_tensor, label_tensor

    def get_num_classes(self):
        """Get the number of classes"""
        return len(self.label_to_idx)
