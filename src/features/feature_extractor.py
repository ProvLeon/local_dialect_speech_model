# src/features/feature_extractor.py
import pandas as pd
import numpy as np
import os
import glob
import torch
from tqdm import tqdm
import json
from typing import Optional, Dict, Any, List
from torch.utils.data import Dataset
# from ..preprocessing.enhanced_audio_processor import EnhancedAudioProcessor as AudioProcessor
from ..preprocessing.audio_processor import AudioProcessor


# ---------------------------------------------------------------------------
# Helper utilities for safe scalar / string checks with pandas objects
# ---------------------------------------------------------------------------

def _non_empty_str(val) -> bool:
    """Return True if val is a non-empty string (after stripping)."""
    return isinstance(val, str) and val.strip() != ""


def _is_valid_scalar(val) -> bool:
    """
    Determine if a value is a usable scalar (non-empty, non-NaN).
    Handles pandas / numpy objects safely without triggering ambiguous truth value errors.
    """
    # Fast path for numeric
    if isinstance(val, (int, float, np.integer, np.floating)):
        # Reject NaN floats
        if isinstance(val, float) and np.isnan(val):
            return False
        return True

    # Strings
    if isinstance(val, str):
        return val.strip() != ""

    # None
    if val is None:
        return False

    # Pandas-specific types
    try:
        if isinstance(val, (pd.Timestamp, pd.Timedelta)):
            return True
        # Use pandas isna if available
        if hasattr(pd, "isna") and pd.isna(val):
            return False
    except Exception:
        # If any issue arises, fall through to final generic check
        pass

    # Generic fallback
    return True


class FeatureExtractor:
    """
    Flexible feature extractor that now supports:
      - Configurable label column (e.g. 'intent' or 'canonical_intent')
      - Skipping commented rows in CSV (lines starting with '#')
      - Capturing slot_type / slot_value columns if present
      - Graceful fallback for audio path resolution when 'file' column is absent
      - Saving parallel slots metadata (slots.json)
    """

    def __init__(
        self,
        metadata_path: str,
        output_dir: str = "data/processed",
        max_length: Optional[int] = None,
        label_column: str = "intent",
        file_column: str = "file",
        id_column: str = "id",
        slot_type_column: str = "slot_type",
        slot_value_column: str = "slot_value",
        audio_root: str = "data/raw",
        audio_extension: str = ".wav"
    ):
        """
        Initialize feature extractor

        Args:
            metadata_path: Path to metadata CSV (supports comments with '#')
            output_dir: Directory to save processed features
            max_length: Max length for feature padding/truncation
            label_column: Column name for class labels (supports 'canonical_intent')
            file_column: Column containing direct audio file paths (if available)
            id_column: Column containing unique identifier (used to build path if file column missing)
            slot_type_column: Column with slot type (optional)
            slot_value_column: Column with slot value (optional)
            audio_root: Root directory for audio if file paths must be constructed
            audio_extension: Default audio file extension used in path construction
        """
        # Read CSV while ignoring commented lines
        self.metadata = pd.read_csv(metadata_path, comment='#')

        self.processor = AudioProcessor()
        self.output_dir = output_dir
        self.max_length = max_length

        self.label_column = label_column
        self.file_column = file_column
        self.id_column = id_column
        self.slot_type_column = slot_type_column
        self.slot_value_column = slot_value_column
        self.audio_root = audio_root
        self.audio_extension = audio_extension

        os.makedirs(self.output_dir, exist_ok=True)

        # Validate label column presence; attempt fallback if not found
        label_cols = list(self.metadata.columns)
        if self.label_column not in label_cols:
            fallback = "canonical_intent" if self.label_column != "canonical_intent" and "canonical_intent" in label_cols else None
            if fallback:
                print(f"[FeatureExtractor] Requested label_column '{self.label_column}' not found. Falling back to '{fallback}'.")
                self.label_column = fallback
            else:
                raise ValueError(
                    f"Label column '{self.label_column}' not found in metadata "
                    f"and no suitable fallback available. Available columns: {label_cols}"
                )

    def _resolve_audio_path(self, row: pd.Series) -> str:
        """
        Resolve audio path with participant-aware search.
        Resolution order:
          1. Direct file_column if provided and exists.
          2. Flat path: audio_root/<id>.wav
          3. Recursive search: audio_root/**/<id>.wav (first match)
        """
        file_value = row.get(self.file_column, None)
        if _non_empty_str(file_value) and os.path.isfile(str(file_value).strip()):
            return str(file_value).strip()
        if _is_valid_scalar(file_value):
            fv = str(file_value)
            if os.path.isfile(fv):
                return fv

        id_value = row.get(self.id_column, None)
        if _non_empty_str(id_value):
            clean_id = str(id_value).strip()
            flat_path = os.path.join(self.audio_root, f"{clean_id}{self.audio_extension}")
            if os.path.isfile(flat_path):
                return flat_path
            # Recursive glob
            pattern = os.path.join(self.audio_root, "**", f"{clean_id}{self.audio_extension}")
            matches = glob.glob(pattern, recursive=True)
            if matches:
                return matches[0]

        raise ValueError(
            f"Cannot resolve audio path for row (id={row.get(self.id_column, '')}) "
            f"using file_column='{self.file_column}' or recursive search under '{self.audio_root}'"
        )

    def extract_all_features(self):
        """Extract features for all audio files in metadata with slot capture."""
        features_list: List[np.ndarray] = []
        labels: List[str] = []
        slots: List[Dict[str, Any]] = []

        # Track missing files for debugging
        missing_files = 0

        for _, row in tqdm(self.metadata.iterrows(), total=len(self.metadata), desc="Extracting features"):
            try:
                audio_path = self._resolve_audio_path(row)

                if not os.path.isfile(audio_path):
                    missing_files += 1
                    print(f"[WARN] Audio file not found: {audio_path}")
                    continue

                # Extract features
                features = self.processor.preprocess(audio_path, self.max_length)

                # Build filename for saved feature
                filename = os.path.basename(audio_path).replace('.wav', '.npy').replace('.mp3', '.npy')
                output_path = os.path.join(self.output_dir, filename)
                np.save(output_path, features)

                # Append feature & label
                label_value = row[self.label_column]
                if not _is_valid_scalar(label_value):
                    continue

                labels.append(str(label_value).strip())
                features_list.append(features)

                # Capture slot & participant info
                slot_entry: Dict[str, Any] = {}
                has_slot_type = self.slot_type_column in list(self.metadata.columns)
                has_slot_value = self.slot_value_column in list(self.metadata.columns)
                slot_type_val = row.get(self.slot_type_column, None) if has_slot_type else None
                slot_value_val = row.get(self.slot_value_column, None) if has_slot_value else None
                if _non_empty_str(slot_type_val):
                    slot_entry["slot_type"] = str(slot_type_val).strip()
                if _non_empty_str(slot_value_val):
                    slot_entry["slot_value"] = str(slot_value_val).strip()
                # Derive participant_id from path structure: audio_root/<participant>/<id>.wav
                try:
                    rel_path = os.path.relpath(audio_path, self.audio_root)
                    parts = rel_path.split(os.sep)
                    if len(parts) >= 2:
                        slot_entry["participant_id"] = parts[0]
                except Exception:
                    pass
                slots.append(slot_entry)

            except Exception as e:
                print(f"[ERROR] Failed processing row id={row.get(self.id_column, 'UNKNOWN')}: {e}")

        if missing_files > 0:
            print(f"[INFO] Skipped {missing_files} rows due to missing audio files.")

        # Create dataset dictionary
        dataset = {
            'features': features_list,
            'labels': labels,
            'slots': slots
        }

        # Save core arrays
        np.save(os.path.join(self.output_dir, 'features.npy'), features_list)
        np.save(os.path.join(self.output_dir, 'labels.npy'), labels)

        # Save slots aligned by index
        with open(os.path.join(self.output_dir, 'slots.json'), 'w') as f:
            json.dump(slots, f, indent=2)

        # Save mapping from label to index (deterministic sorted)
        unique_labels = sorted(set(labels))
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        with open(os.path.join(self.output_dir, 'label_map.json'), 'w') as f:
            json.dump(label_to_idx, f, indent=2)

        print(f"Feature extraction complete. Samples: {len(features_list)} | Classes: {len(unique_labels)} | Saved to {self.output_dir}")
        return dataset, label_to_idx


class TwiDataset(Dataset):
    """PyTorch dataset for Twi audio commands with optional slot attachment."""

    def __init__(
        self,
        features,
        labels,
        label_to_idx=None,
        transform=None,
        slots: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize dataset

        Args:
            features: List or array of features
            labels: List of string labels
            label_to_idx: Mapping from string labels to indices
            transform: Optional transform to apply
            slots: Optional list of slot dictionaries aligned with features
        """
        super().__init__()
        self.features = features
        self.labels = labels
        self.transform = transform
        self.slots = slots if slots is not None else [{} for _ in range(len(features))]

        if label_to_idx is None:
            unique_labels = sorted(set(labels))
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

        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.transform:
            feature_tensor = self.transform(feature_tensor)

        # Returning slots separately (not yet used by model, but available)
        sample_slots = self.slots[idx] if idx < len(self.slots) else {}

        return feature_tensor, label_tensor, sample_slots

    def get_num_classes(self):
        """Get the number of classes"""
        return len(self.label_to_idx)
