#!/usr/bin/env python3
"""
extract_features_from_manifest.py

Extract audio features from a JSONL manifest file and prepare them for model training.

This script reads a JSONL manifest (like the one created by build_audio_manifest_multisample.py)
and extracts features from each audio file, saving them in the format expected by the
training pipeline.

Input:
  - JSONL manifest file with fields: instance_id, base_id, participant_id, sample_number,
    audio_path, intent, text, slot_type, slot_value

Output:
  - features.npy: Array of extracted features
  - labels.npy: Array of intent labels
  - slots.json: Slot information aligned with features
  - label_map.json: Mapping from intent strings to numeric indices

Usage:
  python extract_features_from_manifest.py \
    --manifest data/lean_dataset/audio_manifest_multisample.jsonl \
    --output-dir data/processed_lean_multisample \
    --max-length 16000
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.preprocessing.audio_processor import AudioProcessor


def load_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    """Load JSONL manifest file."""
    entries = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def extract_features_from_manifest(
    manifest_path: str,
    output_dir: str,
    max_length: Optional[int] = None,
    verbose: bool = False
):
    """
    Extract features from all audio files listed in the manifest.

    Args:
        manifest_path: Path to JSONL manifest file
        output_dir: Directory to save extracted features
        max_length: Maximum length for audio features (samples)
        verbose: Print progress information
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize audio processor
    processor = AudioProcessor()

    # Load manifest
    print(f"[INFO] Loading manifest from: {manifest_path}")
    entries = load_manifest(manifest_path)
    print(f"[INFO] Found {len(entries)} audio instances in manifest")

    # Initialize collections
    features_list: List[np.ndarray] = []
    labels: List[str] = []
    slots: List[Dict[str, Any]] = []

    # Track statistics
    missing_files = 0
    processed_files = 0

    # Process each entry
    for entry in tqdm(entries, desc="Extracting features"):
        try:
            audio_path = entry.get('audio_path')
            if not audio_path:
                if verbose:
                    print(f"[WARN] No audio_path for entry: {entry.get('instance_id', 'unknown')}")
                continue

            # Check if audio file exists
            if not os.path.isfile(audio_path):
                missing_files += 1
                if verbose:
                    print(f"[WARN] Audio file not found: {audio_path}")
                continue

            # Extract features
            features = processor.preprocess(audio_path, max_length)

            # Get intent label
            intent = entry.get('intent')
            if not intent:
                if verbose:
                    print(f"[WARN] No intent for entry: {entry.get('instance_id', 'unknown')}")
                continue

            # Add to collections
            features_list.append(features)
            labels.append(intent)

            # Create slot information
            slot_info = {}
            if entry.get('slot_type'):
                slot_info['slot_type'] = entry['slot_type']
            if entry.get('slot_value'):
                slot_info['slot_value'] = entry['slot_value']
            if entry.get('participant_id'):
                slot_info['participant_id'] = entry['participant_id']
            if entry.get('sample_number'):
                slot_info['sample_number'] = entry['sample_number']
            if entry.get('base_id'):
                slot_info['base_id'] = entry['base_id']
            if entry.get('instance_id'):
                slot_info['instance_id'] = entry['instance_id']

            slots.append(slot_info)
            processed_files += 1

        except Exception as e:
            if verbose:
                print(f"[ERROR] Failed processing entry {entry.get('instance_id', 'unknown')}: {e}")
            continue

    if processed_files == 0:
        print("[ERROR] No features were successfully extracted!")
        return

    print(f"[INFO] Successfully processed {processed_files} files")
    if missing_files > 0:
        print(f"[WARN] Skipped {missing_files} files due to missing audio")

    # Save features
    features_path = os.path.join(output_dir, 'features.npy')
    np.save(features_path, features_list)
    print(f"[INFO] Saved features to: {features_path}")

    # Save labels
    labels_path = os.path.join(output_dir, 'labels.npy')
    np.save(labels_path, labels)
    print(f"[INFO] Saved labels to: {labels_path}")

    # Save slots
    slots_path = os.path.join(output_dir, 'slots.json')
    with open(slots_path, 'w', encoding='utf-8') as f:
        json.dump(slots, f, indent=2)
    print(f"[INFO] Saved slots to: {slots_path}")

    # Create label mapping
    unique_labels = sorted(set(labels))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}

    label_map_path = os.path.join(output_dir, 'label_map.json')
    with open(label_map_path, 'w', encoding='utf-8') as f:
        json.dump(label_to_idx, f, indent=2)
    print(f"[INFO] Saved label map to: {label_map_path}")

    # Print summary
    print(f"\n[SUMMARY]")
    print(f"  Total samples: {len(features_list)}")
    print(f"  Number of classes: {len(unique_labels)}")
    print(f"  Feature shape: {features_list[0].shape if features_list else 'N/A'}")
    print(f"  Output directory: {output_dir}")

    # Print class distribution
    from collections import Counter
    label_counts = Counter(labels)
    print(f"\n[CLASS DISTRIBUTION]")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    return {
        'features': features_list,
        'labels': labels,
        'slots': slots,
        'label_map': label_to_idx
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from JSONL manifest for model training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to JSONL manifest file"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed_lean_multisample",
        help="Output directory for processed features"
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum length for audio features (samples)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information"
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.manifest):
        print(f"[ERROR] Manifest file not found: {args.manifest}")
        sys.exit(1)

    # Extract features
    try:
        extract_features_from_manifest(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            max_length=args.max_length,
            verbose=args.verbose
        )
        print("\n[OK] Feature extraction complete!")

    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
