#!/usr/bin/env python3
"""
Convert Participant Recordings to Training Format

This script converts the enhanced participant-based recordings to the format
expected by the model training pipeline.
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RecordingConverter:
    """Convert participant recordings to training format"""

    def __init__(self, recordings_dir="data/recordings", output_dir="data/enhanced_processed"):
        """
        Initialize recording converter

        Args:
            recordings_dir: Directory containing participant recordings
            output_dir: Output directory for training data
        """
        self.recordings_dir = Path(recordings_dir)
        self.output_dir = Path(output_dir)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for collected data
        self.recordings_data = []
        self.participants_info = {}
        self.intent_mapping = {}

    def load_participants_data(self):
        """Load participant information"""
        participants_file = self.recordings_dir / "participants.json"

        if participants_file.exists():
            try:
                with open(participants_file, 'r', encoding='utf-8') as f:
                    self.participants_info = json.load(f)
                logger.info(f"Loaded data for {len(self.participants_info)} participants")
            except Exception as e:
                logger.error(f"Error loading participants data: {e}")
        else:
            logger.warning("No participants.json found, will scan directories")

    def scan_participant_directories(self):
        """Scan participant directories for recordings"""
        if not self.recordings_dir.exists():
            logger.error(f"Recordings directory not found: {self.recordings_dir}")
            return

        participant_dirs = [d for d in self.recordings_dir.iterdir()
                          if d.is_dir() and d.name != '__pycache__']

        logger.info(f"Found {len(participant_dirs)} participant directories")

        for participant_dir in participant_dirs:
            participant_id = participant_dir.name
            logger.info(f"Processing participant: {participant_id}")

            # Scan for session files
            session_files = list(participant_dir.glob("session_*.json"))
            audio_files = list(participant_dir.glob("*.wav"))

            logger.info(f"  Found {len(session_files)} session files, {len(audio_files)} audio files")

            # Process session files (preferred method)
            if session_files:
                for session_file in session_files:
                    self.process_session_file(session_file, participant_id)
            else:
                # Fallback: process audio files directly
                logger.warning(f"No session files found for {participant_id}, processing audio files directly")
                self.process_audio_files_directly(participant_dir, participant_id)

    def process_session_file(self, session_file, participant_id):
        """Process a session file to extract recording information"""
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)

            recordings = session_data.get('recordings', [])
            logger.info(f"  Session {session_file.name}: {len(recordings)} recordings")

            for recording in recordings:
                # Validate file exists
                audio_path = Path(recording['filepath'])
                if not audio_path.exists():
                    logger.warning(f"Audio file not found: {audio_path}")
                    continue

                # Add participant information
                recording['participant_id'] = participant_id
                recording['session_file'] = session_file.name

                # Ensure required fields
                if not all(key in recording for key in ['text', 'intent', 'filepath']):
                    logger.warning(f"Missing required fields in recording: {recording}")
                    continue

                self.recordings_data.append(recording)

        except Exception as e:
            logger.error(f"Error processing session file {session_file}: {e}")

    def process_audio_files_directly(self, participant_dir, participant_id):
        """Process audio files directly when no session files available"""
        audio_files = list(participant_dir.glob("*.wav"))

        for audio_file in audio_files:
            # Extract information from filename
            # Format: {intent}_{safe_text}_s{sample_number}_{timestamp}.wav
            filename = audio_file.stem
            parts = filename.split('_')

            if len(parts) >= 3:
                intent = parts[0]
                # Reconstruct text from middle parts (excluding sample and timestamp)
                text_parts = parts[1:-2] if len(parts) > 3 else [parts[1]]
                text = ' '.join(text_parts).replace('_', ' ')

                recording = {
                    'filename': audio_file.name,
                    'filepath': str(audio_file),
                    'text': text,
                    'intent': intent,
                    'participant_id': participant_id,
                    'meaning': '',  # Not available from filename
                    'section': '',  # Not available from filename
                    'sample_number': 1,  # Default
                    'duration': 0.0,  # Will be calculated if needed
                }

                self.recordings_data.append(recording)
            else:
                logger.warning(f"Could not parse filename: {audio_file.name}")

    def create_intent_mapping(self):
        """Create intent to index mapping"""
        intents = set(recording['intent'] for recording in self.recordings_data)
        intents = sorted(list(intents))

        self.intent_mapping = {intent: idx for idx, intent in enumerate(intents)}

        logger.info(f"Created intent mapping with {len(self.intent_mapping)} intents:")
        for intent, idx in self.intent_mapping.items():
            count = len([r for r in self.recordings_data if r['intent'] == intent])
            logger.info(f"  {idx}: {intent} ({count} recordings)")

    def create_training_metadata(self):
        """Create training metadata compatible with the training pipeline"""
        training_metadata = []

        for recording in self.recordings_data:
            metadata_entry = {
                'file': recording['filepath'],
                'text': recording['text'],
                'intent': recording['intent'],
                'meaning': recording.get('meaning', ''),
                'section': recording.get('section', ''),
                'participant_id': recording['participant_id'],
                'sample_number': recording.get('sample_number', 1),
                'duration': recording.get('duration', 0.0),
                'session_file': recording.get('session_file', ''),
                'timestamp': recording.get('timestamp', '')
            }
            training_metadata.append(metadata_entry)

        return training_metadata

    def save_training_data(self):
        """Save data in training pipeline format"""
        # Create training metadata
        training_metadata = self.create_training_metadata()

        # Save as CSV (compatible with FeatureExtractor)
        df = pd.DataFrame(training_metadata)
        csv_path = self.output_dir / "metadata.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"Saved training CSV: {csv_path}")

        # Save as JSON for detailed information
        json_path = self.output_dir / "recordings_metadata.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(training_metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved detailed JSON: {json_path}")

        # Save intent mapping
        intent_map_json_path = self.output_dir / "label_map.json"
        with open(intent_map_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.intent_mapping, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved intent mapping: {intent_map_json_path}")

        # Save as numpy for compatibility
        intent_map_npy_path = self.output_dir / "label_map.npy"
        np.save(intent_map_npy_path, self.intent_mapping)
        logger.info(f"Saved numpy intent mapping: {intent_map_npy_path}")

        return training_metadata

    def generate_statistics(self):
        """Generate comprehensive statistics about the recordings"""
        stats = {
            'total_recordings': len(self.recordings_data),
            'unique_participants': len(set(r['participant_id'] for r in self.recordings_data)),
            'unique_intents': len(self.intent_mapping),
            'unique_texts': len(set(r['text'] for r in self.recordings_data)),
        }

        # Participant distribution
        participant_counts = defaultdict(int)
        for recording in self.recordings_data:
            participant_counts[recording['participant_id']] += 1
        stats['participant_distribution'] = dict(participant_counts)

        # Intent distribution
        intent_counts = defaultdict(int)
        for recording in self.recordings_data:
            intent_counts[recording['intent']] += 1
        stats['intent_distribution'] = dict(intent_counts)

        # Section distribution (if available)
        section_counts = defaultdict(int)
        for recording in self.recordings_data:
            section = recording.get('section', 'Unknown')
            if section:
                section_counts[section] += 1
        if section_counts:
            stats['section_distribution'] = dict(section_counts)

        # Sample distribution per text
        text_counts = defaultdict(int)
        for recording in self.recordings_data:
            text_counts[recording['text']] += 1
        stats['samples_per_text'] = {
            'min': min(text_counts.values()) if text_counts else 0,
            'max': max(text_counts.values()) if text_counts else 0,
            'avg': sum(text_counts.values()) / len(text_counts) if text_counts else 0
        }

        # Duration statistics (if available)
        durations = [r.get('duration', 0) for r in self.recordings_data if r.get('duration', 0) > 0]
        if durations:
            stats['duration_stats'] = {
                'total_duration': sum(durations),
                'avg_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations)
            }

        return stats

    def save_statistics(self, stats):
        """Save statistics to file"""
        stats_path = self.output_dir / "conversion_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved statistics: {stats_path}")

    def print_summary(self, stats):
        """Print conversion summary"""
        print("\n" + "="*60)
        print("RECORDING CONVERSION SUMMARY")
        print("="*60)
        print(f"Total Recordings: {stats['total_recordings']}")
        print(f"Unique Participants: {stats['unique_participants']}")
        print(f"Unique Intents: {stats['unique_intents']}")
        print(f"Unique Texts: {stats['unique_texts']}")

        if 'duration_stats' in stats:
            duration = stats['duration_stats']
            print(f"Total Duration: {duration['total_duration']:.1f} seconds")
            print(f"Average Duration: {duration['avg_duration']:.2f} seconds")

        print(f"\nSamples per Text:")
        samples = stats['samples_per_text']
        print(f"  Min: {samples['min']}")
        print(f"  Max: {samples['max']}")
        print(f"  Average: {samples['avg']:.1f}")

        print(f"\nTop Participants:")
        for participant, count in sorted(stats['participant_distribution'].items(),
                                       key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {participant}: {count} recordings")

        print(f"\nTop Intents:")
        for intent, count in sorted(stats['intent_distribution'].items(),
                                  key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {intent}: {count} recordings")

        print(f"\nOutput Directory: {self.output_dir}")
        print("="*60)

    def convert(self):
        """Main conversion process"""
        logger.info("Starting recording conversion...")

        # Load participant data
        self.load_participants_data()

        # Scan and process recordings
        self.scan_participant_directories()

        if not self.recordings_data:
            logger.error("No recordings found to convert!")
            return False

        # Create intent mapping
        self.create_intent_mapping()

        # Save training data
        training_metadata = self.save_training_data()

        # Generate and save statistics
        stats = self.generate_statistics()
        self.save_statistics(stats)

        # Print summary
        self.print_summary(stats)

        logger.info("Conversion completed successfully!")
        return True


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Convert participant recordings to training format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert recordings from default directory
  python src/utils/convert_recordings_to_training.py

  # Convert with custom paths
  python src/utils/convert_recordings_to_training.py --recordings-dir data/recordings --output-dir data/enhanced_processed

  # Convert and show verbose output
  python src/utils/convert_recordings_to_training.py --verbose
        """
    )

    parser.add_argument("--recordings-dir", type=str, default="data/recordings",
                       help="Directory containing participant recordings")
    parser.add_argument("--output-dir", type=str, default="data/enhanced_processed",
                       help="Output directory for training data")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize converter
    converter = RecordingConverter(
        recordings_dir=args.recordings_dir,
        output_dir=args.output_dir
    )

    # Run conversion
    success = converter.convert()

    if success:
        print("\n✅ Conversion completed successfully!")
        print(f"Training data saved to: {args.output_dir}")
        print("\nNext steps:")
        print("1. Extract features: python update_model_with_prompts.py --step extract-features")
        print("2. Train model: python update_model_with_prompts.py --step train-model")
    else:
        print("❌ Conversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
