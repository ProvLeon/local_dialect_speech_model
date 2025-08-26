#!/usr/bin/env python3
# update_model_with_prompts.py
"""
Script to process Twi prompts CSV and update the model with new intents and training data.
This script coordinates the entire process of updating the model with the comprehensive prompts.
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.prompts_processor import TwiPromptsProcessor
from src.utils.enhanced_dataset_builder import EnhancedTwiDatasetBuilder
from src.features.feature_extractor import FeatureExtractor
from src.utils.training_pipeline import TrainingPipeline
from config.model_config import MODEL_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelUpdater:
    """
    Main class to coordinate updating the model with new prompts
    """

    def __init__(self, prompts_csv="twi_prompts.csv", base_output_dir="data"):
        """
        Initialize the model updater

        Args:
            prompts_csv: Path to the prompts CSV file
            base_output_dir: Base directory for all outputs
        """
        self.prompts_csv = prompts_csv
        self.base_output_dir = base_output_dir

        # Define subdirectories
        self.processed_prompts_dir = os.path.join(base_output_dir, "processed_prompts")
        self.enhanced_raw_dir = os.path.join(base_output_dir, "enhanced_raw")
        self.enhanced_processed_dir = os.path.join(base_output_dir, "enhanced_processed")
        self.enhanced_models_dir = os.path.join(base_output_dir, "models_enhanced")

        # Create directories
        for directory in [self.processed_prompts_dir, self.enhanced_raw_dir,
                         self.enhanced_processed_dir, self.enhanced_models_dir]:
            os.makedirs(directory, exist_ok=True)

        self.processor = None
        self.builder = None
        self.intent_mapping = None
        self.prompts_data = None

    def step1_process_prompts(self):
        """Step 1: Process the prompts CSV file"""
        logger.info("="*60)
        logger.info("STEP 1: Processing Prompts CSV")
        logger.info("="*60)

        if not os.path.exists(self.prompts_csv):
            raise FileNotFoundError(f"Prompts CSV not found: {self.prompts_csv}")

        # Initialize processor (auto-detect lean schema vs legacy)
        is_lean = False
        try:
            with open(self.prompts_csv, 'r', encoding='utf-8') as f:
                header = f.readline().lower()
                if 'canonical_intent' in header:
                    is_lean = True
        except Exception as e:
            logger.warning(f"Could not inspect CSV header: {e}")

        if is_lean:
            try:
                from src.utils.lean_prompts_processor import LeanPromptsProcessor
                logger.info("Detected lean prompts schema (canonical_intent). Using LeanPromptsProcessor.")
                self.processor = LeanPromptsProcessor(
                    csv_path=self.prompts_csv,
                    output_dir=self.processed_prompts_dir
                )
            except ImportError as e:
                logger.error(f"Failed to import LeanPromptsProcessor ({e}). Falling back to legacy TwiPromptsProcessor.")
                self.processor = TwiPromptsProcessor(
                    csv_path=self.prompts_csv,
                    output_dir=self.processed_prompts_dir
                )
        else:
            self.processor = TwiPromptsProcessor(
                csv_path=self.prompts_csv,
                output_dir=self.processed_prompts_dir
            )

        # Process prompts
        saved_files, stats = self.processor.process()

        logger.info(f"✅ Processed {stats['total_prompts']} prompts")
        logger.info(f"✅ Found {stats['unique_intents']} unique intents")
        logger.info(f"✅ Prompts with intents: {stats['prompts_with_intents']}")

        # Load processed data
        self.load_processed_data()

        return stats

    def load_processed_data(self):
        """Load the processed prompts data"""
        # Load intent mapping
        intent_mapping_path = os.path.join(self.processed_prompts_dir, 'intent_mapping.json')
        with open(intent_mapping_path, 'r', encoding='utf-8') as f:
            self.intent_mapping = json.load(f)

        # Load training metadata
        metadata_path = os.path.join(self.processed_prompts_dir, 'training_metadata.json')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.prompts_data = json.load(f)

        logger.info(f"Loaded {len(self.prompts_data)} prompts with {len(self.intent_mapping)} intents")

    def step2_show_prompts_summary(self):
        """Step 2: Display summary of available prompts"""
        logger.info("="*60)
        logger.info("STEP 2: Prompts Summary")
        logger.info("="*60)

        if not self.prompts_data:
            logger.error("No prompts data loaded!")
            return

        # Initialize builder to use its display method
        if not self.builder:
            self.builder = EnhancedTwiDatasetBuilder(
                prompts_csv=self.prompts_csv,
                output_dir=self.enhanced_raw_dir,
                processed_prompts_dir=self.processed_prompts_dir
            )

        self.builder.display_prompts_summary()

    def step3_collect_audio_data(self, interactive=True, participant_id=None):
        """Step 3: Collect audio data using the enhanced participant-based recorder"""
        logger.info("="*60)
        logger.info("STEP 3: Audio Data Collection")
        logger.info("="*60)

        if interactive:
            # Use enhanced prompt recorder with participant management
            if not participant_id:
                participant_id = input("Enter participant ID (e.g., P01): ").strip()
                if not participant_id:
                    participant_id = "P01"  # Default

            logger.info(f"Starting enhanced recording session for participant: {participant_id}")

            try:
                from src.utils.prompt_recorder import EnhancedPromptRecorder

                recorder = EnhancedPromptRecorder(
                    participant_id=participant_id,
                    output_dir=self.enhanced_raw_dir,
                    prompts_file=os.path.join(self.processed_prompts_dir, 'training_metadata.json'),
                    auto_stop=True,
                    vad_aggressiveness=2,
                    silence_ms=500,
                    allow_early_stop=True
                )

                recorder.interactive_recording_session()

            except ImportError as e:
                logger.warning(f"Enhanced recorder not available: {e}")
                logger.info("Falling back to basic dataset builder...")

                if not self.builder:
                    self.builder = EnhancedTwiDatasetBuilder(
                        prompts_csv=self.prompts_csv,
                        output_dir=self.enhanced_raw_dir,
                        processed_prompts_dir=self.processed_prompts_dir
                    )
                self.builder.interactive_recording_session()
        else:
            logger.info("Skipping audio collection (non-interactive mode)")

    def step4_extract_features(self):
        """Step 4: Extract features from recorded audio"""
        logger.info("="*60)
        logger.info("STEP 4: Feature Extraction")
        logger.info("="*60)

        # Check if we have participant recordings (new format)
        participants_file = os.path.join(self.enhanced_raw_dir, "participants.json")
        metadata_csv = os.path.join(self.enhanced_processed_dir, "metadata.csv")

        if os.path.exists(participants_file):
            # New participant-based recording format
            logger.info("Converting participant recordings to training format...")

            try:
                from src.utils.convert_recordings_to_training import RecordingConverter

                converter = RecordingConverter(
                    recordings_dir=self.enhanced_raw_dir,
                    output_dir=self.enhanced_processed_dir
                )

                success = converter.convert()
                if not success:
                    logger.error("Failed to convert participant recordings")
                    return False

                logger.info("✅ Participant recordings converted successfully")

            except ImportError as e:
                logger.error(f"Recording converter not available: {e}")
                return False

        elif os.path.exists(metadata_csv):
            # Already converted or old format
            logger.info("Using existing metadata file")
        else:
            # Check for old format
            old_metadata_csv = os.path.join(self.enhanced_raw_dir, "metadata.csv")
            if os.path.exists(old_metadata_csv):
                logger.info("Found old format metadata, using directly...")
                # Copy to processed directory
                import shutil
                shutil.copy2(old_metadata_csv, metadata_csv)
            else:
                logger.error("No audio metadata found")
                logger.error("Please record audio data first using step 3")
                return False

        # Extract features using the processed metadata
        extractor = FeatureExtractor(
            metadata_path=metadata_csv,
            output_dir=self.enhanced_processed_dir
        )

        dataset, label_map = extractor.extract_all_features()

        logger.info(f"✅ Extracted features for {len(dataset['features'])} samples")
        logger.info(f"✅ Found {len(label_map)} unique labels")

        return True

    def step5_train_model(self, epochs=50):
        """Step 5: Train the enhanced model"""
        logger.info("="*60)
        logger.info("STEP 5: Model Training")
        logger.info("="*60)

        # Check if features exist
        features_path = os.path.join(self.enhanced_processed_dir, "features.npy")
        if not os.path.exists(features_path):
            logger.error(f"No features found at {features_path}")
            logger.error("Please extract features first using step 4")
            return False

        # Update model config with new settings
        config = MODEL_CONFIG.copy()
        config.update({
            'data_dir': self.enhanced_processed_dir,
            'model_dir': self.enhanced_models_dir,
            'num_epochs': epochs,
            'batch_size': 32,
            'learning_rate': 0.001,
            'early_stopping_patience': 15,
            'hidden_dim': 128,
            'dropout': 0.3,
        })

        # Train model
        pipeline = TrainingPipeline(config)
        model, trainer, history = pipeline.run()

        logger.info("✅ Model training completed!")
        return True

    def step6_update_api_config(self):
        """Step 6: Update API configuration to use new model"""
        logger.info("="*60)
        logger.info("STEP 6: Updating API Configuration")
        logger.info("="*60)

        # Update .env file
        env_file = ".env"
        env_updates = {
            'ENHANCED_MODEL_PATH': os.path.join(self.enhanced_models_dir, 'best_model.pt'),
            'MODEL_PATH': os.path.join(self.enhanced_models_dir, 'best_model.pt'),
            'LABEL_MAP_PATH': os.path.join(self.enhanced_processed_dir, 'label_map.npy')
        }

        # Read existing .env
        env_content = {}
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        env_content[key] = value

        # Update with new values
        env_content.update(env_updates)

        # Write updated .env
        with open(env_file, 'w') as f:
            f.write("# Akan Speech Model Configuration\n")
            for key, value in env_content.items():
                f.write(f"{key}={value}\n")

        logger.info(f"✅ Updated {env_file} with new model paths")

        # Create model info summary
        self.create_model_summary()

    def create_model_summary(self):
        """Create a summary of the updated model"""
        summary = {
            'model_update_info': {
                'prompts_csv': self.prompts_csv,
                'total_intents': len(self.intent_mapping) if self.intent_mapping else 0,
                'directories': {
                    'processed_prompts': self.processed_prompts_dir,
                    'enhanced_raw': self.enhanced_raw_dir,
                    'enhanced_processed': self.enhanced_processed_dir,
                    'enhanced_models': self.enhanced_models_dir
                },
                'intent_mapping': self.intent_mapping
            }
        }

        summary_path = os.path.join(self.enhanced_models_dir, 'model_update_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ Created model summary at {summary_path}")

    def run_complete_pipeline(self, interactive_audio=True, epochs=50, participant_id=None):
        """Run the complete model update pipeline"""
        logger.info("🚀 Starting Complete Model Update Pipeline")
        logger.info("="*60)

        try:
            # Step 1: Process prompts
            stats = self.step1_process_prompts()

            # Step 2: Show prompts summary
            self.step2_show_prompts_summary()

            # Step 3: Collect audio data
            if interactive_audio:
                self.step3_collect_audio_data(interactive=True, participant_id=participant_id)
            else:
                logger.info("Skipping audio collection (use --collect-audio to enable)")

            # Step 4: Extract features (check both old and new formats)
            has_participant_data = os.path.exists(os.path.join(self.enhanced_raw_dir, "participants.json"))
            has_old_metadata = os.path.exists(os.path.join(self.enhanced_raw_dir, "metadata.csv"))
            has_processed_metadata = os.path.exists(os.path.join(self.enhanced_processed_dir, "metadata.csv"))

            if has_participant_data or has_old_metadata or has_processed_metadata:
                if self.step4_extract_features():
                    # Step 5: Train model
                    if self.step5_train_model(epochs=epochs):
                        # Step 6: Update API config
                        self.step6_update_api_config()
                        logger.info("🎉 Complete pipeline finished successfully!")
                    else:
                        logger.error("❌ Model training failed")
                else:
                    logger.error("❌ Feature extraction failed")
            else:
                logger.info("⚠️ No audio data found. Pipeline completed up to prompts processing.")

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()

    def run_individual_step(self, step: str, **kwargs):
        """Run an individual step of the pipeline"""
        step_map = {
            'process-prompts': self.step1_process_prompts,
            'show-summary': self.step2_show_prompts_summary,
            'collect-audio': lambda: self.step3_collect_audio_data(
                interactive=True,
                participant_id=kwargs.get('participant_id')
            ),
            'extract-features': self.step4_extract_features,
            'train-model': lambda: self.step5_train_model(epochs=kwargs.get('epochs', 50)),
            'update-config': self.step6_update_api_config,
            'convert-recordings': self.convert_participant_recordings
        }

        if step in step_map:
            logger.info(f"Running step: {step}")
            step_map[step]()
        else:
            logger.error(f"Unknown step: {step}")
            logger.info(f"Available steps: {list(step_map.keys())}")

    def convert_participant_recordings(self):
        """Convert participant recordings to training format"""
        logger.info("="*60)
        logger.info("STEP: Converting Participant Recordings")
        logger.info("="*60)

        try:
            from src.utils.convert_recordings_to_training import RecordingConverter

            converter = RecordingConverter(
                recordings_dir=self.enhanced_raw_dir,
                output_dir=self.enhanced_processed_dir
            )

            success = converter.convert()
            if success:
                logger.info("✅ Participant recordings converted successfully")
            else:
                logger.error("❌ Failed to convert participant recordings")

        except ImportError as e:
            logger.error(f"Recording converter not available: {e}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Update Akan speech model with comprehensive prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline with interactive audio collection
  python update_model_with_prompts.py --complete --collect-audio

  # Process prompts only
  python update_model_with_prompts.py --step process-prompts

  # Train model with specific epochs
  python update_model_with_prompts.py --step train-model --epochs 100

  # Show prompts summary
  python update_model_with_prompts.py --step show-summary
        """
    )

    parser.add_argument("--prompts-csv", type=str, default="twi_prompts.csv",
                       help="Path to prompts CSV file (default: twi_prompts.csv)")
    parser.add_argument("--output-dir", type=str, default="data",
                       help="Base output directory (default: data)")
    parser.add_argument("--complete", action="store_true",
                       help="Run complete pipeline")
    parser.add_argument("--collect-audio", action="store_true",
                       help="Include interactive audio collection in complete pipeline")
    parser.add_argument("--participant", type=str,
                       help="Participant ID for recording (e.g., P01, P02)")
    parser.add_argument("--step", type=str,
                       choices=['process-prompts', 'show-summary', 'collect-audio',
                               'extract-features', 'train-model', 'update-config', 'convert-recordings'],
                       help="Run individual step")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs (default: 50)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate inputs
    if not os.path.exists(args.prompts_csv):
        logger.error(f"Prompts CSV file not found: {args.prompts_csv}")
        sys.exit(1)

    # Initialize updater
    updater = ModelUpdater(
        prompts_csv=args.prompts_csv,
        base_output_dir=args.output_dir
    )

    # Run appropriate action
    if args.complete:
        updater.run_complete_pipeline(
            interactive_audio=args.collect_audio,
            epochs=args.epochs,
            participant_id=args.participant
        )
    elif args.step:
        updater.run_individual_step(
            args.step,
            epochs=args.epochs,
            participant_id=args.participant
        )
    else:
        parser.print_help()
        print("\nAvailable options:")
        print("  --complete           Run the complete pipeline")
        print("  --step STEP_NAME     Run individual step")
        print("  --collect-audio      Include audio collection (use with --complete)")


if __name__ == "__main__":
    main()
