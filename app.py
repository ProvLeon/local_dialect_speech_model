# app.py
import os
import argparse
# import torch
import logging
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_api_server(host="0.0.0.0", port=8000):
    """Run the FastAPI server"""
    from src.api.speech_api import app

    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

def run_training(data_dir="data/processed", model_dir="data/models", epochs=20):
    """Run model training with the enhanced training pipeline"""
    from src.utils.training_pipeline import TrainingPipeline
    from config.model_config import MODEL_CONFIG

    # Update config with command line arguments
    config = MODEL_CONFIG.copy()
    config.update({
        'data_dir': data_dir,
        'model_dir': model_dir,
        'num_epochs': epochs
    })

    logger.info("Starting training pipeline...")

    # Create and run the training pipeline
    pipeline = TrainingPipeline(config)
    model, trainer, history = pipeline.run()

    logger.info("Training complete!")
    return model

def run_data_collection():
    """Run data collection interface"""
    from src.utils.dataset_builder import TwiDatasetBuilder

    logger.info("Starting data collection...")

    builder = TwiDatasetBuilder()
    try:
        builder.collect_dataset()
    except KeyboardInterrupt:
        logger.info("Data collection interrupted.")

def run_feature_extraction(metadata_path="data/raw/metadata.csv", output_dir="data/processed"):
    """Run feature extraction"""
    from src.features.feature_extractor import FeatureExtractor

    logger.info("Starting feature extraction...")

    extractor = FeatureExtractor(metadata_path, output_dir)
    extractor.extract_all_features()

    logger.info("Feature extraction complete!")

def run_augmentation(metadata_path="data/raw/metadata.csv", output_dir="data/augmented", factor=3, balanced=True):
    """Run advanced data augmentation"""
    from src.utils.data_augmenter import AdvancedDataAugmenter

    logger.info("Starting advanced data augmentation...")

    augmenter = AdvancedDataAugmenter(metadata_path, output_dir)
    generated_files, new_metadata = augmenter.augment_dataset(
        augmentation_factor=factor,
        balanced=balanced
    )

    logger.info(f"Augmentation complete! Generated {len(generated_files)} new audio files.")
    return generated_files, new_metadata

def run_enhanced_training(data_dir="data/processed", model_dir="data/models/enhanced", epochs=100):
    """Run enhanced model training"""
    from train_enhanced_model import train_enhanced_model

    logger.info("Starting enhanced model training...")
    model, trainer, test_acc = train_enhanced_model(data_dir, model_dir, epochs)
    logger.info(f"Enhanced model training complete with {test_acc:.2f}% test accuracy!")

    return model, trainer, test_acc

def main():
    """Main function to parse arguments and run appropriate function"""
    parser = argparse.ArgumentParser(description="Akan (Twi) Speech-to-Action System")

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # API server command
    api_parser = subparsers.add_parser("api", help="Run API server")
    api_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to bind")

    # Training command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--data-dir", type=str, default="data/processed", help="Data directory")
    train_parser.add_argument("--model-dir", type=str, default="data/models", help="Model directory")
    train_parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    train_parser.add_argument("--model-type", type=str, default="improved", choices=["standard", "improved"], help="Model type to train")

    # Data collection command
    subparsers.add_parser("collect", help="Collect audio data")

    # Feature extraction command
    extract_parser = subparsers.add_parser("extract", help="Extract features from audio")
    extract_parser.add_argument("--metadata", type=str, default="data/raw/metadata.csv", help="Metadata file")
    extract_parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory")
    extract_parser.add_argument("--feature-type", type=str, default="combined", choices=["mfcc", "melspec", "combined"], help="Feature type to extract")

    # Data augmentation command (NEW)
    augment_parser = subparsers.add_parser("augment", help="Augment audio dataset")
    augment_parser.add_argument("--metadata", type=str, default="data/raw/metadata.csv", help="Metadata file")
    augment_parser.add_argument("--output-dir", type=str, default="data/augmented", help="Output directory")
    augment_parser.add_argument("--factor", type=int, default=3, help="Augmentation factor")
    augment_parser.add_argument("--balanced", action="store_true", help="Balance classes through augmentation")

    # Enhanced training command
    enhanced_train_parser = subparsers.add_parser("train-enhanced", help="Train enhanced model")
    enhanced_train_parser.add_argument("--data-dir", type=str, default="data/processed", help="Data directory")
    enhanced_train_parser.add_argument("--model-dir", type=str, default="data/models/enhanced", help="Model directory")
    enhanced_train_parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")

    args = parser.parse_args()

    # Run appropriate function based on command
    if args.command == "api":
        run_api_server(args.host, args.port)
    elif args.command == "train":
        run_training(args.data_dir, args.model_dir, args.epochs)
    elif args.command == "train-enhanced":
        run_enhanced_training(args.data_dir, args.model_dir, args.epochs)
    elif args.command == "collect":
        run_data_collection()
    elif args.command == "extract":
        run_feature_extraction(args.metadata, args.output_dir)
    elif args.command == "augment":
        run_augmentation(args.metadata, args.output_dir, args.factor, args.balanced)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
