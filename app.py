# app.py
import os
import argparse
# import torch
import logging
import uvicorn
from dotenv import load_dotenv
# Defer importing the FastAPI app until after optional MODEL_BASE_DIR override
# to allow dynamic model directory selection at runtime.
import importlib
from typing import Optional

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_api_server(host="0.0.0.0",
                   port=8000,
                   reload: bool = False,
                   workers: int = 1,
                   log_level: str = "info",
                   root_path: str = "",
                   model_base_dir: Optional[str] = None):
    """
    Run the FastAPI server with improved configurability and dynamic model directory override.
    If model_base_dir is provided, it sets/overrides the MODEL_BASE_DIR environment variable
    BEFORE importing the API module so that model discovery uses that directory.
    """
    if model_base_dir:
        os.environ["MODEL_BASE_DIR"] = model_base_dir
        logger.info(f"MODEL_BASE_DIR overridden via CLI: {model_base_dir}")
    else:
        logger.info("No MODEL_BASE_DIR override provided (using existing environment or defaults).")

    logger.info(
        f"Starting API server on {host}:{port} "
        f"(reload={reload}, workers={workers}, log_level={log_level}, root_path='{root_path}', "
        f"model_base_dir='{model_base_dir}')"
    )

    if reload:
        # For reload mode, use import string so uvicorn can properly reload
        uvicorn.run(
            "src.api.speech_api:app",
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
            root_path=root_path
        )
    else:
        # For production mode, import the app object directly
        speech_api = importlib.import_module("src.api.speech_api")
        fastapi_app = getattr(speech_api, "app")

        uvicorn.run(
            fastapi_app,
            host=host,
            port=port,
            reload=reload,
            workers=workers,
            log_level=log_level,
            root_path=root_path
        )

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

def run_47_class_test(model_path: str, label_map: Optional[str] = None, audio_file: Optional[str] = None, duration: int = 3, loop: bool = False):
    """
    Run the 47-class (or auto-detected class count) model test utility from the CLI.

    Args:
        model_path: Path to the trained model checkpoint (.pt)
        label_map: Optional path to label_map.json
        audio_file: If provided, run a single prediction on this file; otherwise enter live mode
        duration: Recording duration for live audio
        loop: If True and no audio_file is provided, continue testing in a loop
    """
    try:
        from test_trained_47_class_model import Model47ClassTester
    except ImportError as e:
        logger.error("Could not import Model47ClassTester. Ensure test_trained_47_class_model.py is present.")
        raise e

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    tester = Model47ClassTester(model_path, label_map)
    logger.info(f"Loaded model for testing (num_classes={tester.num_classes})")

    if audio_file:
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        logger.info(f"Running single-file test on {audio_file}")
        intent, confidence = tester.test_audio_file(audio_file)
        logger.info(f"Prediction: {intent} ({confidence:.2%})")
        return {"intent": intent, "confidence": confidence, "mode": "file"}
    else:
        logger.info("Entering live audio testing mode")
        try:
            while True:
                tester.test_live_audio(duration)
                if not loop:
                    break
        except KeyboardInterrupt:
            logger.info("Live testing interrupted by user")
        return {"mode": "live", "loop": loop}

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

    # 47-class (dynamic) model test command
    test47_parser = subparsers.add_parser("test-model", help="Run 47-class (or dynamic) model test utility")
    test47_parser.add_argument("--model", required=True, help="Path to best_model.pt (or checkpoint)")
    test47_parser.add_argument("--label-map", help="Optional label_map.json path")
    test47_parser.add_argument("--file", help="Optional single audio file to test")
    test47_parser.add_argument("--duration", type=int, default=3, help="Live recording duration (seconds)")
    test47_parser.add_argument("--loop", action="store_true", help="Loop live testing until interrupted")

    # API server extended options
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    api_parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    api_parser.add_argument("--log-level", type=str, default="info", choices=["critical", "error", "warning", "info", "debug", "trace"], help="Uvicorn log level")
    api_parser.add_argument("--root-path", type=str, default="", help="Root path for reverse proxy mounting")
    api_parser.add_argument("--model-base-dir", type=str, help="Override base directory for model + label map discovery (sets MODEL_BASE_DIR)")

    args = parser.parse_args()

    # Run appropriate function based on command
    if args.command == "api":
        run_api_server(
            host=args.host,
            port=args.port,
            reload=getattr(args, "reload", False),
            workers=getattr(args, "workers", 1),
            log_level=getattr(args, "log_level", "info"),
            root_path=getattr(args, "root_path", ""),
            model_base_dir=getattr(args, "model_base_dir", None)
        )
    elif args.command == "train":
        run_training(args.data_dir, args.model_dir, args.epochs)
    elif args.command == "train-enhanced":
        run_enhanced_training(args.data_dir, args.model_dir, args.epochs)
    elif args.command == "test-model":
        run_47_class_test(
            model_path=args.model,
            label_map=args.label_map,
            audio_file=args.file,
            duration=args.duration,
            loop=args.loop
        )
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
