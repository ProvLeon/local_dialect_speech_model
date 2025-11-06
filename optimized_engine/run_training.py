#!/usr/bin/env python3
"""
Simple Run Script for Twi Speech Engine Training
===============================================

This script provides an easy way to train both Whisper and Intent Classifier
models using your Twi audio data and prompts.

Usage:
    python run_training.py                    # Full training pipeline
    python run_training.py --quick            # Quick training for testing
    python run_training.py --whisper-only     # Only train Whisper
    python run_training.py --intent-only      # Only train Intent Classifier

Author: AI Assistant
Date: 2025-11-05
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print training banner."""
    print("\n" + "=" * 70)
    print("        TWI SPEECH RECOGNITION - TRAINING PIPELINE")
    print("=" * 70)
    print("🎯 Goal: Fine-tune Whisper + Train Intent Classifier")
    print("📊 Data: Your recorded Twi audio + prompts CSV")
    print("🚀 Result: Production-ready speech recognition system")
    print("=" * 70)


def check_data_availability():
    """Check if required data is available."""
    logger.info("🔍 Checking data availability...")

    base_dir = Path(__file__).parent
    project_root = base_dir.parent
    data_dir = project_root / "data" / "raw"
    prompts_file = project_root / "prompts_lean.csv"

    issues = []

    # Check audio data
    if not data_dir.exists():
        issues.append(f"Audio data directory not found: {data_dir}")
    else:
        audio_files = list(data_dir.rglob("*.wav")) + list(data_dir.rglob("*.mp3"))
        if len(audio_files) == 0:
            issues.append(f"No audio files found in {data_dir}")
        else:
            logger.info(f"✅ Found {len(audio_files)} audio files")

    # Check prompts file
    if not prompts_file.exists():
        issues.append(f"Prompts file not found: {prompts_file}")
    else:
        try:
            import pandas as pd

            df = pd.read_csv(prompts_file, comment="#")
            logger.info(f"✅ Found {len(df)} prompts in CSV")
        except Exception as e:
            issues.append(f"Invalid prompts file: {e}")

    if issues:
        logger.error("❌ Data check failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False

    logger.info("✅ All required data found")
    return True


def install_dependencies():
    """Install required dependencies from requirements_clean.txt."""
    logger.info("📦 Installing training dependencies from requirements_clean.txt...")
    
    requirements_file = Path(__file__).parent / "requirements_clean.txt"
    if not requirements_file.exists():
        logger.error(f"❌ {requirements_file} not found!")
        return False

    try:
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600, # Increased timeout for pip install
        )

        if result.returncode != 0:
            logger.error(f"⚠️ Failed to install dependencies: {result.stderr}")
            return False
        else:
            logger.info(f"✅ Dependencies installed successfully.")
            return True

    except Exception as e:
        logger.error(f"❌ Dependency installation failed: {e}")
        return False


def train_whisper_model(quick=False):
    """Train Whisper model on Twi data."""
    logger.info("🎤 Starting Whisper fine-tuning...")

    base_dir = Path(__file__).parent
    script_path = base_dir / "train_whisper_twi.py"

    if not script_path.exists():
        logger.error(f"❌ Training script not found: {script_path}")
        return False

    # Training parameters
    model_size = "tiny" if quick else "small"
    epochs = 3 if quick else 10
    batch_size = 4 if quick else 8
    eval_steps = 50 if quick else 500

    cmd = [
        sys.executable,
        str(script_path),
        "--model_size",
        model_size,
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--eval_steps",
        str(eval_steps),
        "--data_dir",
        "../data/raw",
        "--prompts_file",
        "../prompts_lean.csv",
        "--output_dir",
        "./models/whisper_twi",
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    logger.info(
        f"Training config: model={model_size}, epochs={epochs}, batch_size={batch_size}"
    )

    try:
        start_time = time.time()
        result = subprocess.run(cmd, timeout=7200)  # 2 hour timeout

        if result.returncode == 0:
            elapsed = time.time() - start_time
            logger.info(f"✅ Whisper training completed in {elapsed / 60:.1f} minutes")
            return True
        else:
            logger.error("❌ Whisper training failed")
            return False

    except subprocess.TimeoutExpired:
        logger.error("❌ Whisper training timed out (2 hours)")
        return False
    except Exception as e:
        logger.error(f"❌ Whisper training failed: {e}")
        return False


def train_intent_classifier(quick=False):
    """Train intent classification model."""
    logger.info("🎯 Starting Intent Classifier training...")

    base_dir = Path(__file__).parent
    script_path = base_dir / "train_intent_classifier.py"

    if not script_path.exists():
        logger.error(f"❌ Training script not found: {script_path}")
        return False

    epochs = 3 if quick else 10

    cmd = [
        sys.executable,
        str(script_path),
        "--data",
        "../prompts_lean.csv",
        "--output",
        "./models/intent_classifier",
        "--augment",
    ]

    if epochs != 10:  # Only add if different from default
        cmd.extend(["--epochs", str(epochs)])

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        start_time = time.time()
        result = subprocess.run(cmd, timeout=1800)  # 30 minute timeout

        if result.returncode == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"✅ Intent classifier training completed in {elapsed / 60:.1f} minutes"
            )
            return True
        else:
            logger.error("❌ Intent classifier training failed")
            return False

    except subprocess.TimeoutExpired:
        logger.error("❌ Intent classifier training timed out (30 minutes)")
        return False
    except Exception as e:
        logger.error(f"❌ Intent classifier training failed: {e}")
        return False


def test_trained_models():
    """Test the trained models."""
    logger.info("🧪 Testing trained models...")

    base_dir = Path(__file__).parent
    whisper_model_dir = base_dir / "models" / "whisper_twi"
    intent_model_dir = base_dir / "models" / "intent_classifier"

    success = True

    # Test Whisper model
    if whisper_model_dir.exists():
        try:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            model = WhisperForConditionalGeneration.from_pretrained(whisper_model_dir)
            processor = WhisperProcessor.from_pretrained(whisper_model_dir)

            logger.info("✅ Whisper model loads successfully")

        except Exception as e:
            logger.error(f"❌ Whisper model test failed: {e}")
            success = False
    else:
        logger.warning("⚠️ Whisper model not found")

    # Test Intent classifier
    if intent_model_dir.exists():
        try:
            from transformers import pipeline

            classifier = pipeline("text-classification", model=str(intent_model_dir))

            # Test with sample Twi phrases
            test_phrases = [
                "Kɔ fie",  # go home
                "Kɔ cart mu",  # go to cart
                "Hwehwɛ nneɛma",  # search items
            ]

            logger.info("Testing intent classifier:")
            for phrase in test_phrases:
                result = classifier(phrase)
                intent = result[0]["label"]
                confidence = result[0]["score"]
                logger.info(f"  '{phrase}' -> {intent} ({confidence:.3f})")

            logger.info("✅ Intent classifier working correctly")

        except Exception as e:
            logger.error(f"❌ Intent classifier test failed: {e}")
            success = False
    else:
        logger.warning("⚠️ Intent classifier not found")

    return success


def print_usage_instructions():
    """Print usage instructions."""
    print("\n" + "=" * 70)
    print("                    TRAINING COMPLETED!")
    print("=" * 70)
    print("🎉 Your Twi speech recognition models are ready!")
    print()
    print("📋 Next Steps:")
    print("1. Start the server:")
    print("   python main.py server")
    print()
    print("2. Test with audio:")
    print("   curl -X POST -F 'file=@audio.wav' http://localhost:8000/test-intent")
    print()
    print("3. View API documentation:")
    print("   Open http://localhost:8000/docs in your browser")
    print()
    print("4. Run interactive demo:")
    print("   python main.py demo")
    print()
    print("📊 Models trained:")

    base_dir = Path(__file__).parent
    whisper_dir = base_dir / "models" / "whisper_twi"
    intent_dir = base_dir / "models" / "intent_classifier"

    if whisper_dir.exists():
        print(f"  🎤 Fine-tuned Whisper: {whisper_dir}")
    if intent_dir.exists():
        print(f"  🎯 Intent Classifier: {intent_dir}")

    print("=" * 70)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train Twi Speech Recognition Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_training.py                    # Full training pipeline
  python run_training.py --quick            # Quick training for testing
  python run_training.py --whisper-only     # Only train Whisper
  python run_training.py --intent-only      # Only train Intent Classifier
        """,
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick training with smaller models (for testing)",
    )
    parser.add_argument(
        "--whisper-only", action="store_true", help="Only train Whisper model"
    )
    parser.add_argument(
        "--intent-only", action="store_true", help="Only train Intent Classifier"
    )
    parser.add_argument(
        "--skip-deps", action="store_true", help="Skip dependency installation"
    )
    parser.add_argument("--skip-test", action="store_true", help="Skip model testing")

    args = parser.parse_args()

    print_banner()

    # Check data availability
    if not check_data_availability():
        logger.error("❌ Required data not found. Please ensure:")
        logger.error("  1. Audio files are in data/raw/ directory")
        logger.error("  2. prompts_lean.csv exists in project root")
        sys.exit(1)

    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies():
            logger.error("❌ Failed to install dependencies")
            sys.exit(1)

    start_time = time.time()
    success = True

    # Training pipeline
    if not args.intent_only:
        if not train_whisper_model(quick=args.quick):
            success = False
            if not args.whisper_only:
                logger.error("❌ Whisper training failed, skipping intent classifier")

    if not args.whisper_only and success:
        if not train_intent_classifier(quick=args.quick):
            success = False

    # Test models
    if not args.skip_test and success:
        if not test_trained_models():
            logger.warning("⚠️ Some model tests failed, but training completed")

    # Print results
    total_time = time.time() - start_time

    if success:
        logger.info(
            f"🎉 Training pipeline completed successfully in {total_time / 60:.1f} minutes"
        )
        print_usage_instructions()
    else:
        logger.error(f"❌ Training pipeline failed after {total_time / 60:.1f} minutes")
        logger.error("Check the logs above for specific error messages")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
