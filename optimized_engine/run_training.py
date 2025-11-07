#!/usr/bin/env python3
"""
Simple Run Script for Twi Speech Engine Training
===============================================

This script provides an easy way to train the multi-task Whisper model
using your Twi audio data and prompts.

Usage:
    python run_training.py          # Full training pipeline
    python run_training.py --quick  # Quick training for testing

Author: AI Assistant
Date: 2025-11-07
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
    print("        TWI SPEECH RECOGNITION - MULTI-TASK TRAINING PIPELINE")
    print("=" * 70)
    print("🎯 Goal: Fine-tune a multi-task Whisper model for transcription and intent")
    print("📊 Data: Your recorded Twi audio + prompts CSV")
    print("🚀 Result: A single, efficient speech recognition model")
    print("=" * 70)


def check_data_availability():
    """Check if required data is available."""
    logger.info("🔍 Checking data availability...")
    base_dir = Path(__file__).parent
    data_dir = base_dir.parent / "data" / "raw"
    prompts_file = base_dir.parent / "prompts_lean.csv"
    issues = []
    if not data_dir.exists():
        issues.append(f"Audio data directory not found: {data_dir}")
    else:
        audio_files = list(data_dir.rglob("*.wav")) + list(data_dir.rglob("*.mp3"))
        if not audio_files:
            issues.append(f"No audio files found in {data_dir}")
        else:
            logger.info(f"✅ Found {len(audio_files)} audio files")
    if not prompts_file.exists():
        issues.append(f"Prompts file not found: {prompts_file}")
    else:
        logger.info("✅ Found prompts file.")
    if issues:
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    logger.info("✅ All required data found")
    return True


def install_dependencies():
    """Install required dependencies."""
    logger.info("📦 Installing training dependencies...")
    requirements_file = Path(__file__).parent / "requirements.txt"
    if not requirements_file.exists():
        logger.error(f"❌ {requirements_file} not found!")
        return False
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
        logger.info("✅ Dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"⚠️ Failed to install dependencies: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"❌ Dependency installation failed: {e}")
        return False


def train_multitask_model(quick=False):
    """Train the multi-task Whisper model."""
    logger.info("🎤 Starting multi-task model training...")
    base_dir = Path(__file__).parent
    script_path = base_dir / "train_whisper_twi.py"
    if not script_path.exists():
        logger.error(f"❌ Training script not found: {script_path}")
        return False

    epochs = 3 if quick else 15
    batch_size = 4 if quick else 8
    model_size = "tiny" if quick else "small"

    cmd = [
        sys.executable,
        str(script_path),
        "--model_size",
        model_size,
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    try:
        start_time = time.time()
        subprocess.run(cmd, check=True, timeout=14400)  # 4 hour timeout
        elapsed = time.time() - start_time
        logger.info(f"✅ Multi-task training completed in {elapsed / 60:.1f} minutes")
        return True
    except subprocess.TimeoutExpired:
        logger.error("❌ Training timed out (4 hours)")
        return False
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return False


def test_trained_model():
    """Test the trained multi-task model."""
    logger.info("🧪 Testing trained model...")
    base_dir = Path(__file__).parent
    model_dir = base_dir / "models" / "whisper_twi_multitask"
    if not model_dir.exists():
        logger.warning("⚠️ Multi-task model not found, skipping test.")
        return True

    try:
        from train_whisper_twi import WhisperForMultiTask, TwiWhisperConfig
        from transformers import WhisperProcessor
        import torch

        model = WhisperForMultiTask.from_pretrained(str(model_dir))
        processor = WhisperProcessor.from_pretrained(str(model_dir))
        logger.info("✅ Multi-task model loads successfully")

        # Dummy input for testing
        dummy_input = torch.randn(1, 80, 3000)
        with torch.no_grad():
            outputs = model(input_features=dummy_input)
        
        if outputs and "transcription_logits" in outputs and "classification_logits" in outputs:
             logger.info("✅ Model produces both transcription and classification logits.")
        else:
            raise ValueError("Model output is not as expected.")

        return True
    except Exception as e:
        logger.error(f"❌ Model test failed: {e}")
        return False


def print_usage_instructions():
    """Print usage instructions."""
    print("\n" + "=" * 70)
    print("                    TRAINING COMPLETED!")
    print("=" * 70)
    print("🎉 Your multi-task Twi speech recognition model is ready!")
    print()
    print("📋 Next Steps:")
    print("1. Start the server:")
    print("   python main.py server")
    print("2. Test with an audio file:")
    print("   curl -X POST -F 'file=@path/to/your/audio.wav' http://localhost:8000/recognize")
    print("=" * 70)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train Twi Multi-Task Speech Recognition Model",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick training for testing"
    )
    parser.add_argument(
        "--skip-deps", action="store_true", help="Skip dependency installation"
    )
    parser.add_argument("--skip-test", action="store_true", help="Skip model testing")
    args = parser.parse_args()

    print_banner()

    if not check_data_availability():
        sys.exit(1)

    if not args.skip_deps and not install_dependencies():
        sys.exit(1)

    start_time = time.time()
    success = train_multitask_model(quick=args.quick)

    if success and not args.skip_test:
        success = test_trained_model()

    total_time = time.time() - start_time
    if success:
        logger.info(
            f"🎉 Training pipeline completed successfully in {total_time / 60:.1f} minutes"
        )
        print_usage_instructions()
    else:
        logger.error(f"❌ Training pipeline failed after {total_time / 60:.1f} minutes")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(1)
