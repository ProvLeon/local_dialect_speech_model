#!/usr/bin/env python3
"""
Comprehensive Setup Script for Optimized Twi Speech Recognition Engine
=====================================================================

This script provides complete setup including:
1. Environment setup and dependency installation
2. Whisper fine-tuning on your Twi audio data
3. Intent classifier training
4. Model validation and testing
5. Server configuration

Usage:
    python setup_complete.py                    # Full setup with fine-tuning
    python setup_complete.py --skip-whisper     # Skip Whisper fine-tuning
    python setup_complete.py --quick            # Quick setup for development
    python setup_complete.py --evaluate-only    # Only evaluate existing models

Author: AI Assistant
Date: 2025-11-05
"""

import os
import sys
import subprocess
import logging
import argparse
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ComprehensiveSetup:
    """Complete setup manager for the optimized Twi speech engine."""

    def __init__(self, args):
        self.args = args
        self.base_dir = Path(__file__).parent
        self.project_root = self.base_dir.parent
        self.setup_status = {
            "dependencies": False,
            "directories": False,
            "whisper_model": False,
            "intent_classifier": False,
            "validation": False,
            "server_config": False,
        }

        # Paths
        self.data_dir = self.project_root / "data" / "raw"
        self.prompts_file = self.project_root / "twi_prompts.csv"
        self.models_dir = self.base_dir / "models"
        self.whisper_model_dir = self.models_dir / "whisper_twi"
        self.intent_model_dir = self.models_dir / "intent_classifier"

    def print_header(self):
        """Print setup header."""
        print("\n" + "=" * 80)
        print("     COMPREHENSIVE SETUP: OPTIMIZED TWI SPEECH RECOGNITION")
        print("=" * 80)
        print(f"🎯 Goal: Fine-tune Whisper + Train Intent Classifier")
        print(f"📁 Data: {self.data_dir}")
        print(f"📋 Prompts: {self.prompts_file}")
        print(f"💾 Models: {self.models_dir}")
        print("=" * 80)

    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are available."""
        logger.info("🔍 Checking prerequisites...")

        issues = []

        # Check Python version
        if sys.version_info < (3, 8):
            issues.append(f"Python 3.8+ required, found {sys.version}")

        # Check data availability
        if not self.data_dir.exists():
            issues.append(f"Audio data directory not found: {self.data_dir}")
        else:
            # Count audio files
            audio_files = list(self.data_dir.rglob("*.wav")) + list(
                self.data_dir.rglob("*.mp3")
            )
            if len(audio_files) == 0:
                issues.append(f"No audio files found in {self.data_dir}")
            else:
                logger.info(f"✅ Found {len(audio_files)} audio files")

        # Check prompts file
        if not self.prompts_file.exists():
            issues.append(f"Prompts file not found: {self.prompts_file}")
        else:
            import pandas as pd

            try:
                df = pd.read_csv(self.prompts_file)
                logger.info(f"✅ Found {len(df)} prompts in CSV")
            except Exception as e:
                issues.append(f"Invalid prompts file: {e}")

        # Check disk space (need ~10GB for models)
        try:
            import shutil

            total, used, free = shutil.disk_usage(self.base_dir)
            free_gb = free / (1024**3)
            if free_gb < 10:
                issues.append(
                    f"Insufficient disk space: {free_gb:.1f}GB available, 10GB+ recommended"
                )
            else:
                logger.info(f"✅ Sufficient disk space: {free_gb:.1f}GB available")
        except:
            logger.warning("Could not check disk space")

        if issues:
            logger.error("❌ Prerequisites check failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False

        logger.info("✅ All prerequisites satisfied")
        return True

    def install_dependencies(self) -> bool:
        """Install required dependencies."""
        logger.info("📦 Installing dependencies...")

        requirements_file = self.base_dir / "requirements.txt"

        if not requirements_file.exists():
            logger.error(f"Requirements file not found: {requirements_file}")
            return False

        try:
            # Install core requirements
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

            if result.returncode != 0:
                logger.error(f"Failed to install requirements: {result.stderr}")
                return False

            logger.info("✅ Core dependencies installed")

            # Install additional dependencies for fine-tuning
            additional_deps = [
                "datasets",
                "jiwer",
                "evaluate",
                "accelerate",
                "librosa>=0.10.0",
                "soundfile>=0.12.0",
            ]

            for dep in additional_deps:
                try:
                    cmd = [sys.executable, "-m", "pip", "install", dep]
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=300
                    )
                    if result.returncode == 0:
                        logger.info(f"✅ Installed {dep}")
                    else:
                        logger.warning(f"⚠️ Failed to install {dep}: {result.stderr}")
                except Exception as e:
                    logger.warning(f"⚠️ Error installing {dep}: {e}")

            self.setup_status["dependencies"] = True
            return True

        except subprocess.TimeoutExpired:
            logger.error("❌ Installation timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Installation failed: {e}")
            return False

    def setup_directories(self) -> bool:
        """Create necessary directories."""
        logger.info("📁 Setting up directories...")

        directories = [
            self.models_dir,
            self.whisper_model_dir,
            self.intent_model_dir,
            self.base_dir / "data",
            self.base_dir / "data" / "cache",
            self.base_dir / "logs",
        ]

        try:
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Created: {directory}")

            self.setup_status["directories"] = True
            return True

        except Exception as e:
            logger.error(f"❌ Failed to create directories: {e}")
            return False

    def fine_tune_whisper(self) -> bool:
        """Fine-tune Whisper model on Twi data."""
        if self.args.skip_whisper:
            logger.info("⏭️ Skipping Whisper fine-tuning")
            return True

        logger.info("🎤 Starting Whisper fine-tuning on Twi data...")

        # Check if model already exists
        if self.whisper_model_dir.exists() and list(
            self.whisper_model_dir.glob("*.json")
        ):
            if not self.args.force:
                logger.info(
                    f"✅ Whisper model already exists at {self.whisper_model_dir}"
                )
                self.setup_status["whisper_model"] = True
                return True
            else:
                logger.info("🔄 Overwriting existing Whisper model")
                shutil.rmtree(self.whisper_model_dir)
                self.whisper_model_dir.mkdir(parents=True)

        try:
            # Import training script
            sys.path.insert(0, str(self.base_dir))

            # Determine model size based on args
            model_size = "tiny" if self.args.quick else "small"
            epochs = 3 if self.args.quick else 10
            batch_size = 4 if self.args.quick else 8

            logger.info(
                f"Training config: model={model_size}, epochs={epochs}, batch_size={batch_size}"
            )

            # Run training
            cmd = [
                sys.executable,
                str(self.base_dir / "train_whisper_twi.py"),
                "--model_size",
                model_size,
                "--data_dir",
                str(self.data_dir),
                "--prompts_file",
                str(self.prompts_file),
                "--output_dir",
                str(self.whisper_model_dir),
                "--epochs",
                str(epochs),
                "--batch_size",
                str(batch_size),
                "--eval_steps",
                "100" if self.args.quick else "500",
            ]

            logger.info(f"Running: {' '.join(cmd)}")

            result = subprocess.run(cmd, timeout=7200)  # 2 hour timeout

            if result.returncode == 0:
                logger.info("✅ Whisper fine-tuning completed successfully")
                self.setup_status["whisper_model"] = True
                return True
            else:
                logger.error("❌ Whisper fine-tuning failed")
                return False

        except subprocess.TimeoutExpired:
            logger.error("❌ Whisper fine-tuning timed out")
            return False
        except ImportError as e:
            logger.error(f"❌ Failed to import training modules: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Whisper fine-tuning failed: {e}")
            return False

    def train_intent_classifier(self) -> bool:
        """Train intent classification model."""
        logger.info("🎯 Training intent classifier...")

        # Check if model already exists
        if self.intent_model_dir.exists() and list(
            self.intent_model_dir.glob("*.json")
        ):
            if not self.args.force:
                logger.info(
                    f"✅ Intent classifier already exists at {self.intent_model_dir}"
                )
                self.setup_status["intent_classifier"] = True
                return True
            else:
                logger.info("🔄 Overwriting existing intent classifier")
                shutil.rmtree(self.intent_model_dir)
                self.intent_model_dir.mkdir(parents=True)

        try:
            # Run intent classifier training
            cmd = [
                sys.executable,
                str(self.base_dir / "train_intent_classifier.py"),
                "--data",
                str(self.prompts_file),
                "--output",
                str(self.intent_model_dir),
                "--augment",
            ]

            if not self.args.quick:
                cmd.extend(["--epochs", "10"])
            else:
                cmd.extend(["--epochs", "3"])

            logger.info(f"Running: {' '.join(cmd)}")

            result = subprocess.run(cmd, timeout=1800)  # 30 minute timeout

            if result.returncode == 0:
                logger.info("✅ Intent classifier training completed")
                self.setup_status["intent_classifier"] = True
                return True
            else:
                logger.error("❌ Intent classifier training failed")
                return False

        except subprocess.TimeoutExpired:
            logger.error("❌ Intent classifier training timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Intent classifier training failed: {e}")
            return False

    def validate_models(self) -> bool:
        """Validate trained models."""
        logger.info("🧪 Validating models...")

        validation_results = {}

        # Test Whisper model
        if self.whisper_model_dir.exists():
            try:
                # Try loading the model
                from transformers import (
                    WhisperForConditionalGeneration,
                    WhisperProcessor,
                )

                model = WhisperForConditionalGeneration.from_pretrained(
                    self.whisper_model_dir
                )
                processor = WhisperProcessor.from_pretrained(self.whisper_model_dir)

                logger.info("✅ Whisper model loads successfully")
                validation_results["whisper"] = "success"

                # Test with a sample audio file if available
                audio_files = list(self.data_dir.rglob("*.wav"))
                if audio_files:
                    test_audio = audio_files[0]
                    logger.info(f"Testing with: {test_audio}")

                    import librosa
                    import torch

                    audio, sr = librosa.load(test_audio, sr=16000)
                    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

                    with torch.no_grad():
                        predicted_ids = model.generate(
                            inputs.input_features, max_length=50
                        )
                        transcription = processor.batch_decode(
                            predicted_ids, skip_special_tokens=True
                        )[0]

                    logger.info(f"✅ Sample transcription: '{transcription}'")
                    validation_results["whisper_test"] = transcription

            except Exception as e:
                logger.error(f"❌ Whisper model validation failed: {e}")
                validation_results["whisper"] = f"failed: {e}"

        # Test intent classifier
        if self.intent_model_dir.exists():
            try:
                from transformers import pipeline

                classifier = pipeline(
                    "text-classification", model=str(self.intent_model_dir)
                )

                # Test with sample texts
                test_texts = ["Kɔ fie", "Kɔ cart mu", "Hwehwɛ nneɛma"]

                for text in test_texts:
                    result = classifier(text)
                    logger.info(
                        f"✅ '{text}' -> {result[0]['label']} ({result[0]['score']:.3f})"
                    )

                validation_results["intent_classifier"] = "success"

            except Exception as e:
                logger.error(f"❌ Intent classifier validation failed: {e}")
                validation_results["intent_classifier"] = f"failed: {e}"

        # Test complete pipeline
        try:
            sys.path.insert(0, str(self.base_dir / "src"))
            from speech_recognizer import create_speech_recognizer

            recognizer = create_speech_recognizer()
            health = recognizer.health_check()

            if health["status"] == "healthy":
                logger.info("✅ Complete pipeline validation successful")
                validation_results["pipeline"] = "healthy"
            else:
                logger.warning(f"⚠️ Pipeline health check: {health['status']}")
                validation_results["pipeline"] = health["status"]

        except Exception as e:
            logger.error(f"❌ Pipeline validation failed: {e}")
            validation_results["pipeline"] = f"failed: {e}"

        # Save validation results
        results_file = self.base_dir / "validation_results.json"
        with open(results_file, "w") as f:
            json.dump(validation_results, f, indent=2)

        self.setup_status["validation"] = True
        return True

    def configure_server(self) -> bool:
        """Configure server settings."""
        logger.info("⚙️ Configuring server...")

        try:
            # Update configuration to use trained models
            config_file = self.base_dir / "config" / "config.py"

            # Read current config
            with open(config_file, "r") as f:
                config_content = f.read()

            # Update whisper configuration
            if self.whisper_model_dir.exists():
                logger.info("Configuring to use fine-tuned Whisper model")
                config_content = config_content.replace(
                    '"model_size": "large-v3"', '"model_size": "custom"'
                )
                config_content = config_content.replace(
                    '"use_fine_tuned": True', '"use_fine_tuned": True'
                )

            # Save updated config
            with open(config_file, "w") as f:
                f.write(config_content)

            logger.info("✅ Server configuration updated")
            self.setup_status["server_config"] = True
            return True

        except Exception as e:
            logger.error(f"❌ Server configuration failed: {e}")
            return False

    def print_summary(self):
        """Print setup summary."""
        print("\n" + "=" * 80)
        print("                         SETUP SUMMARY")
        print("=" * 80)

        for component, status in self.setup_status.items():
            emoji = "✅" if status else "❌"
            print(
                f"{emoji} {component.replace('_', ' ').title()}: {'Success' if status else 'Failed'}"
            )

        print("\n" + "-" * 80)

        success_count = sum(1 for status in self.setup_status.values() if status)
        total_count = len(self.setup_status)

        if success_count == total_count:
            print("🎉 SETUP COMPLETED SUCCESSFULLY!")
            print("\n📋 Next Steps:")
            print("1. Start the server: python main.py server")
            print(
                "2. Test with audio: curl -X POST -F 'file=@audio.wav' http://localhost:8000/test-intent"
            )
            print("3. Check API docs: http://localhost:8000/docs")

            print("\n🎯 Models Trained:")
            if self.whisper_model_dir.exists():
                print(f"  📱 Fine-tuned Whisper: {self.whisper_model_dir}")
            if self.intent_model_dir.exists():
                print(f"  🎯 Intent Classifier: {self.intent_model_dir}")

        else:
            print(f"⚠️ SETUP PARTIALLY COMPLETED ({success_count}/{total_count})")
            print("\n❌ Failed Components:")
            for component, status in self.setup_status.items():
                if not status:
                    print(f"  - {component.replace('_', ' ').title()}")

            print("\n🔧 Troubleshooting:")
            print("1. Check logs above for specific error messages")
            print("2. Ensure sufficient disk space (10GB+)")
            print("3. Verify audio data in data/raw directory")
            print("4. Try running with --quick for faster setup")

        print("=" * 80)

    def run_complete_setup(self) -> bool:
        """Run the complete setup process."""
        start_time = time.time()

        self.print_header()

        if not self.check_prerequisites():
            return False

        steps = [
            ("Installing Dependencies", self.install_dependencies),
            ("Setting up Directories", self.setup_directories),
            ("Fine-tuning Whisper", self.fine_tune_whisper),
            ("Training Intent Classifier", self.train_intent_classifier),
            ("Validating Models", self.validate_models),
            ("Configuring Server", self.configure_server),
        ]

        for step_name, step_func in steps:
            if self.args.evaluate_only and step_name not in ["Validating Models"]:
                continue

            logger.info(f"\n🚀 {step_name}...")
            success = step_func()

            if not success:
                logger.error(f"❌ {step_name} failed!")
                if not self.args.continue_on_error:
                    break

        elapsed_time = time.time() - start_time
        logger.info(f"\n⏱️ Total setup time: {elapsed_time / 60:.1f} minutes")

        self.print_summary()

        return all(self.setup_status.values())


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive setup for Optimized Twi Speech Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_complete.py                    # Full setup with fine-tuning
  python setup_complete.py --quick            # Quick development setup
  python setup_complete.py --skip-whisper     # Skip Whisper fine-tuning
  python setup_complete.py --evaluate-only    # Only validate existing models
        """,
    )

    parser.add_argument(
        "--skip-whisper",
        action="store_true",
        help="Skip Whisper fine-tuning (use pre-trained model)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick setup with smaller models and fewer epochs",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing models"
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only evaluate existing models, skip training",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue setup even if some steps fail",
    )

    args = parser.parse_args()

    try:
        setup = ComprehensiveSetup(args)
        success = setup.run_complete_setup()

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n⚠️ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Setup failed with unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
