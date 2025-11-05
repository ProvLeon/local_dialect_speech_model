#!/usr/bin/env python3
"""
Setup script for Optimized Twi Speech Recognition Engine
=======================================================

This script sets up the optimized speech recognition engine with all
necessary dependencies and configurations.

Author: AI Assistant
Date: 2025-11-05
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_python_version():
    """Check if Python version is compatible."""
    required_version = (3, 8)
    current_version = sys.version_info[:2]

    if current_version < required_version:
        logger.error(
            f"Python {required_version[0]}.{required_version[1]}+ required, but {current_version[0]}.{current_version[1]} found"
        )
        return False

    logger.info(
        f"Python version {current_version[0]}.{current_version[1]} is compatible"
    )
    return True


def install_requirements():
    """Install required packages."""
    requirements_file = Path(__file__).parent / "requirements.txt"

    if not requirements_file.exists():
        logger.error(f"Requirements file not found: {requirements_file}")
        return False

    try:
        logger.info("Installing requirements...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
            capture_output=True,
            text=True,
            timeout=1800,
        )  # 30 minute timeout

        if result.returncode != 0:
            logger.error(f"Failed to install requirements: {result.stderr}")
            return False

        logger.info("Requirements installed successfully")
        return True

    except subprocess.TimeoutExpired:
        logger.error("Installation timed out")
        return False
    except Exception as e:
        logger.error(f"Error installing requirements: {e}")
        return False


def setup_directories():
    """Create necessary directories."""
    base_dir = Path(__file__).parent
    directories = [
        base_dir / "data",
        base_dir / "data" / "audio",
        base_dir / "data" / "models",
        base_dir / "data" / "cache",
        base_dir / "logs",
        base_dir / "models",
        base_dir / "models" / "intent_classifier",
        base_dir / "models" / "whisper_cache",
    ]

    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            return False

    return True


def download_whisper_model():
    """Download and cache Whisper model."""
    try:
        logger.info("Downloading Whisper model (this may take a few minutes)...")

        # Import here to ensure whisper is installed
        import whisper

        # Download the large-v3 model
        model = whisper.load_model("large-v3")
        logger.info("Whisper model downloaded and cached successfully")

        # Clean up
        del model
        return True

    except ImportError:
        logger.error("Whisper not installed. Please install openai-whisper package.")
        return False
    except Exception as e:
        logger.error(f"Failed to download Whisper model: {e}")
        return False


def create_config_files():
    """Create default configuration files."""
    base_dir = Path(__file__).parent

    # Create .env file
    env_file = base_dir / ".env"
    if not env_file.exists():
        env_content = """# Optimized Engine Environment Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
WHISPER_MODEL_SIZE=large-v3
DEVICE=auto
API_HOST=0.0.0.0
API_PORT=8000
ENABLE_GPU=true
CACHE_RESULTS=true
"""
        try:
            with open(env_file, "w") as f:
                f.write(env_content)
            logger.info(f"Created environment file: {env_file}")
        except Exception as e:
            logger.error(f"Failed to create .env file: {e}")
            return False

    return True


def verify_installation():
    """Verify that the installation is working."""
    try:
        logger.info("Verifying installation...")

        # Test imports
        import torch
        import whisper
        import transformers
        import fastapi
        import librosa
        import soundfile

        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"Whisper available: {whisper.__version__}")
        logger.info(f"Transformers version: {transformers.__version__}")
        logger.info(f"FastAPI version: {fastapi.__version__}")

        # Test basic functionality
        try:
            from src.speech_recognizer import create_speech_recognizer

            recognizer = create_speech_recognizer()
            health = recognizer.health_check()

            if health["status"] == "healthy":
                logger.info("✅ Speech recognizer is working correctly")
            else:
                logger.warning(f"⚠️ Speech recognizer health check: {health['status']}")

        except Exception as e:
            logger.warning(f"Could not test speech recognizer: {e}")

        logger.info("Installation verification completed")
        return True

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


def print_usage_instructions():
    """Print usage instructions."""
    base_dir = Path(__file__).parent

    instructions = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    OPTIMIZED TWI SPEECH ENGINE SETUP COMPLETE               ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎉 Installation completed successfully!

📁 Project Structure:
   {base_dir}/
   ├── src/              # Source code
   ├── config/           # Configuration files
   ├── data/             # Data storage
   ├── models/           # Model storage
   ├── logs/             # Log files
   └── tests/            # Test files

🚀 Quick Start:

1. Start the API server:
   cd {base_dir}
   python -m src.api_server

2. Test the health endpoint:
   curl http://localhost:8000/health

3. Upload audio for recognition:
   curl -X POST -F "file=@audio.wav" http://localhost:8000/test-
intent

4. View API documentation:
   Open http://localhost:8000/docs in your browser

📖 Configuration:
   - Edit .env file for environment settings
   - Modify config/config.py for advanced configuration
   - Check logs/ directory for debugging information

🔧 Supported Features:

   ✅ Whisper speech-to-text (25+ languages)
   ✅ Twi intent classification (25 intents)
   ✅ WebM/WAV audio support
   ✅ Real-time processing
   ✅ Batch processing
   ✅ Performance monitoring

💡 Tips:
   - Use GPU for faster processing (CUDA detected: {torch.cuda.is_available() if "torch" in globals() else "Unknown"})
   - Monitor logs/optimized_engine.log for debugging
   - Check /statistics endpoint for performance metrics

Need help? Check the documentation or logs for troubleshooting.
"""

    print(instructions)


def main():
    """Main setup function."""
    logger.info("Starting Optimized Twi Speech Engine setup...")

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Setup directories
    if not setup_directories():
        logger.error("Failed to setup directories")
        sys.exit(1)

    # Install requirements
    if not install_requirements():
        logger.error("Failed to install requirements")
        sys.exit(1)

    # Create configuration files
    if not create_config_files():
        logger.error("Failed to create configuration files")
        sys.exit(1)

    # Download Whisper model
    if not download_whisper_model():
        logger.warning("Failed to download Whisper model (will download on first use)")

    # Verify installation
    if not verify_installation():
        logger.warning(
            "Installation verification had some issues, but setup may still work"
        )

    # Print usage instructions
    print_usage_instructions()

    logger.info("Setup completed successfully! 🎉")


if __name__ == "__main__":
    main()
