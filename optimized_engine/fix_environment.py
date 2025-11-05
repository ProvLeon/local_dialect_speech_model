#!/usr/bin/env python3
"""
Environment Cleanup and Setup Script
===================================

This script fixes dependency conflicts and sets up a clean environment
for Whisper fine-tuning on Twi data.

Usage:
    python fix_environment.py

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


def run_command(cmd, timeout=600):
    """Run a command and return success status."""
    try:
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if result.returncode == 0:
            logger.info("✅ Command succeeded")
            return True
        else:
            logger.error(f"❌ Command failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Command timed out after {timeout} seconds")
        return False
    except Exception as e:
        logger.error(f"❌ Command failed with exception: {e}")
        return False


def uninstall_conflicting_packages():
    """Uninstall packages that cause conflicts."""
    logger.info("🧹 Cleaning up conflicting packages...")

    packages_to_remove = [
        "transformers",
        "torch",
        "torchaudio",
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "huggingface-hub",
        "datasets",
        "accelerate",
        "librosa",
        "soundfile",
        "fsspec",
        "tokenizers",
    ]

    for package in packages_to_remove:
        cmd = [sys.executable, "-m", "pip", "uninstall", package, "-y"]
        logger.info(f"Removing {package}...")
        run_command(cmd, timeout=120)

    logger.info("✅ Cleanup completed")


def install_compatible_versions():
    """Install compatible versions of all packages."""
    logger.info("📦 Installing compatible package versions...")

    # Install packages in specific order to avoid conflicts
    package_groups = [
        # Core numerical packages first
        [
            "numpy>=1.21.0,<1.25.0",
            "scipy>=1.9.0,<1.11.0",
        ],
        # PyTorch ecosystem
        [
            "torch==2.0.0",
            "torchaudio==2.0.0",
        ],
        # Scientific packages
        [
            "scikit-learn>=1.2.0,<1.4.0",
            "matplotlib>=3.6.0,<3.8.0",
            "pandas>=1.5.0",
        ],
        # Audio processing
        [
            "librosa>=0.9.0,<0.11.0",
            "soundfile>=0.10.0,<0.13.0",
        ],
        # HuggingFace ecosystem (careful order)
        [
            "tokenizers>=0.13.0,<0.15.0",
            "huggingface-hub>=0.15.0,<0.18.0",
            "fsspec>=2023.1.0,<=2023.10.0",
            "datasets>=2.12.0,<2.15.0",
            "transformers>=4.30.0,<4.35.0",
            "accelerate>=0.20.0,<0.25.0",
        ],
        # Whisper and utilities
        [
            "openai-whisper>=20231117",
            "jiwer>=2.2.0",
            "evaluate>=0.4.0",
        ],
        # Web framework
        [
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.23.0",
            "python-multipart>=0.0.6",
            "pydantic>=2.0.0",
        ],
        # Utilities
        [
            "tqdm>=4.64.0",
            "pyyaml>=6.0",
        ],
    ]

    for i, group in enumerate(package_groups, 1):
        logger.info(f"Installing group {i}/{len(package_groups)}: {', '.join(group)}")

        cmd = [sys.executable, "-m", "pip", "install"] + group
        if not run_command(cmd, timeout=600):
            logger.error(f"❌ Failed to install group {i}")
            return False

    logger.info("✅ All packages installed successfully")
    return True


def verify_installation():
    """Verify that key packages can be imported."""
    logger.info("🧪 Verifying installation...")

    test_imports = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("whisper", "OpenAI Whisper"),
        ("librosa", "Librosa"),
        ("datasets", "HuggingFace Datasets"),
        ("jiwer", "JIWER"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
    ]

    success_count = 0

    for module, name in test_imports:
        try:
            __import__(module)
            logger.info(f"✅ {name} imported successfully")
            success_count += 1
        except ImportError as e:
            logger.error(f"❌ Failed to import {name}: {e}")

    if success_count == len(test_imports):
        logger.info("🎉 All packages verified successfully!")
        return True
    else:
        logger.error(
            f"❌ {len(test_imports) - success_count} packages failed verification"
        )
        return False


def check_versions():
    """Check and display versions of key packages."""
    logger.info("📋 Checking package versions...")

    version_checks = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("numpy", "NumPy"),
        ("librosa", "Librosa"),
    ]

    for module, name in version_checks:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            logger.info(f"  {name}: {version}")
        except ImportError:
            logger.warning(f"  {name}: not installed")


def main():
    """Main function."""
    logger.info("=" * 60)
    logger.info("ENVIRONMENT CLEANUP AND SETUP")
    logger.info("=" * 60)

    try:
        # Step 1: Clean up
        uninstall_conflicting_packages()

        # Step 2: Install compatible versions
        if not install_compatible_versions():
            logger.error("❌ Installation failed")
            sys.exit(1)

        # Step 3: Verify installation
        if not verify_installation():
            logger.error("❌ Verification failed")
            sys.exit(1)

        # Step 4: Show versions
        check_versions()

        logger.info("=" * 60)
        logger.info("🎉 ENVIRONMENT SETUP COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("Now you can run:")
        logger.info("  python run_training.py --quick")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.error("❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
