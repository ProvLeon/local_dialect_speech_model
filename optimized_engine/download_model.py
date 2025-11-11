#!/usr/bin/env python3
"""
HuggingFace Model Pre-Download Script
===================================

Standalone script to pre-download HuggingFace models with robust
timeout handling, resume capabilities, and progress monitoring.

This script addresses network timeout issues on cPanel servers
and other environments with limited bandwidth or connection stability.

Usage:
    python download_model.py TwiWhisperModel/TwiWhisperModel
    python download_model.py openai/whisper-small
    python download_model.py --help

Author: AI Assistant
Date: 2025-11-11
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("download_model.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


class ModelDownloader:
    """Robust HuggingFace model downloader with timeout and retry handling."""

    def __init__(self, base_dir: str = None):
        """Initialize the model downloader."""
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.models_dir = self.base_dir / "models" / "huggingface"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Download configuration
        self.max_retries = 3
        self.connect_timeout = 30
        self.read_timeout = 600  # 10 minutes
        self.retry_delay_base = 30

    def setup_session(self) -> requests.Session:
        """Setup requests session with retry strategy and timeouts."""
        retry_strategy = Retry(
            total=self.max_retries,
            status_forcelist=[429, 500, 502, 503, 504, 520, 521, 522, 524],
            method_whitelist=["HEAD", "GET", "OPTIONS"],
            backoff_factor=2,
            raise_on_status=False,
        )

        session = requests.Session()
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_maxsize=20)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.timeout = (self.connect_timeout, self.read_timeout)

        return session

    def check_existing_model(self, repo_id: str) -> Optional[str]:
        """Check if model already exists locally."""
        model_dir = self.models_dir / repo_id.replace("/", "_")

        if model_dir.exists():
            # Check for essential files
            essential_files = ["config.json"]
            if all((model_dir / f).exists() for f in essential_files):
                logger.info(f"✅ Model already exists: {model_dir}")
                return str(model_dir)

        return None

    def download_with_huggingface_hub(self, repo_id: str) -> Optional[str]:
        """Download model using huggingface_hub with enhanced settings."""
        try:
            from huggingface_hub import snapshot_download

            model_dir = self.models_dir / repo_id.replace("/", "_")
            logger.info(f"📦 Downloading {repo_id} using huggingface_hub...")

            for attempt in range(self.max_retries):
                try:
                    logger.info(f"🔄 Attempt {attempt + 1}/{self.max_retries}")

                    model_path = snapshot_download(
                        repo_id=repo_id,
                        local_dir=str(model_dir),
                        local_dir_use_symlinks=False,
                        resume_download=True,
                        timeout=self.read_timeout,
                        max_workers=1,  # Reduce concurrent downloads
                    )

                    logger.info(f"✅ Successfully downloaded to: {model_path}")
                    return model_path

                except Exception as e:
                    logger.warning(f"⚠️ Attempt {attempt + 1} failed: {e}")

                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay_base * (attempt + 1)
                        logger.info(f"⏳ Waiting {delay} seconds before retry...")
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"❌ All attempts failed for huggingface_hub download"
                        )

        except ImportError:
            logger.error(
                "❌ huggingface_hub not installed. Run: pip install huggingface_hub"
            )

        return None

    def download_with_git(self, repo_id: str) -> Optional[str]:
        """Download model using git clone as fallback."""
        try:
            import subprocess

            model_dir = self.models_dir / repo_id.replace("/", "_")
            model_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"🔄 Attempting git clone for {repo_id}...")

            # Use git clone with depth=1 for faster download
            result = subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth=1",
                    f"https://huggingface.co/{repo_id}",
                    str(model_dir),
                ],
                capture_output=True,
                text=True,
                timeout=1800,
            )  # 30 minute timeout

            if result.returncode == 0:
                logger.info(f"✅ Git clone successful: {model_dir}")
                return str(model_dir)
            else:
                logger.error(f"❌ Git clone failed: {result.stderr}")

        except FileNotFoundError:
            logger.error(
                "❌ Git not found. Please install git or use huggingface_hub method"
            )
        except subprocess.TimeoutExpired:
            logger.error("❌ Git clone timed out after 30 minutes")
        except Exception as e:
            logger.error(f"❌ Git clone error: {e}")

        return None

    def download_individual_files(self, repo_id: str) -> Optional[str]:
        """Download model files individually with smaller timeouts."""
        try:
            from huggingface_hub import hf_hub_download, list_repo_files

            model_dir = self.models_dir / repo_id.replace("/", "_")
            model_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"📂 Listing files for {repo_id}...")

            # Get list of files in the repository
            files = list_repo_files(repo_id=repo_id)

            # Prioritize essential files first
            essential_files = [
                "config.json",
                "generation_config.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "normalizer.json",
                "added_tokens.json",
                "special_tokens_map.json",
                "preprocessor_config.json",
            ]

            # Download essential files first
            for filename in essential_files:
                if filename in files:
                    self._download_single_file(repo_id, filename, model_dir)

            # Download remaining files
            for filename in files:
                if filename not in essential_files:
                    success = self._download_single_file(repo_id, filename, model_dir)
                    if not success and filename == "pytorch_model.bin":
                        logger.error(
                            "❌ Failed to download model weights - model may not work properly"
                        )

            if (model_dir / "config.json").exists():
                logger.info(f"✅ Individual file download completed: {model_dir}")
                return str(model_dir)

        except Exception as e:
            logger.error(f"❌ Individual file download failed: {e}")

        return None

    def _download_single_file(
        self, repo_id: str, filename: str, model_dir: Path
    ) -> bool:
        """Download a single file with retry logic."""
        try:
            from huggingface_hub import hf_hub_download

            file_path = model_dir / filename
            if file_path.exists():
                logger.info(f"⏭️ Skipping {filename} (already exists)")
                return True

            logger.info(f"⬇️ Downloading {filename}...")

            # Adjust timeout based on likely file size
            if filename.endswith(".bin") or filename.endswith(".safetensors"):
                timeout = self.read_timeout  # Long timeout for model files
            else:
                timeout = 60  # Short timeout for config files

            for attempt in range(2):  # Fewer retries for individual files
                try:
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        local_dir=str(model_dir),
                        local_dir_use_symlinks=False,
                        resume_download=True,
                        timeout=timeout,
                    )
                    logger.info(f"✅ Downloaded {filename}")
                    return True

                except Exception as e:
                    logger.warning(
                        f"⚠️ Failed to download {filename} (attempt {attempt + 1}): {e}"
                    )
                    if attempt == 0:
                        time.sleep(10)  # Short delay before retry

        except Exception as e:
            logger.error(f"❌ Error downloading {filename}: {e}")

        return False

    def verify_model(self, model_path: str) -> bool:
        """Verify that the downloaded model is complete."""
        model_dir = Path(model_path)

        # Check for essential files
        essential_files = ["config.json"]
        missing_files = []

        for filename in essential_files:
            if not (model_dir / filename).exists():
                missing_files.append(filename)

        if missing_files:
            logger.error(
                f"❌ Model verification failed. Missing files: {missing_files}"
            )
            return False

        # Try to load config
        try:
            with open(model_dir / "config.json", "r") as f:
                config = json.load(f)
            logger.info("✅ Model config loaded successfully")

            # Log model info
            model_type = config.get("model_type", "unknown")
            architectures = config.get("architectures", ["unknown"])
            logger.info(f"📋 Model type: {model_type}")
            logger.info(
                f"📋 Architecture: {architectures[0] if architectures else 'unknown'}"
            )

        except Exception as e:
            logger.warning(f"⚠️ Could not validate config: {e}")

        logger.info("✅ Model verification completed")
        return True

    def download_model(self, repo_id: str) -> bool:
        """Download model using multiple fallback methods."""
        logger.info(f"🚀 Starting download for {repo_id}")

        # Check if already exists
        existing_path = self.check_existing_model(repo_id)
        if existing_path:
            return self.verify_model(existing_path)

        # Method 1: Try huggingface_hub
        model_path = self.download_with_huggingface_hub(repo_id)
        if model_path and self.verify_model(model_path):
            return True

        # Method 2: Try git clone
        logger.info("🔄 Trying git clone method...")
        model_path = self.download_with_git(repo_id)
        if model_path and self.verify_model(model_path):
            return True

        # Method 3: Try individual file download
        logger.info("🔄 Trying individual file download...")
        model_path = self.download_individual_files(repo_id)
        if model_path and self.verify_model(model_path):
            return True

        logger.error(f"❌ All download methods failed for {repo_id}")
        return False

    def show_manual_instructions(self, repo_id: str):
        """Show manual download instructions."""
        target_dir = self.models_dir / repo_id.replace("/", "_")

        print("\n" + "=" * 70)
        print("📋 MANUAL DOWNLOAD INSTRUCTIONS")
        print("=" * 70)
        print("All automatic download methods failed. Try these manual options:")
        print()
        print("Option 1 - Use git (if available):")
        print(f"  mkdir -p {target_dir}")
        print(f"  git clone https://huggingface.co/{repo_id} {target_dir}")
        print()
        print("Option 2 - Use huggingface-cli:")
        print(f"  pip install huggingface_hub[cli]")
        print(f"  huggingface-cli download {repo_id} --local-dir {target_dir}")
        print()
        print("Option 3 - Download from browser:")
        print(f"  1. Visit: https://huggingface.co/{repo_id}")
        print(f"  2. Download files manually to: {target_dir}")
        print()
        print("Option 4 - Use a smaller model:")
        print("  Try: openai/whisper-small or openai/whisper-base")
        print("=" * 70)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download HuggingFace models with robust timeout handling"
    )
    parser.add_argument(
        "repo_id",
        help="HuggingFace repository ID (e.g., TwiWhisperModel/TwiWhisperModel)",
    )
    parser.add_argument(
        "--base-dir", help="Base directory for downloads (default: current directory)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retry attempts (default: 3)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Download timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing model, don't download",
    )

    args = parser.parse_args()

    # Create downloader
    downloader = ModelDownloader(base_dir=args.base_dir)
    downloader.max_retries = args.max_retries
    downloader.read_timeout = args.timeout

    logger.info(f"🎯 Target model: {args.repo_id}")
    logger.info(f"📁 Download directory: {downloader.models_dir}")
    logger.info(f"🔄 Max retries: {args.max_retries}")
    logger.info(f"⏱️ Timeout: {args.timeout} seconds")

    if args.verify_only:
        # Just verify existing model
        existing_path = downloader.check_existing_model(args.repo_id)
        if existing_path:
            success = downloader.verify_model(existing_path)
            sys.exit(0 if success else 1)
        else:
            logger.error(f"❌ Model {args.repo_id} not found locally")
            sys.exit(1)

    # Download model
    try:
        success = downloader.download_model(args.repo_id)

        if success:
            logger.info("🎉 Download completed successfully!")
            logger.info(
                f"📁 Model location: {downloader.models_dir / args.repo_id.replace('/', '_')}"
            )
            logger.info("You can now start the server with:")
            logger.info(f"  python main.py server --huggingface {args.repo_id}")
            sys.exit(0)
        else:
            logger.error("❌ Download failed!")
            downloader.show_manual_instructions(args.repo_id)
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("⏹️ Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
