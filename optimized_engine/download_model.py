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
import subprocess
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

    def __init__(self, base_dir: Optional[str] = None):
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
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=2,
            raise_on_status=False,
        )

        session = requests.Session()
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_maxsize=20)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def check_existing_model(self, repo_id: str) -> Optional[str]:
        """Check if model already exists locally with comprehensive file verification."""
        model_dir = self.models_dir / repo_id.replace("/", "_")

        if not model_dir.exists():
            return None

        # Check for essential configuration files
        essential_config_files = ["config.json", "tokenizer_config.json", "vocab.json"]

        # Check for model weight files (at least one must exist)
        model_weight_files = [
            "pytorch_model.bin",
            "model.safetensors",
            "pytorch_model.safetensors",
            "model.bin",
        ]

        # Verify config files exist
        missing_config = [
            f for f in essential_config_files if not (model_dir / f).exists()
        ]
        if missing_config:
            logger.warning(f"⚠️ Missing config files: {missing_config}")
            return None

        # Verify at least one model weight file exists and has reasonable size
        weight_file_found = False
        for weight_file in model_weight_files:
            weight_path = model_dir / weight_file
            if weight_path.exists():
                # Check if file has reasonable size (> 1MB for model weights)
                if weight_path.stat().st_size > 1024 * 1024:
                    weight_file_found = True
                    logger.info(
                        f"✅ Found model weights: {weight_file} ({weight_path.stat().st_size / (1024 * 1024):.1f} MB)"
                    )
                    break
                else:
                    logger.warning(
                        f"⚠️ Weight file {weight_file} too small ({weight_path.stat().st_size} bytes)"
                    )

        if not weight_file_found:
            logger.warning(f"⚠️ No valid model weight files found in {model_dir}")
            return None

        logger.info(f"✅ Complete model verified: {model_dir}")
        return str(model_dir)

    def download_with_huggingface_hub(self, repo_id: str) -> Optional[str]:
        """Download model using huggingface_hub with enhanced settings."""
        try:
            import concurrent.futures
            import os

            from huggingface_hub import snapshot_download

            model_dir = self.models_dir / repo_id.replace("/", "_")
            logger.info(f"📦 Downloading {repo_id} using huggingface_hub...")

            # Set environment variables for huggingface_hub timeout
            original_timeout = os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT")
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(self.read_timeout)

            try:
                for attempt in range(self.max_retries):
                    try:
                        logger.info(f"🔄 Attempt {attempt + 1}/{self.max_retries}")

                        # Use ThreadPoolExecutor with timeout for cross-platform compatibility
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=1
                        ) as executor:
                            future = executor.submit(
                                snapshot_download,
                                repo_id=repo_id,
                                local_dir=str(model_dir),
                                local_dir_use_symlinks=False,
                                resume_download=True,
                                max_workers=1,  # Reduce concurrent downloads
                            )

                            try:
                                model_path = future.result(timeout=self.read_timeout)
                                logger.info(
                                    f"✅ Successfully downloaded to: {model_path}"
                                )
                                return model_path
                            except concurrent.futures.TimeoutError:
                                logger.warning(
                                    f"⚠️ Download timed out after {self.read_timeout} seconds"
                                )
                                # Cancel the future to clean up
                                future.cancel()
                                raise TimeoutError("Download timed out")

                    except (TimeoutError, Exception) as e:
                        logger.warning(f"⚠️ Attempt {attempt + 1} failed: {e}")

                        if attempt < self.max_retries - 1:
                            delay = self.retry_delay_base * (attempt + 1)
                            logger.info(f"⏳ Waiting {delay} seconds before retry...")
                            time.sleep(delay)
                        else:
                            logger.error(
                                "❌ All attempts failed for huggingface_hub download"
                            )
            finally:
                # Restore original timeout
                if original_timeout:
                    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = original_timeout
                else:
                    os.environ.pop("HF_HUB_DOWNLOAD_TIMEOUT", None)

        except ImportError:
            logger.error(
                "❌ huggingface_hub not installed. Run: pip install huggingface_hub"
            )

        return None

    def download_with_git(self, repo_id: str) -> Optional[str]:
        """Download model using git clone with LFS support."""
        try:
            model_dir = self.models_dir / repo_id.replace("/", "_")
            model_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"🔄 Attempting git clone with LFS for {repo_id}...")

            # Check if git-lfs is available
            try:
                lfs_check = subprocess.run(
                    ["git", "lfs", "version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if lfs_check.returncode != 0:
                    logger.warning(
                        "⚠️ Git LFS not available, large files may not download"
                    )
                else:
                    logger.info("✅ Git LFS available")
            except Exception:
                logger.warning("⚠️ Could not check Git LFS availability")

            # Configure git for large files and LFS
            git_env = os.environ.copy()
            git_env.update(
                {
                    "GIT_LFS_SKIP_SMUDGE": "0",  # Ensure LFS files are downloaded
                    "GIT_HTTP_LOW_SPEED_LIMIT": "1000",
                    "GIT_HTTP_LOW_SPEED_TIME": "600",
                }
            )

            # First attempt: git lfs clone (if available)
            try:
                logger.info("🔄 Trying git lfs clone...")
                result = subprocess.run(
                    [
                        "git",
                        "lfs",
                        "clone",
                        "--depth=1",
                        f"https://huggingface.co/{repo_id}",
                        str(model_dir),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=1800,
                    env=git_env,
                )

                if result.returncode == 0:
                    logger.info(f"✅ Git LFS clone successful: {model_dir}")
                    return str(model_dir)
                else:
                    logger.warning(f"⚠️ Git LFS clone failed: {result.stderr}")
            except Exception as e:
                logger.warning(f"⚠️ Git LFS clone error: {e}")

            # Fallback: regular clone + LFS pull
            logger.info("🔄 Trying regular git clone + LFS pull...")

            # Step 1: Regular clone
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
                timeout=900,  # 15 minutes for clone
                env=git_env,
            )

            if result.returncode != 0:
                logger.error(f"❌ Git clone failed: {result.stderr}")
                return None

            # Step 2: Pull LFS files
            try:
                logger.info("🔄 Pulling LFS files...")
                lfs_result = subprocess.run(
                    ["git", "lfs", "pull"],
                    cwd=str(model_dir),
                    capture_output=True,
                    text=True,
                    timeout=1800,  # 30 minutes for LFS files
                    env=git_env,
                )

                if lfs_result.returncode == 0:
                    logger.info("✅ LFS files downloaded successfully")
                else:
                    logger.warning(f"⚠️ LFS pull had issues: {lfs_result.stderr}")
                    # Continue anyway, some files might have downloaded

            except Exception as e:
                logger.warning(f"⚠️ LFS pull error: {e}")

            logger.info(f"✅ Git clone completed: {model_dir}")
            return str(model_dir)

        except FileNotFoundError:
            logger.error(
                "❌ Git not found. Please install git or use huggingface_hub method"
            )
        except subprocess.TimeoutExpired:
            logger.error("❌ Git clone timed out")
        except Exception as e:
            logger.error(f"❌ Git clone error: {e}")

        return None

    def download_individual_files(self, repo_id: str) -> Optional[str]:
        """Download model files individually with smaller timeouts."""
        try:
            from huggingface_hub import list_repo_files

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
            import concurrent.futures
            import os

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

            # Set environment variable for huggingface_hub timeout
            original_timeout = os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT")
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(timeout)

            try:
                for attempt in range(2):  # Fewer retries for individual files
                    try:
                        # Use ThreadPoolExecutor with timeout for cross-platform compatibility
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=1
                        ) as executor:
                            future = executor.submit(
                                hf_hub_download,
                                repo_id=repo_id,
                                filename=filename,
                                local_dir=str(model_dir),
                                local_dir_use_symlinks=False,
                                resume_download=True,
                            )

                            try:
                                future.result(timeout=timeout)
                                logger.info(f"✅ Downloaded {filename}")
                                return True
                            except concurrent.futures.TimeoutError:
                                logger.warning(
                                    f"⚠️ Download of {filename} timed out after {timeout} seconds"
                                )
                                future.cancel()
                                raise TimeoutError(f"Download of {filename} timed out")

                    except (TimeoutError, Exception) as e:
                        logger.warning(
                            f"⚠️ Failed to download {filename} (attempt {attempt + 1}): {e}"
                        )
                        if attempt == 0:
                            time.sleep(10)  # Short delay before retry
            finally:
                # Restore original timeout
                if original_timeout:
                    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = original_timeout
                else:
                    os.environ.pop("HF_HUB_DOWNLOAD_TIMEOUT", None)

        except Exception as e:
            logger.error(f"❌ Error downloading {filename}: {e}")

        return False

    def verify_model(self, model_path: str) -> bool:
        """Verify that the downloaded model is complete with comprehensive checks."""
        model_dir = Path(model_path)

        # Check for essential configuration files
        essential_config_files = ["config.json", "tokenizer_config.json", "vocab.json"]

        # Check for model weight files (at least one must exist)
        model_weight_files = [
            "pytorch_model.bin",
            "model.safetensors",
            "pytorch_model.safetensors",
            "model.bin",
        ]

        # Optional but recommended files
        recommended_files = [
            "generation_config.json",
            "merges.txt",
            "normalizer.json",
            "added_tokens.json",
            "special_tokens_map.json",
            "preprocessor_config.json",
        ]

        # Verify essential config files
        missing_config_files = []
        for filename in essential_config_files:
            file_path = model_dir / filename
            if not file_path.exists():
                missing_config_files.append(filename)
            elif file_path.stat().st_size == 0:
                missing_config_files.append(f"{filename} (empty)")

        if missing_config_files:
            logger.error(
                f"❌ Model verification failed. Missing config files: {missing_config_files}"
            )
            return False

        # Verify model weight files
        weight_file_found = False
        total_weight_size = 0

        for weight_file in model_weight_files:
            weight_path = model_dir / weight_file
            if weight_path.exists():
                file_size = weight_path.stat().st_size
                if file_size > 1024 * 1024:  # Must be > 1MB
                    weight_file_found = True
                    total_weight_size += file_size
                    logger.info(
                        f"✅ Model weights: {weight_file} ({file_size / (1024 * 1024):.1f} MB)"
                    )
                else:
                    logger.warning(
                        f"⚠️ Weight file {weight_file} too small: {file_size} bytes"
                    )

        if not weight_file_found:
            logger.error(
                "❌ Model verification failed. No valid model weight files found!"
            )
            logger.error(
                "   This usually means Git LFS files weren't downloaded properly."
            )
            return False

        logger.info(f"✅ Total model size: {total_weight_size / (1024 * 1024):.1f} MB")

        # Check recommended files
        missing_recommended = []
        for filename in recommended_files:
            if not (model_dir / filename).exists():
                missing_recommended.append(filename)

        if missing_recommended:
            logger.warning(f"⚠️ Missing recommended files: {missing_recommended}")
            logger.warning("   Model may still work but some features might be limited")

        # Try to load and validate config
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
        print("Option 1 - Use git with LFS (recommended):")
        print("  git lfs install")
        print(f"  mkdir -p {target_dir}")
        print(f"  git lfs clone https://huggingface.co/{repo_id} {target_dir}")
        print()
        print("Option 2 - Use huggingface-cli:")
        print("  pip install huggingface_hub[cli]")
        print(f"  huggingface-cli download {repo_id} --local-dir {target_dir}")
        print()
        print("Option 3 - Download from browser:")
        print(f"  1. Visit: https://huggingface.co/{repo_id}")
        print(f"  2. Download ALL files (especially .bin files) to: {target_dir}")
        print()
        print("Option 4 - Use wget for individual files:")
        print(f"  mkdir -p {target_dir} && cd {target_dir}")
        print(f"  wget https://huggingface.co/{repo_id}/resolve/main/pytorch_model.bin")
        print(f"  wget https://huggingface.co/{repo_id}/resolve/main/config.json")
        print("  # Download other required files similarly")
        print()
        print("Option 5 - Use a smaller model:")
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
