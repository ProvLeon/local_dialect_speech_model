#!/usr/bin/env python3
"""
Main Launcher for Optimized Twi Speech Recognition Engine
========================================================

This is the main entry point for the optimized speech recognition system.
It provides a unified interface for starting the server, running tests,
and managing the engine.

Usage:
    python main.py server                    # Start API server
    python main.py test                      # Run test suite
    python main.py setup                     # Run setup
    python main.py demo                      # Run interactive demo
    python main.py --help                    # Show help

Author: AI Assistant
Date: 2025-11-05
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))
sys.path.insert(0, str(current_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


import uvicorn

from src.api_server import app


class OptimizedEngineManager:
    """Manager for the optimized speech recognition engine."""

    def __init__(self, huggingface_repo=None):
        self.base_dir = Path(__file__).parent
        self.running = False
        self.huggingface_repo = huggingface_repo
        self.model_type = None  # Will be detected: 'single' or 'multi'

    def setup_environment(self):
        """Setup environment variables and paths."""
        # Set environment variables
        os.environ.setdefault("ENVIRONMENT", "development")
        os.environ.setdefault("LOG_LEVEL", "INFO")
        os.environ.setdefault("PYTHONPATH", str(self.base_dir))

        # Create necessary directories
        for dir_name in ["logs", "data", "models"]:
            (self.base_dir / dir_name).mkdir(exist_ok=True)

    def download_huggingface_model(self):
        """Download and setup HuggingFace model with retry and timeout handling."""
        if not self.huggingface_repo:
            return True

        logger.info(f"📥 Downloading HuggingFace model: {self.huggingface_repo}")

        try:
            # Create models directory first
            models_dir = self.base_dir / "models" / "huggingface"
            models_dir.mkdir(parents=True, exist_ok=True)

            # Check if model already exists
            existing_model = self._check_existing_model(models_dir)
            if existing_model:
                logger.info(f"📁 Found existing model at: {existing_model}")
                os.environ["HUGGINGFACE_MODEL_PATH"] = existing_model
                self._detect_and_set_model_type(existing_model)
                logger.info(
                    f"🔧 Set HUGGINGFACE_MODEL_PATH={os.environ.get('HUGGINGFACE_MODEL_PATH')}"
                )
                logger.info(
                    f"🔧 Set HUGGINGFACE_MODEL_TYPE={os.environ.get('HUGGINGFACE_MODEL_TYPE')}"
                )
                return True
            import json
            import time

            import requests
            from huggingface_hub import hf_hub_download, snapshot_download
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            # Configure retry strategy for downloads
            retry_strategy = Retry(
                total=3,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS"],
                backoff_factor=1,
            )

            # Create session with timeout and retry configuration
            session = requests.Session()
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)

            # Set longer timeout for large files
            session.timeout = (30, 300)  # (connect_timeout, read_timeout)

            logger.info(
                "🔄 Starting model download with extended timeout (5 minutes)..."
            )

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"📦 Download attempt {attempt + 1}/{max_retries}")

                    # Download model with extended timeout
                    model_path = snapshot_download(
                        repo_id=self.huggingface_repo,
                        local_dir=str(
                            models_dir / self.huggingface_repo.replace("/", "_")
                        ),
                        local_dir_use_symlinks=False,
                        resume_download=True,  # Enable resume for interrupted downloads
                    )

                    logger.info(
                        f"✅ Model downloaded successfully on attempt {attempt + 1}"
                    )
                    break

                except Exception as download_error:
                    logger.warning(
                        f"⚠️ Download attempt {attempt + 1} failed: {download_error}"
                    )

                    if attempt < max_retries - 1:
                        wait_time = (
                            attempt + 1
                        ) * 30  # Progressive backoff: 30s, 60s, 90s
                        logger.info(f"⏳ Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                    else:
                        # Try fallback download method
                        logger.info("🔄 Trying fallback individual file download...")
                        model_path = self._download_model_files_individually(
                            models_dir, session
                        )
                        if model_path:
                            logger.info("✅ Fallback download successful")
                            break
                        else:
                            raise download_error

            # Set up model path and detect type
            self._detect_and_set_model_type(model_path)
            logger.info(f"✅ Model downloaded successfully: {model_path}")
            return True

        except ImportError:
            logger.error(
                "❌ huggingface_hub not installed. Run: pip install huggingface_hub"
            )
            return False
        except Exception as e:
            logger.error(f"❌ Failed to download HuggingFace model: {e}")
            self._show_manual_download_instructions()
            return False

    def _download_model_files_individually(self, models_dir, session):
        """
        Fallback method to download model files individually.
        This can help when snapshot_download times out on large files.
        """
        try:
            from huggingface_hub import hf_hub_download

            model_local_dir = models_dir / self.huggingface_repo.replace("/", "_")
            model_local_dir.mkdir(parents=True, exist_ok=True)

            # Essential files to download (skip large pytorch_model.bin initially)
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

            logger.info(f"📦 Downloading essential files for {self.huggingface_repo}")

            # Download essential files first
            for filename in essential_files:
                try:
                    logger.info(f"⬇️ Downloading {filename}...")
                    hf_hub_download(
                        repo_id=self.huggingface_repo,
                        filename=filename,
                        local_dir=str(model_local_dir),
                        local_dir_use_symlinks=False,
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Could not download {filename}: {e}")

            # Try to download the model weights with extended timeout
            try:
                logger.info(
                    "⬇️ Downloading pytorch_model.bin (this may take several minutes)..."
                )
                hf_hub_download(
                    repo_id=self.huggingface_repo,
                    filename="pytorch_model.bin",
                    local_dir=str(model_local_dir),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                logger.info("✅ Model weights downloaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to download model weights: {e}")
                logger.info(
                    "🔄 You may need to download the model manually or use a smaller model"
                )
                return None

            return str(model_local_dir)

        except Exception as e:
            logger.error(f"❌ Individual file download failed: {e}")
            return None

    def _detect_and_set_model_type(self, model_path):
        """Detect model type and set environment variables."""
        # Detect model type by checking config and files
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)

            # Check for multi-task indicators
            if any(
                key in config for key in ["num_labels", "custom_model", "task_types"]
            ) or any(
                file.exists()
                for file in [
                    Path(model_path) / "intent_labels.json",
                    Path(model_path) / "label_map.json",
                    Path(model_path) / "classification_head.bin",
                ]
            ):
                self.model_type = "multi"
                logger.info(
                    "🔍 Detected multi-task model (transcription + intent classification)"
                )
            else:
                self.model_type = "single"
                logger.info("🔍 Detected single-task model (transcription only)")
        else:
            self.model_type = "single"
            logger.info("🔍 No config found, assuming single-task model")

        # Set environment variables
        os.environ["HUGGINGFACE_MODEL_PATH"] = model_path
        os.environ["HUGGINGFACE_MODEL_TYPE"] = self.model_type

    def _check_existing_model(self, models_dir):
        """Check if the model already exists locally."""
        if not self.huggingface_repo:
            logger.warning("⚠️ No huggingface_repo specified")
            return None

        model_local_dir = models_dir / self.huggingface_repo.replace("/", "_")
        logger.info(f"🔍 Checking for model at: {model_local_dir}")

        if model_local_dir.exists():
            logger.info(f"📁 Directory exists: {model_local_dir}")
            # Check if essential files exist
            essential_files = ["config.json"]
            missing_files = []
            for f in essential_files:
                file_path = model_local_dir / f
                if not file_path.exists():
                    missing_files.append(f)
                else:
                    logger.info(f"✅ Found essential file: {file_path}")

            if not missing_files:
                logger.info(f"✅ Complete model found at: {model_local_dir}")
                return str(model_local_dir)
            else:
                logger.warning(f"⚠️ Missing essential files: {missing_files}")
        else:
            logger.warning(f"⚠️ Model directory does not exist: {model_local_dir}")

        return None

    def _show_manual_download_instructions(self):
        """Show instructions for manual model download."""
        models_dir = self.base_dir / "models" / "huggingface"
        target_dir = models_dir / self.huggingface_repo.replace("/", "_")

        logger.error("=" * 60)
        logger.error("📋 MANUAL DOWNLOAD INSTRUCTIONS")
        logger.error("=" * 60)
        logger.error(
            "Due to network timeouts, you may need to download the model manually."
        )
        logger.error("")
        logger.error("Option 1 - Use git to clone the model:")
        logger.error(f"  mkdir -p {target_dir}")
        logger.error(f"  cd {target_dir}")
        logger.error(f"  git clone https://huggingface.co/{self.huggingface_repo} .")
        logger.error("")
        logger.error("Option 2 - Download using huggingface-cli:")
        logger.error(
            f"  huggingface-cli download {self.huggingface_repo} --local-dir {target_dir}"
        )
        logger.error("")
        logger.error("Option 3 - Use a smaller/faster model:")
        logger.error("  python main.py server --huggingface openai/whisper-small")
        logger.error("")
        logger.error("After manual download, restart the server.")
        logger.error("=" * 60)

    def run_setup(self):
        """Run the setup script."""
        logger.info("🔧 Running setup...")

        try:
            # Download HuggingFace model if specified
            if not self.download_huggingface_model():
                return False

            # Import and run setup
            sys.path.insert(0, str(self.base_dir))
            from setup import main as setup_main

            setup_main()

            logger.info("✅ Setup completed successfully")
            return True

        except ImportError:
            logger.error("❌ Setup script not found")
            return False
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False

    def start_server(self, host="0.0.0.0", port=8000, reload=False):
        """Start the optimized FastAPI server with HuggingFace model support."""
        logger.info(f"🚀 Starting server on {host}:{port}")

        # Log current HuggingFace settings
        if self.huggingface_repo:
            logger.info(f"🤗 HuggingFace repository: {self.huggingface_repo}")
        else:
            logger.info(
                "📝 No HuggingFace repository specified, using default Whisper model"
            )

        # Download and setup HuggingFace model if specified
        if self.huggingface_repo and not self.download_huggingface_model():
            logger.error("❌ Failed to setup HuggingFace model")
            return False

        try:
            # Override port from environment
            port = int(os.environ.get("PORT", port))

            # Set up signal handlers for graceful shutdown
            def signal_handler(signum, frame):
                logger.info("📡 Received shutdown signal")
                self.running = False
                sys.exit(0)

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            self.running = True

            # Start server with uvicorn
            uvicorn.run(
                app,
                host=host,
                port=port,
                log_level="info",
                access_log=True,
                reload=reload,
            )

        except Exception as e:
            logger.error(f"❌ Server failed to start: {e}")
            return False

    async def run_tests(self, verbose=False):
        """Run the test suite."""
        logger.info("🧪 Running test suite...")

        try:
            sys.path.insert(0, str(self.base_dir))
            from test_engine import EngineTestSuite

            test_suite = EngineTestSuite()
            results = await test_suite.run_all_tests()

            # Summary
            passed = sum(1 for result in results.values() if result)
            total = len(results)

            if passed == total:
                logger.info("🎉 All tests passed!")
                return True
            else:
                logger.warning(f"⚠️ {passed}/{total} tests passed")
                return False

        except ImportError:
            logger.error("❌ Test suite not found")
            return False
        except Exception as e:
            logger.error(f"❌ Tests failed: {e}")
            return False

    def run_demo(self):
        """Run interactive demo."""
        logger.info("🎮 Starting interactive demo...")

        try:
            sys.path.insert(0, str(self.base_dir / "src"))
            from speech_recognizer import create_speech_recognizer

            print("\n" + "=" * 60)
            print("    OPTIMIZED TWI SPEECH RECOGNITION DEMO")
            print("=" * 60)

            # Create recognizer
            recognizer = create_speech_recognizer()

            # Health check
            health = recognizer.health_check()
            print(f"System Status: {health['status']}")

            # Show supported intents
            intents = recognizer.get_supported_intents()
            print(f"\nSupported Intents ({len(intents)}):")
            for i, intent in enumerate(intents[:10], 1):
                print(f"  {i:2d}. {intent['intent']:20s} - {intent['description']}")
            if len(intents) > 10:
                print(f"      ... and {len(intents) - 10} more")

            print("\n" + "=" * 60)

            # Interactive mode
            print("\nDemo Mode:")
            print("1. Place your audio files in the current directory")
            print("2. Enter the filename when prompted")
            print("3. Type 'quit' to exit")
            print("-" * 60)

            while True:
                try:
                    # Get audio file
                    filename = input("\nEnter audio filename (or 'quit'): ").strip()

                    if filename.lower() in ["quit", "exit", "q"]:
                        break

                    if not filename:
                        continue

                    # Check if file exists
                    audio_path = Path(filename)
                    if not audio_path.exists():
                        print(f"❌ File not found: {filename}")
                        continue

                    # Process audio
                    print(f"🔄 Processing {filename}...")
                    start_time = time.time()

                    result = recognizer.recognize(str(audio_path))

                    processing_time = time.time() - start_time

                    # Show results
                    if result["status"] == "success":
                        print(f"✅ Recognition completed in {processing_time:.2f}s")
                        print(f"📝 Transcription: '{result['transcription']['text']}'")
                        print(f"🎯 Intent: {result['intent']['intent']}")
                        print(f"📊 Confidence: {result['intent']['confidence']:.3f}")

                        # Show alternatives if available
                        alternatives = result["intent"].get("alternatives", [])
                        if len(alternatives) > 1:
                            print("📋 Alternatives:")
                            for alt in alternatives[1:4]:  # Show top 3 alternatives
                                print(
                                    f"    - {alt.get('label', 'unknown')}: {alt.get('score', 0):.3f}"
                                )
                    else:
                        print(
                            f"❌ Recognition failed: {result.get('error', 'Unknown error')}"
                        )

                except KeyboardInterrupt:
                    print("\n👋 Demo interrupted")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")

            print("\n👋 Demo completed")
            return True

        except ImportError:
            logger.error("❌ Speech recognizer not available")
            return False
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            return False

    def show_status(self):
        """Show system status."""
        print("\n" + "=" * 60)
        print("    OPTIMIZED TWI SPEECH ENGINE STATUS")
        print("=" * 60)

        try:
            sys.path.insert(0, str(self.base_dir / "src"))
            from speech_recognizer import create_speech_recognizer

            recognizer = create_speech_recognizer()

            # Health check
            health = recognizer.health_check()
            print(f"System Status: {health['status']}")

            if "components" in health:
                print("\nComponents:")
                for component, status in health["components"].items():
                    emoji = "✅" if status == "healthy" else "❌"
                    print(f"  {emoji} {component}: {status}")

            if "device_info" in health:
                print(f"\nDevice: {health['device_info'].get('device', 'unknown')}")
                print(
                    f"CUDA Available: {health['device_info'].get('cuda_available', False)}"
                )

            # Statistics
            stats = recognizer.get_statistics()
            print(f"\nStatistics:")
            print(f"  Total Requests: {stats.get('total_requests', 0)}")
            print(f"  Success Rate: {stats.get('success_rate', 0):.1f}%")
            print(f"  Avg Processing Time: {stats.get('avg_processing_time', 0):.3f}s")

            # Supported intents
            intents = recognizer.get_supported_intents()
            print(f"  Supported Intents: {len(intents)}")

            return True

        except Exception as e:
            print(f"❌ Failed to get status: {e}")
            return False

    def show_info(self):
        """Show system information."""
        print("\n" + "=" * 60)
        print("    OPTIMIZED TWI SPEECH ENGINE INFO")
        print("=" * 60)

        try:
            # Basic info
            print(f"Base Directory: {self.base_dir}")
            print(f"Python Version: {sys.version}")

            # Check dependencies
            print("\nDependencies:")
            dependencies = [
                ("torch", "PyTorch"),
                ("whisper", "OpenAI Whisper"),
                ("transformers", "HuggingFace Transformers"),
                ("fastapi", "FastAPI"),
                ("librosa", "Librosa"),
                ("soundfile", "SoundFile"),
            ]

            for module_name, display_name in dependencies:
                try:
                    module = __import__(module_name)
                    version = getattr(module, "__version__", "unknown")
                    print(f"  ✅ {display_name}: {version}")
                except ImportError:
                    print(f"  ❌ {display_name}: Not installed")

            # Configuration
            print(f"\nConfiguration:")
            try:
                sys.path.insert(0, str(self.base_dir))
                sys.path.insert(0, str(self.base_dir))
                from config.config import OptimizedConfig

                config = OptimizedConfig()
                print(f"  Whisper Model: {config.WHISPER['model_size']}")
                print(f"  Supported Intents: {len(config.INTENTS)}")
                print(f"  Device: {config.get_device()}")
            except Exception as e:
                print(f"  ❌ Configuration error: {e}")

            return True

        except Exception as e:
            print(f"❌ Failed to get info: {e}")
            return False


def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Optimized Twi Speech Recognition Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py server                    # Start API server
  python main.py server --port 9000       # Start on custom port
  python main.py server --huggingface username/model-name  # Use HuggingFace model
  python main.py test                      # Run tests
  python main.py demo                      # Interactive demo
  python main.py status                    # Show status
  python main.py setup                     # Run setup
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Server command
    server_parser = subparsers.add_parser("server", help="Start API server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host address")
    server_parser.add_argument("--port", type=int, default=8000, help="Port number")
    server_parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload"
    )
    server_parser.add_argument(
        "--huggingface",
        type=str,
        help="HuggingFace model repository ID (e.g., username/model-name)",
    )

    # Test command
    test_parser = subparsers.add_parser("test", help="Run test suite")
    test_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run interactive demo")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show system information")

    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Run setup script")

    return parser


def main():
    """Main function."""
    parser = create_parser()
    args = parser.parse_args()

    # Extract HuggingFace repo from server args if present
    huggingface_repo = None
    if args.command == "server" and hasattr(args, "huggingface") and args.huggingface:
        huggingface_repo = args.huggingface
        logger.info(f"🤗 Using HuggingFace model: {huggingface_repo}")

    # Create manager with HuggingFace support
    manager = OptimizedEngineManager(huggingface_repo=huggingface_repo)
    manager.setup_environment()

    # Print banner
    print("\n" + "=" * 60)
    print("    OPTIMIZED TWI SPEECH RECOGNITION ENGINE")
    print("=" * 60)

    # Handle commands
    if args.command == "server":
        success = manager.start_server(
            host=args.host, port=args.port, reload=args.reload
        )

    elif args.command == "test":
        # Run tests in asyncio context
        import asyncio

        success = asyncio.run(manager.run_tests(verbose=args.verbose))

    elif args.command == "demo":
        success = manager.run_demo()

    elif args.command == "status":
        success = manager.show_status()

    elif args.command == "info":
        success = manager.show_info()

    elif args.command == "setup":
        success = manager.run_setup()

    else:
        # No command specified, show help
        parser.print_help()

        # Show quick status
        print("\n" + "-" * 60)
        print("QUICK STATUS:")
        try:
            manager.show_status()
        except:
            print("❌ Engine not ready. Run 'python main.py setup' first.")

        success = True

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
