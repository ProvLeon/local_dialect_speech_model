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

import os
import sys
import argparse
import asyncio
import logging
from pathlib import Path
import signal
import time

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

    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.running = False

    def setup_environment(self):
        """Setup environment variables and paths."""
        # Set environment variables
        os.environ.setdefault("ENVIRONMENT", "development")
        os.environ.setdefault("LOG_LEVEL", "INFO")
        os.environ.setdefault("PYTHONPATH", str(self.base_dir))

        # Create necessary directories
        for dir_name in ["logs", "data", "models"]:
            (self.base_dir / dir_name).mkdir(exist_ok=True)

    def run_setup(self):
        """Run the setup script."""
        logger.info("🔧 Running setup...")

        try:
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
        """Start the FastAPI server."""
        logger.info(f"🚀 Starting server on {host}:{port}")

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

    # Create manager
    manager = OptimizedEngineManager()
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
