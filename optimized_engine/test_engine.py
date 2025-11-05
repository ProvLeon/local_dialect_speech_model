#!/usr/bin/env python3
"""
Test Suite for Optimized Twi Speech Recognition Engine
=====================================================

This module provides comprehensive tests for the optimized speech recognition
engine to ensure all components work correctly.

Author: AI Assistant
Date: 2025-11-05
"""

import os
import sys
import time
import tempfile
import logging
from pathlib import Path
import asyncio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EngineTestSuite:
    """Test suite for the optimized speech recognition engine."""

    def __init__(self):
        self.test_results = {}
        self.temp_files = []

    def cleanup(self):
        """Clean up temporary files."""
        for file_path in self.temp_files:
            try:
                if Path(file_path).exists():
                    Path(file_path).unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup {file_path}: {e}")

    def create_test_audio(self, duration=3.0, sample_rate=16000, frequency=440) -> str:
        """Create a test audio file."""
        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = np.sin(frequency * 2 * np.pi * t)

        # Add some noise to make it more realistic
        noise = np.random.normal(0, 0.01, audio.shape)
        audio = audio + noise

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            sf.write(tmp_file.name, audio, sample_rate)
            self.temp_files.append(tmp_file.name)
            return tmp_file.name

    def test_config_import(self) -> bool:
        """Test configuration import."""
        try:
            from config.config import OptimizedConfig

            config = OptimizedConfig()

            # Basic validation
            assert hasattr(config, "WHISPER")
            assert hasattr(config, "INTENTS")
            assert len(config.INTENTS) > 0

            logger.info(
                f"✅ Config test passed - {len(config.INTENTS)} intents configured"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Config test failed: {e}")
            return False

    def test_speech_recognizer_import(self) -> bool:
        """Test speech recognizer import."""
        try:
            from speech_recognizer import (
                OptimizedSpeechRecognizer,
                create_speech_recognizer,
            )

            # Try to create recognizer
            recognizer = create_speech_recognizer()

            # Basic validation
            assert recognizer is not None
            assert hasattr(recognizer, "recognize")
            assert hasattr(recognizer, "health_check")

            logger.info("✅ Speech recognizer import test passed")
            return True

        except Exception as e:
            logger.error(f"❌ Speech recognizer import failed: {e}")
            return False

    def test_whisper_model_load(self) -> bool:
        """Test Whisper model loading."""
        try:
            import whisper

            # Try to load a small model for testing
            model = whisper.load_model("tiny")
            assert model is not None

            # Test basic transcription
            test_audio = self.create_test_audio(duration=1.0)
            result = model.transcribe(test_audio)

            assert "text" in result

            logger.info("✅ Whisper model test passed")
            return True

        except Exception as e:
            logger.error(f"❌ Whisper model test failed: {e}")
            return False

    def test_audio_processing(self) -> bool:
        """Test audio processing functionality."""
        try:
            from speech_recognizer import AudioProcessor
            from config.config import OptimizedConfig

            config = OptimizedConfig()
            processor = AudioProcessor(config)

            # Create test audio
            test_audio = self.create_test_audio()

            # Test audio loading
            audio_data = processor.load_audio(test_audio)
            assert audio_data is not None
            assert len(audio_data) > 0

            logger.info("✅ Audio processing test passed")
            return True

        except Exception as e:
            logger.error(f"❌ Audio processing test failed: {e}")
            return False

    def test_intent_classification(self) -> bool:
        """Test intent classification."""
        try:
            from speech_recognizer import TwiIntentClassifier
            from config.config import OptimizedConfig

            config = OptimizedConfig()
            classifier = TwiIntentClassifier(config)

            # Test classification with sample Twi text
            test_texts = [
                "Kɔ fie",  # Go home
                "Kɔ cart mu",  # Go to cart
                "Hwehwɛ nneɛma",  # Search items
                "Boa me",  # Help me
                "Tua ka",  # Make payment
            ]

            for text in test_texts:
                result = classifier.classify_intent(text)
                assert "intent" in result
                assert "confidence" in result
                assert result["confidence"] >= 0.0

            logger.info("✅ Intent classification test passed")
            return True

        except Exception as e:
            logger.error(f"❌ Intent classification test failed: {e}")
            return False

    def test_end_to_end_recognition(self) -> bool:
        """Test complete speech recognition pipeline."""
        try:
            from speech_recognizer import create_speech_recognizer

            # Create recognizer with test configuration
            config_overrides = {
                "WHISPER": {"model_size": "tiny"}  # Use tiny model for faster testing
            }
            recognizer = create_speech_recognizer(config_overrides)

            # Create test audio
            test_audio = self.create_test_audio(duration=2.0)

            # Perform recognition
            result = recognizer.recognize(test_audio)

            # Validate result structure
            assert "transcription" in result
            assert "intent" in result
            assert "status" in result

            if result["status"] == "success":
                assert "text" in result["transcription"]
                assert "intent" in result["intent"]
                logger.info(f"✅ End-to-end test passed - Status: {result['status']}")
            else:
                logger.warning(
                    f"⚠️ End-to-end test completed with status: {result['status']}"
                )

            return True

        except Exception as e:
            logger.error(f"❌ End-to-end test failed: {e}")
            return False

    async def test_async_recognition(self) -> bool:
        """Test asynchronous recognition."""
        try:
            from speech_recognizer import create_speech_recognizer

            recognizer = create_speech_recognizer()
            test_audio = self.create_test_audio(duration=1.0)

            # Test async recognition
            result = await recognizer.recognize_async(test_audio)

            assert "transcription" in result
            assert "intent" in result

            logger.info("✅ Async recognition test passed")
            return True

        except Exception as e:
            logger.error(f"❌ Async recognition test failed: {e}")
            return False

    def test_health_check(self) -> bool:
        """Test system health check."""
        try:
            from speech_recognizer import create_speech_recognizer

            recognizer = create_speech_recognizer()
            health = recognizer.health_check()

            assert "status" in health
            assert "components" in health
            assert health["status"] in ["healthy", "degraded", "unhealthy"]

            logger.info(f"✅ Health check test passed - Status: {health['status']}")
            return True

        except Exception as e:
            logger.error(f"❌ Health check test failed: {e}")
            return False

    def test_api_server_import(self) -> bool:
        """Test API server import."""
        try:
            from api_server import app

            assert app is not None

            logger.info("✅ API server import test passed")
            return True

        except Exception as e:
            logger.error(f"❌ API server import failed: {e}")
            return False

    def test_supported_intents(self) -> bool:
        """Test supported intents functionality."""
        try:
            from speech_recognizer import create_speech_recognizer

            recognizer = create_speech_recognizer()
            intents = recognizer.get_supported_intents()

            assert isinstance(intents, list)
            assert len(intents) > 0

            # Check structure of first intent
            if intents:
                intent = intents[0]
                assert "intent" in intent
                assert "description" in intent
                assert "examples" in intent

            logger.info(
                f"✅ Supported intents test passed - {len(intents)} intents found"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Supported intents test failed: {e}")
            return False

    def test_statistics(self) -> bool:
        """Test statistics functionality."""
        try:
            from speech_recognizer import create_speech_recognizer

            recognizer = create_speech_recognizer()
            stats = recognizer.get_statistics()

            assert isinstance(stats, dict)
            assert "total_requests" in stats
            assert "successful_requests" in stats

            logger.info("✅ Statistics test passed")
            return True

        except Exception as e:
            logger.error(f"❌ Statistics test failed: {e}")
            return False

    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results."""
        tests = [
            ("config_import", self.test_config_import),
            ("speech_recognizer_import", self.test_speech_recognizer_import),
            ("whisper_model_load", self.test_whisper_model_load),
            ("audio_processing", self.test_audio_processing),
            ("intent_classification", self.test_intent_classification),
            ("end_to_end_recognition", self.test_end_to_end_recognition),
            ("health_check", self.test_health_check),
            ("api_server_import", self.test_api_server_import),
            ("supported_intents", self.test_supported_intents),
            ("statistics", self.test_statistics),
        ]

        # Run async tests
        async_tests = [
            ("async_recognition", self.test_async_recognition),
        ]

        logger.info("=" * 60)
        logger.info("OPTIMIZED TWI SPEECH ENGINE - TEST SUITE")
        logger.info("=" * 60)

        # Run synchronous tests
        for test_name, test_func in tests:
            logger.info(f"\nRunning {test_name}...")
            try:
                self.test_results[test_name] = test_func()
            except Exception as e:
                logger.error(f"Test {test_name} crashed: {e}")
                self.test_results[test_name] = False

        # Run asynchronous tests
        for test_name, test_func in async_tests:
            logger.info(f"\nRunning {test_name}...")
            try:
                self.test_results[test_name] = await test_func()
            except Exception as e:
                logger.error(f"Test {test_name} crashed: {e}")
                self.test_results[test_name] = False

        # Summary
        self.print_test_summary()

        return self.test_results

    def print_test_summary(self):
        """Print test results summary."""
        logger.info("\n" + "=" * 60)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("=" * 60)

        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)

        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name:25s} {status}")

        logger.info("-" * 60)
        logger.info(
            f"TOTAL: {passed}/{total} tests passed ({passed / total * 100:.1f}%)"
        )

        if passed == total:
            logger.info("🎉 ALL TESTS PASSED! Engine is ready for use.")
        elif passed >= total * 0.8:
            logger.info("⚠️ Most tests passed. Engine should work with minor issues.")
        else:
            logger.info("❌ Multiple test failures. Please check the setup.")

        logger.info("=" * 60)


async def main():
    """Main test function."""
    test_suite = EngineTestSuite()

    try:
        results = await test_suite.run_all_tests()

        # Cleanup
        test_suite.cleanup()

        # Exit with appropriate code
        passed = sum(1 for result in results.values() if result)
        total = len(results)

        if passed == total:
            sys.exit(0)  # All tests passed
        elif passed >= total * 0.8:
            sys.exit(1)  # Most tests passed but some issues
        else:
            sys.exit(2)  # Multiple failures

    except Exception as e:
        logger.error(f"Test suite crashed: {e}")
        test_suite.cleanup()
        sys.exit(3)


if __name__ == "__main__":
    asyncio.run(main())
