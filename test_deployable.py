#!/usr/bin/env python3
"""
Test script for the deployable Twi speech model package.
This script verifies that the model can be loaded and used for inference.
"""

import os
import sys
import json
import tempfile
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_package_structure():
    """Test that all required files are present in the deployable package."""
    logger.info("Testing package structure...")

    package_path = Path(__file__).parent / "deployable_twi_speech_model"

    required_files = [
        "config/config.json",
        "tokenizer/label_map.json",
        "utils/inference.py",
        "utils/serve.py",
        "requirements.txt",
    ]

    required_dirs = [
        "model",
        "config",
        "tokenizer",
        "utils"
    ]

    missing_files = []
    missing_dirs = []

    for file_path in required_files:
        full_path = package_path / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            logger.info(f"✓ Found: {file_path}")

    for dir_path in required_dirs:
        full_path = package_path / dir_path
        if not full_path.is_dir():
            missing_dirs.append(dir_path)
        else:
            logger.info(f"✓ Found directory: {dir_path}")

    if missing_files or missing_dirs:
        logger.error(f"Missing files: {missing_files}")
        logger.error(f"Missing directories: {missing_dirs}")
        return False

    logger.info("✓ Package structure is complete")
    return True

def test_model_loading():
    """Test that the model can be loaded successfully."""
    logger.info("Testing model loading...")

    try:
        # Add the deployable package to path
        package_path = Path(__file__).parent / "deployable_twi_speech_model"
        utils_path = package_path / "utils"
        sys.path.insert(0, str(utils_path))

        # Import the inference module
        from inference import ModelInference

        # Load the model
        model = ModelInference(str(package_path))

        logger.info("✓ Model loaded successfully")
        return model

    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        return None

def test_model_info(model):
    """Test model info retrieval."""
    logger.info("Testing model info...")

    try:
        info = model.get_model_info()
        logger.info("✓ Model info retrieved successfully")
        logger.info(f"Model: {info.get('model_name', 'Unknown')}")
        logger.info(f"Version: {info.get('version', 'Unknown')}")
        logger.info(f"Classes: {info.get('num_classes', 'Unknown')}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to get model info: {e}")
        return False

def test_health_check(model):
    """Test model health check."""
    logger.info("Testing health check...")

    try:
        health = model.health_check()
        logger.info("✓ Health check completed")
        logger.info(f"Status: {health.get('status', 'Unknown')}")
        return health.get('status') == 'healthy'
    except Exception as e:
        logger.error(f"✗ Health check failed: {e}")
        return False

def create_dummy_audio():
    """Create a dummy audio file for testing."""
    try:
        import numpy as np
        import soundfile as sf

        # Create 3 seconds of dummy audio at 16kHz
        duration = 3.0
        sample_rate = 16000
        samples = int(duration * sample_rate)

        # Generate some sine waves (like speech-ish audio)
        t = np.linspace(0, duration, samples)
        audio = 0.1 * (np.sin(2 * np.pi * 200 * t) +
                      0.5 * np.sin(2 * np.pi * 400 * t) +
                      0.3 * np.sin(2 * np.pi * 800 * t))

        # Add some noise
        audio += 0.01 * np.random.randn(samples)

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_file.name, audio, sample_rate)

        return temp_file.name

    except ImportError:
        logger.warning("soundfile not available, skipping audio generation test")
        return None
    except Exception as e:
        logger.error(f"Failed to create dummy audio: {e}")
        return None

def test_inference(model):
    """Test model inference with dummy audio."""
    logger.info("Testing inference...")

    audio_file = create_dummy_audio()
    if audio_file is None:
        logger.warning("Skipping inference test (no audio file)")
        return True

    try:
        # Test single prediction
        intent, confidence = model.predict(audio_file)
        logger.info(f"✓ Single prediction: {intent} (confidence: {confidence:.3f})")

        # Test top-k prediction
        intent, confidence, top_predictions = model.predict_topk(audio_file, top_k=3)
        logger.info(f"✓ Top-k prediction completed")
        logger.info(f"Top prediction: {intent} (confidence: {confidence:.3f})")

        # Clean up
        os.unlink(audio_file)

        return True

    except Exception as e:
        logger.error(f"✗ Inference test failed: {e}")
        if audio_file and os.path.exists(audio_file):
            os.unlink(audio_file)
        return False

def test_fastapi_import():
    """Test that FastAPI components can be imported."""
    logger.info("Testing FastAPI imports...")

    try:
        package_path = Path(__file__).parent / "deployable_twi_speech_model"
        utils_path = package_path / "utils"
        sys.path.insert(0, str(utils_path))

        # Change to utils directory for relative imports
        original_cwd = os.getcwd()
        os.chdir(utils_path)

        try:
            from serve import app
            logger.info("✓ FastAPI app imported successfully")
            return True
        finally:
            os.chdir(original_cwd)

    except Exception as e:
        logger.error(f"✗ FastAPI import failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("=== Testing Deployable Twi Speech Model ===")

    tests = [
        ("Package Structure", test_package_structure),
        ("Model Loading", test_model_loading),
        ("FastAPI Import", test_fastapi_import)
    ]

    results = {}
    model = None

    # Run basic tests
    for test_name, test_func in tests:
        try:
            if test_name == "Model Loading":
                result = test_func()
                if result:
                    model = result
                    results[test_name] = True
                else:
                    results[test_name] = False
            else:
                results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results[test_name] = False

    # Run model-dependent tests if model loaded successfully
    if model:
        model_tests = [
            ("Model Info", lambda: test_model_info(model)),
            ("Health Check", lambda: test_health_check(model)),
            ("Inference", lambda: test_inference(model))
        ]

        for test_name, test_func in model_tests:
            try:
                results[test_name] = test_func()
            except Exception as e:
                logger.error(f"Test '{test_name}' crashed: {e}")
                results[test_name] = False

    # Summary
    logger.info("\n=== Test Results ===")
    passed = 0
    total = 0

    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
        if passed_test:
            passed += 1
        total += 1

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! The deployable package is ready.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please fix the issues before deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
