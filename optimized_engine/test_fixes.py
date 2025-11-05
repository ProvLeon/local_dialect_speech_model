#!/usr/bin/env python3
"""
Test script to verify fixes for language and data collator issues.
"""

import sys
import os
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))


def test_data_collator():
    """Test the fixed data collator."""
    print("Testing data collator fix...")

    try:
        from train_whisper_twi import TwiWhisperDataCollator
        from transformers import WhisperProcessor

        # Load processor
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        collator = TwiWhisperDataCollator(processor)

        # Create dummy features (simulating what the dataset would return)
        dummy_features = [
            {
                "input_features": np.random.randn(80, 3000).astype(np.float32),
                "labels": [50258, 50259, 50260, 50261, 50257],  # Dummy token IDs
            },
            {
                "input_features": np.random.randn(80, 3000).astype(np.float32),
                "labels": [50258, 50259, 50262, 50257],  # Different length
            },
        ]

        # Test collator
        batch = collator(dummy_features)

        print(f"✅ Data collator test passed!")
        print(f"   Batch keys: {list(batch.keys())}")
        print(f"   Input features shape: {batch['input_features'].shape}")
        print(f"   Labels shape: {batch['labels'].shape}")

        return True

    except Exception as e:
        print(f"❌ Data collator test failed: {e}")
        return False


def test_speech_recognizer():
    """Test the speech recognizer with language fixes."""
    print("\nTesting speech recognizer language handling...")

    try:
        from src.speech_recognizer import OptimizedSpeechRecognizer
        from config.config import OptimizedConfig

        # Create test audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            # Generate 2 seconds of dummy audio
            sample_rate = 16000
            duration = 2
            audio_data = np.random.randn(sample_rate * duration) * 0.1
            sf.write(temp_audio.name, audio_data, sample_rate)

            # Initialize recognizer
            config = OptimizedConfig()
            recognizer = OptimizedSpeechRecognizer(config)

            # Test with auto-detect language (None)
            result = recognizer.transcribe.transcribe(temp_audio.name, language=None)

            print(f"✅ Speech recognizer test passed!")
            print(f"   Detected language: {result.get('language', 'unknown')}")
            print(f"   Transcription: '{result.get('text', '')}'")
            print(f"   Confidence: {result.get('confidence', 0):.2f}")

            # Clean up
            os.unlink(temp_audio.name)

            return True

    except Exception as e:
        print(f"❌ Speech recognizer test failed: {e}")
        # Clean up on error
        try:
            os.unlink(temp_audio.name)
        except:
            pass
        return False


def test_config():
    """Test configuration language settings."""
    print("\nTesting configuration...")

    try:
        from config.config import OptimizedConfig

        config = OptimizedConfig()

        # Check language setting
        whisper_language = config.WHISPER["language"]
        print(f"✅ Configuration test passed!")
        print(f"   Whisper language setting: {whisper_language}")
        print(f"   Device setting: {config.get_device()}")

        # Validate configuration
        is_valid = config.validate_config()
        print(f"   Configuration valid: {is_valid}")

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_api_server_imports():
    """Test API server imports and basic initialization."""
    print("\nTesting API server...")

    try:
        from src.api_server import app, RecognitionRequest

        # Test creating a request with auto-detect language
        request = RecognitionRequest()
        print(f"✅ API server test passed!")
        print(f"   Default language: {request.language}")
        print(f"   App created successfully")

        return True

    except Exception as e:
        print(f"❌ API server test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING FIXES FOR LANGUAGE AND DATA COLLATOR ISSUES")
    print("=" * 60)

    tests = [
        ("Configuration", test_config),
        ("Data Collator", test_data_collator),
        ("API Server", test_api_server_imports),
        ("Speech Recognizer", test_speech_recognizer),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<20}: {status}")
        if result:
            passed += 1

    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! The fixes should work correctly.")
        print("\nNext steps:")
        print("1. Try running: python run_training.py --quick")
        print("2. If training works, start the server: python main.py server")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
