#!/usr/bin/env python3
"""
Debug script to identify timeout issues in the audio processing pipeline
"""

import os
import sys
import time
import logging
import tempfile
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_audio_conversion(audio_file_path: str):
    """Test audio conversion step by step"""
    print(f"\n{'='*60}")
    print(f"Testing audio conversion for: {audio_file_path}")
    print(f"{'='*60}")

    if not os.path.exists(audio_file_path):
        print(f"❌ File does not exist: {audio_file_path}")
        return False

    file_size = os.path.getsize(audio_file_path)
    print(f"📁 File size: {file_size} bytes")

    try:
        from src.utils.audio_converter import convert_audio_to_wav, validate_audio_file

        # Test conversion
        print("\n🔄 Testing audio conversion...")
        start_time = time.time()

        converted_path = convert_audio_to_wav(audio_file_path, timeout_seconds=30)
        conversion_time = time.time() - start_time

        if converted_path:
            print(f"✅ Conversion successful in {conversion_time:.2f}s")
            print(f"📁 Converted file: {converted_path}")
            print(f"📁 Converted size: {os.path.getsize(converted_path)} bytes")

            # Test validation
            print("\n🔍 Testing audio validation...")
            start_time = time.time()

            is_valid = validate_audio_file(converted_path, timeout_seconds=10)
            validation_time = time.time() - start_time

            if is_valid:
                print(f"✅ Validation successful in {validation_time:.2f}s")
            else:
                print(f"❌ Validation failed in {validation_time:.2f}s")

            # Clean up if temporary file
            if converted_path != audio_file_path:
                os.unlink(converted_path)
                print(f"🧹 Cleaned up temporary file")

            return is_valid
        else:
            print(f"❌ Conversion failed in {conversion_time:.2f}s")
            return False

    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        traceback.print_exc()
        return False

def test_audio_preprocessing(audio_file_path: str):
    """Test audio preprocessing step by step"""
    print(f"\n{'='*60}")
    print(f"Testing audio preprocessing for: {audio_file_path}")
    print(f"{'='*60}")

    try:
        from src.preprocessing.audio_processor import AudioProcessor

        # Create processor
        print("🔧 Creating AudioProcessor...")
        processor = AudioProcessor(
            sample_rate=16000,
            n_mfcc=13,
            enable_deltas=True,
            enable_audio_augment=False,  # Disable for debugging
            enable_spec_augment=False    # Disable for debugging
        )
        print("✅ AudioProcessor created")

        # Test loading
        print("\n📥 Testing audio loading...")
        start_time = time.time()

        audio, sr = processor.load_audio(audio_file_path, timeout_seconds=30)
        load_time = time.time() - start_time

        print(f"✅ Audio loaded in {load_time:.2f}s")
        print(f"📊 Audio shape: {audio.shape}, Sample rate: {sr}")

        # Test MFCC extraction
        print("\n🎵 Testing MFCC extraction...")
        start_time = time.time()

        features = processor.extract_mfcc(audio)
        mfcc_time = time.time() - start_time

        print(f"✅ MFCC extracted in {mfcc_time:.2f}s")
        print(f"📊 Features shape: {features.shape}")

        # Test full preprocessing
        print("\n⚙️ Testing full preprocessing...")
        start_time = time.time()

        processed_features = processor.preprocess(audio_file_path, timeout_seconds=60)
        preprocess_time = time.time() - start_time

        print(f"✅ Preprocessing completed in {preprocess_time:.2f}s")
        print(f"📊 Final features shape: {processed_features.shape}")

        return True

    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        traceback.print_exc()
        return False

def test_model_loading():
    """Test model loading"""
    print(f"\n{'='*60}")
    print(f"Testing model loading")
    print(f"{'='*60}")

    try:
        import torch
        from src.api.speech_api import get_active_classifier

        print("🔧 Testing model loading...")
        start_time = time.time()

        clf, model_type = get_active_classifier()
        load_time = time.time() - start_time

        if clf:
            print(f"✅ Model loaded in {load_time:.2f}s")
            print(f"🤖 Model type: {model_type}")
            print(f"📊 Available intents: {len(clf['label_map'])}")
            return True
        else:
            print(f"❌ No model loaded")
            return False

    except Exception as e:
        print(f"❌ Error during model loading: {e}")
        traceback.print_exc()
        return False

def test_full_pipeline(audio_file_path: str):
    """Test the full pipeline like the API would"""
    print(f"\n{'='*60}")
    print(f"Testing FULL PIPELINE for: {audio_file_path}")
    print(f"{'='*60}")

    try:
        import torch
        from src.api.speech_api import get_active_classifier, preprocess_audio

        # Load model
        print("🤖 Loading model...")
        clf, model_type = get_active_classifier()
        if not clf:
            print("❌ No model available")
            return False
        print(f"✅ Model loaded: {model_type}")

        # Preprocess audio
        print("\n⚙️ Preprocessing audio...")
        start_time = time.time()

        tensor = preprocess_audio(audio_file_path, clf)
        preprocess_time = time.time() - start_time

        print(f"✅ Preprocessing completed in {preprocess_time:.2f}s")
        print(f"📊 Tensor shape: {tensor.shape}")

        # Run inference
        print("\n🧠 Running inference...")
        start_time = time.time()

        with torch.no_grad():
            outputs = clf["model"](tensor)

            # Handle tuple output
            if isinstance(outputs, tuple):
                intent_logits = outputs[0]
            else:
                intent_logits = outputs

            probs = torch.softmax(intent_logits, dim=1)
            conf, idx = probs.max(1)

        inference_time = time.time() - start_time

        # Get result
        label_map = clf["label_map"]
        idx_to_label = {v: k for k, v in label_map.items()}
        predicted_intent = idx_to_label.get(idx.item(), f"cls_{idx.item()}")

        print(f"✅ Inference completed in {inference_time:.2f}s")
        print(f"🎯 Predicted intent: {predicted_intent}")
        print(f"📊 Confidence: {float(conf.item()):.3f}")

        return True

    except Exception as e:
        print(f"❌ Error in full pipeline: {e}")
        traceback.print_exc()
        return False

def create_test_audio():
    """Create a simple test audio file"""
    try:
        import numpy as np
        import soundfile as sf

        # Create 3 seconds of test audio (sine wave)
        duration = 3.0
        sample_rate = 16000
        frequency = 440  # A4 note

        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.1 * np.sin(2 * np.pi * frequency * t)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            return tmp.name

    except Exception as e:
        print(f"❌ Failed to create test audio: {e}")
        return None

def main():
    """Main debugging function"""
    print("🔍 Audio Processing Pipeline Debugger")
    print("=====================================")

    # Check for command line audio file
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        if not os.path.exists(audio_file):
            print(f"❌ Audio file not found: {audio_file}")
            sys.exit(1)
    else:
        # Create test audio
        print("📝 No audio file provided, creating test audio...")
        audio_file = create_test_audio()
        if not audio_file:
            print("❌ Failed to create test audio")
            sys.exit(1)
        print(f"✅ Test audio created: {audio_file}")

    # Run tests
    tests = [
        ("Model Loading", lambda: test_model_loading()),
        ("Audio Conversion", lambda: test_audio_conversion(audio_file)),
        ("Audio Preprocessing", lambda: test_audio_preprocessing(audio_file)),
        ("Full Pipeline", lambda: test_full_pipeline(audio_file)),
    ]

    results = {}
    total_start = time.time()

    for test_name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"🧪 RUNNING TEST: {test_name}")
        print(f"{'='*80}")

        test_start = time.time()
        try:
            success = test_func()
            test_time = time.time() - test_start
            results[test_name] = (success, test_time)

            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"\n{status} - {test_name} completed in {test_time:.2f}s")

        except Exception as e:
            test_time = time.time() - test_start
            results[test_name] = (False, test_time)
            print(f"\n❌ FAILED - {test_name} crashed in {test_time:.2f}s: {e}")
            traceback.print_exc()

    # Summary
    total_time = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"📋 TEST SUMMARY")
    print(f"{'='*80}")

    passed = 0
    failed = 0

    for test_name, (success, test_time) in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name:<25} {test_time:>8.2f}s")
        if success:
            passed += 1
        else:
            failed += 1

    print(f"\n📊 Results: {passed} passed, {failed} failed")
    print(f"⏱️  Total time: {total_time:.2f}s")

    # Clean up test audio if we created it
    if len(sys.argv) <= 1 and audio_file and os.path.exists(audio_file):
        os.unlink(audio_file)
        print(f"🧹 Cleaned up test audio file")

    # Exit with error code if any tests failed
    if failed > 0:
        print(f"\n❌ {failed} test(s) failed. Check the logs above for details.")
        sys.exit(1)
    else:
        print(f"\n✅ All tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()
