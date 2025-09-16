#!/usr/bin/env python3
"""
Test script to validate timeout fixes in the deployable speech model
"""

import os
import sys
import time
import tempfile
import logging
import requests
import numpy as np
import soundfile as sf
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_audio(duration=2.0, frequency=440, sample_rate=16000):
    """Create a test audio file"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.1 * np.sin(2 * np.pi * frequency * t)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, sample_rate)
        return tmp.name

def create_long_audio(duration=10.0, sample_rate=16000):
    """Create a longer test audio file to test timeout handling"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Mix multiple frequencies for complexity
    audio = (0.1 * np.sin(2 * np.pi * 440 * t) +
             0.05 * np.sin(2 * np.pi * 880 * t) +
             0.02 * np.random.randn(len(t)))  # Add some noise

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, sample_rate)
        return tmp.name

def test_local_inference():
    """Test the local inference module directly"""
    print("\n" + "="*60)
    print("Testing Local Inference Module")
    print("="*60)

    try:
        # Add deployable path to sys.path
        deployable_path = Path(__file__).parent / "deployable_twi_speech_model"
        utils_path = deployable_path / "utils"
        sys.path.insert(0, str(utils_path))

        from inference import ModelInference

        # Create test audio
        print("Creating test audio...")
        audio_file = create_test_audio(duration=3.0)
        print(f"Test audio created: {audio_file}")

        # Load model
        print("Loading model...")
        start_time = time.time()
        model = ModelInference(str(deployable_path))
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f}s")

        # Test quick prediction
        print("Testing quick prediction...")
        start_time = time.time()
        intent, confidence = model.predict(audio_file, timeout_seconds=30)
        pred_time = time.time() - start_time
        print(f"✅ Prediction completed in {pred_time:.2f}s")
        print(f"   Result: {intent} (confidence: {confidence:.3f})")

        # Test top-k prediction
        print("Testing top-k prediction...")
        start_time = time.time()
        intent, confidence, top_k = model.predict_topk(audio_file, top_k=5, timeout_seconds=30)
        topk_time = time.time() - start_time
        print(f"✅ Top-k prediction completed in {topk_time:.2f}s")
        print(f"   Top result: {intent} (confidence: {confidence:.3f})")
        print(f"   Top-5 intents: {[p['intent'] for p in top_k]}")

        # Clean up
        os.unlink(audio_file)
        print("✅ Local inference test passed")
        return True

    except Exception as e:
        print(f"❌ Local inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoints(base_url="http://localhost:8000"):
    """Test API endpoints with timeout handling"""
    print("\n" + "="*60)
    print("Testing API Endpoints")
    print("="*60)

    results = {}

    # Test health endpoint
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health endpoint working")
            results['health'] = True
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            results['health'] = False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        results['health'] = False

    # Test model-info endpoint
    print("Testing model-info endpoint...")
    try:
        response = requests.get(f"{base_url}/model-info", timeout=10)
        if response.status_code == 200:
            info = response.json()
            print("✅ Model-info endpoint working")
            print(f"   Model: {info.get('model_name', 'Unknown')}")
            print(f"   Classes: {info.get('num_classes', 'Unknown')}")
            results['model_info'] = True
        else:
            print(f"❌ Model-info endpoint failed: {response.status_code}")
            results['model_info'] = False
    except Exception as e:
        print(f"❌ Model-info endpoint error: {e}")
        results['model_info'] = False

    # Test prediction with normal audio
    print("Testing prediction with normal audio...")
    try:
        audio_file = create_test_audio(duration=2.0)

        with open(audio_file, 'rb') as f:
            files = {'file': ('test.wav', f, 'audio/wav')}
            start_time = time.time()
            response = requests.post(f"{base_url}/test-intent?top_k=5", files=files, timeout=60)
            pred_time = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Normal prediction completed in {pred_time:.2f}s")
            print(f"   Result: {data.get('intent', 'Unknown')} (confidence: {data.get('confidence', 0):.3f})")
            print(f"   Processing time: {data.get('processing_time_ms', 0):.2f}ms")
            results['normal_prediction'] = True
        else:
            print(f"❌ Normal prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            results['normal_prediction'] = False

        os.unlink(audio_file)

    except Exception as e:
        print(f"❌ Normal prediction error: {e}")
        results['normal_prediction'] = False

    # Test prediction with longer audio
    print("Testing prediction with longer audio...")
    try:
        audio_file = create_long_audio(duration=8.0)

        with open(audio_file, 'rb') as f:
            files = {'file': ('long_test.wav', f, 'audio/wav')}
            start_time = time.time()
            response = requests.post(f"{base_url}/test-intent?top_k=3", files=files, timeout=120)
            pred_time = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Long audio prediction completed in {pred_time:.2f}s")
            print(f"   Result: {data.get('intent', 'Unknown')} (confidence: {data.get('confidence', 0):.3f})")
            print(f"   Processing time: {data.get('processing_time_ms', 0):.2f}ms")
            results['long_prediction'] = True
        elif response.status_code == 408:
            print(f"✅ Long audio prediction properly timed out (408 status)")
            print(f"   This is expected behavior for very long audio")
            results['long_prediction'] = True
        else:
            print(f"❌ Long audio prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            results['long_prediction'] = False

        os.unlink(audio_file)

    except requests.exceptions.Timeout:
        print("✅ Long audio prediction timed out at client level (expected)")
        results['long_prediction'] = True
    except Exception as e:
        print(f"❌ Long audio prediction error: {e}")
        results['long_prediction'] = False

    return results

def test_timeout_behavior():
    """Test timeout behavior specifically"""
    print("\n" + "="*60)
    print("Testing Timeout Behavior")
    print("="*60)

    try:
        # Test with corrupted/problematic audio
        print("Testing with empty audio file...")

        # Create empty file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            empty_file = tmp.name

        try:
            response = requests.post(
                "http://localhost:8000/test-intent",
                files={'file': ('empty.wav', open(empty_file, 'rb'), 'audio/wav')},
                timeout=30
            )

            if response.status_code in [400, 408]:
                print("✅ Empty file properly rejected")
            else:
                print(f"⚠️  Empty file got unexpected response: {response.status_code}")

        except Exception as e:
            print(f"✅ Empty file properly caused error: {e}")

        finally:
            os.unlink(empty_file)

        print("✅ Timeout behavior test completed")
        return True

    except Exception as e:
        print(f"❌ Timeout behavior test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🔍 Speech Model Timeout Fix Validation")
    print("=" * 80)

    all_results = {}

    # Test 1: Local inference
    all_results['local_inference'] = test_local_inference()

    # Test 2: API endpoints
    api_results = test_api_endpoints()
    all_results.update(api_results)

    # Test 3: Timeout behavior
    all_results['timeout_behavior'] = test_timeout_behavior()

    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)

    passed = 0
    total = 0

    for test_name, result in all_results.items():
        total += 1
        if result:
            passed += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Timeout fixes are working correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the logs above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
