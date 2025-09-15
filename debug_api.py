#!/usr/bin/env python3
"""
Debug script to test API model loading and identify issues
"""

import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported"""
    print("=== Testing Imports ===")
    try:
        from src.api.speech_api import app
        print("✓ FastAPI app import successful")
    except Exception as e:
        print(f"✗ FastAPI app import failed: {e}")
        return False

    try:
        from src.models.speech_model import ImprovedTwiSpeechModel
        print("✓ ImprovedTwiSpeechModel import successful")
    except Exception as e:
        print(f"✗ ImprovedTwiSpeechModel import failed: {e}")
        return False

    try:
        from src.preprocessing.audio_processor import AudioProcessor
        print("✓ AudioProcessor import successful")
    except Exception as e:
        print(f"✗ AudioProcessor import failed: {e}")
        return False

    return True

def test_path_resolution():
    """Test path resolution logic"""
    print("\n=== Testing Path Resolution ===")

    try:
        from src.api.speech_api import (
            resolve_model_paths,
            apply_resolved_paths,
            initialize_paths,
            ENHANCED_MODEL_PATH,
            STANDARD_MODEL_PATH,
            LABEL_MAP_PATH
        )

        print("Before path resolution:")
        print(f"  ENHANCED_MODEL_PATH: '{ENHANCED_MODEL_PATH}'")
        print(f"  STANDARD_MODEL_PATH: '{STANDARD_MODEL_PATH}'")
        print(f"  LABEL_MAP_PATH: '{LABEL_MAP_PATH}'")

        # Test resolve_model_paths directly
        resolved_paths, candidates = resolve_model_paths()
        print(f"\nResolved paths:")
        for key, path in resolved_paths.items():
            exists = os.path.exists(path)
            print(f"  {key}: '{path}' (exists: {exists})")

        # Test apply_resolved_paths
        apply_resolved_paths(resolved_paths, candidates)

        # Import the variables again to check if they were updated
        import importlib
        speech_api_module = importlib.import_module('src.api.speech_api')

        print(f"\nAfter apply_resolved_paths:")
        print(f"  ENHANCED_MODEL_PATH: '{speech_api_module.ENHANCED_MODEL_PATH}'")
        print(f"  STANDARD_MODEL_PATH: '{speech_api_module.STANDARD_MODEL_PATH}'")
        print(f"  LABEL_MAP_PATH: '{speech_api_module.LABEL_MAP_PATH}'")

        return resolved_paths

    except Exception as e:
        print(f"✗ Path resolution failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_model_loading(resolved_paths):
    """Test model loading with resolved paths"""
    print("\n=== Testing Model Loading ===")

    if not resolved_paths:
        print("✗ No resolved paths available")
        return False

    try:
        from src.api.speech_api import (
            load_label_map_safe,
            create_model,
            create_processor
        )

        # Test label map loading
        print("Testing label map loading...")
        label_map = load_label_map_safe()
        if label_map:
            print(f"✓ Label map loaded with {len(label_map)} classes")
            print(f"  Sample classes: {list(label_map.keys())[:5]}")
        else:
            print("✗ Label map loading failed")
            return False

        # Test enhanced model loading
        enhanced_path = resolved_paths.get("ENHANCED_MODEL_PATH")
        if enhanced_path and os.path.exists(enhanced_path):
            print(f"Testing enhanced model loading from: {enhanced_path}")
            model = create_model(enhanced_path, label_map)
            if model:
                print(f"✓ Enhanced model loaded successfully")
                print(f"  Model type: {type(model)}")
                print(f"  Input dim: {getattr(model, 'input_dim', 'unknown')}")
                print(f"  Num classes: {getattr(model, 'num_classes', 'unknown')}")
            else:
                print("✗ Enhanced model loading failed")
                return False
        else:
            print(f"✗ Enhanced model path not found: {enhanced_path}")
            return False

        # Test processor creation
        print("Testing processor creation...")
        processor = create_processor()
        if processor:
            print(f"✓ Processor created: {type(processor)}")
        else:
            print("✗ Processor creation failed")
            return False

        return True

    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_classifier_creation():
    """Test full classifier creation"""
    print("\n=== Testing Classifier Creation ===")

    try:
        from src.api.speech_api import (
            load_enhanced_model,
            load_standard_model,
            get_active_classifier
        )

        # Test enhanced model loading
        print("Testing enhanced model loading...")
        enhanced_loaded = load_enhanced_model()
        print(f"Enhanced model loaded: {enhanced_loaded}")

        if not enhanced_loaded:
            print("Testing standard model loading...")
            standard_loaded = load_standard_model()
            print(f"Standard model loaded: {standard_loaded}")

        # Test getting active classifier
        classifier, model_type = get_active_classifier()
        if classifier:
            print(f"✓ Active classifier found: {model_type}")
            print(f"  Label map classes: {len(classifier.get('label_map', {}))}")
            print(f"  Model type: {classifier.get('model_type', 'unknown')}")
        else:
            print("✗ No active classifier found")
            return False

        return True

    except Exception as e:
        print(f"✗ Classifier creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_startup_sequence():
    """Test the full startup sequence"""
    print("\n=== Testing Startup Sequence ===")

    try:
        from src.api.speech_api import startup_event
        import asyncio

        print("Running startup event...")
        asyncio.run(startup_event())

        # Check final state
        from src.api.speech_api import get_active_classifier
        classifier, model_type = get_active_classifier()

        if classifier:
            print(f"✓ Startup successful - Active classifier: {model_type}")
            return True
        else:
            print("✗ Startup failed - No active classifier")
            return False

    except Exception as e:
        print(f"✗ Startup sequence failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests"""
    print("Starting API Diagnostic Tests\n")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}...")

    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import tests failed - cannot continue")
        return False

    # Test 2: Path resolution
    resolved_paths = test_path_resolution()
    if not resolved_paths:
        print("\n❌ Path resolution failed - cannot continue")
        return False

    # Test 3: Model loading
    if not test_model_loading(resolved_paths):
        print("\n❌ Model loading failed - cannot continue")
        return False

    # Test 4: Classifier creation
    if not test_classifier_creation():
        print("\n❌ Classifier creation failed - cannot continue")
        return False

    # Test 5: Full startup sequence
    if not test_startup_sequence():
        print("\n❌ Startup sequence failed")
        return False

    print("\n✅ All diagnostic tests passed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
