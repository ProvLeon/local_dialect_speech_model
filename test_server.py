#!/usr/bin/env python3
"""
Simple test server to debug the API issues
"""

import os
import sys
import asyncio
import uvicorn
from src.api.speech_api import app, startup_event, get_active_classifier

async def test_startup():
    """Test the startup process"""
    print("=== Testing Startup Process ===")

    # Run startup event
    await startup_event()

    # Check classifier status
    classifier, model_type = get_active_classifier()
    if classifier:
        print(f"✓ Startup successful - Active classifier: {model_type}")
        print(f"✓ Label map classes: {len(classifier.get('label_map', {}))}")
        return True
    else:
        print("✗ Startup failed - No active classifier")
        return False

def run_test_server():
    """Run a test server"""
    print("=== Starting Test Server ===")
    print(f"Current working directory: {os.getcwd()}")

    # Test startup first
    startup_success = asyncio.run(test_startup())

    if not startup_success:
        print("❌ Startup test failed - not starting server")
        return False

    print("\n=== Starting Uvicorn Server ===")
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        return False

    return True

if __name__ == "__main__":
    success = run_test_server()
    sys.exit(0 if success else 1)
