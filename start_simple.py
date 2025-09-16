#!/usr/bin/env python3
"""
Simple backup start script for Render deployment
This is a minimal script that just starts the server without complex checks
"""

import os
import sys
from pathlib import Path

# Get port from environment (Render sets this)
port = int(os.environ.get("PORT", 8000))
print(f"Starting on port: {port}")

# Set up paths
project_root = Path(__file__).parent
deployable_path = project_root / "deployable_twi_speech_model"
serve_path = deployable_path / "utils"

# Change to the serve directory
os.chdir(serve_path)

# Add to Python path
sys.path.insert(0, str(deployable_path))
sys.path.insert(0, str(serve_path))

# Set environment for the serve script
os.environ["PORT"] = str(port)

# Import and run
try:
    from serve import app
    import uvicorn

    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=port)

except Exception as e:
    print(f"Error: {e}")

    # Fallback: execute serve.py directly
    serve_file = serve_path / "serve.py"
    if serve_file.exists():
        print("Trying fallback method...")
        exec(open(serve_file).read())
    else:
        print("Could not find serve.py")
        sys.exit(1)
