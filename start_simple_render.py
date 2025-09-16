#!/usr/bin/env python3
"""
Simple, foolproof start script for Render deployment.
This script bypasses complex imports and directly starts the server.
"""

import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("=== Simple Render Start Script ===")

    # Get port from Render
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting on port: {port}")

    # Set up paths
    project_root = Path(__file__).parent
    deployable_path = project_root / "deployable_twi_speech_model"
    utils_path = deployable_path / "utils"

    # Verify structure
    if not deployable_path.exists():
        logger.error(f"Deployable package not found: {deployable_path}")
        sys.exit(1)

    if not utils_path.exists():
        logger.error(f"Utils directory not found: {utils_path}")
        sys.exit(1)

    serve_file = utils_path / "serve.py"
    if not serve_file.exists():
        logger.error(f"Serve file not found: {serve_file}")
        sys.exit(1)

    # Change to utils directory
    os.chdir(utils_path)
    logger.info(f"Changed to directory: {utils_path}")

    # Add to Python path
    sys.path.insert(0, str(utils_path))
    sys.path.insert(0, str(deployable_path))

    # Set environment
    os.environ["PORT"] = str(port)

    # Execute serve.py directly
    logger.info("Executing serve.py...")

    try:
        # Read and execute the serve.py file
        with open(serve_file, 'r') as f:
            serve_code = f.read()

        # Execute in the current environment
        exec(serve_code, {
            '__name__': '__main__',
            '__file__': str(serve_file),
        })

    except Exception as e:
        logger.error(f"Failed to execute serve.py: {e}")

        # Last resort: try direct import
        try:
            logger.info("Trying direct import method...")
            import uvicorn
            from serve import app

            logger.info("Starting uvicorn server...")
            uvicorn.run(app, host="0.0.0.0", port=port)

        except Exception as e2:
            logger.error(f"All methods failed: {e2}")
            sys.exit(1)

if __name__ == "__main__":
    main()
