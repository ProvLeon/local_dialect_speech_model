#!/usr/bin/env python3
"""
Render start script for Twi Speech Model
This script ensures the model starts correctly on Render regardless of configuration issues
"""

import os
import sys
import subprocess
import logging
import json
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if we're running on Render and set up environment"""
    logger.info("Checking environment...")

    # Get port from environment (Render sets this automatically)
    port = os.environ.get("PORT", "8000")
    logger.info(f"Using port: {port}")

    # Check if we're on Render
    if os.environ.get("RENDER"):
        logger.info("Running on Render platform")
    else:
        logger.info("Running locally or on other platform")

    # Get project paths
    project_root = Path(__file__).parent.absolute()
    deployable_path = project_root / "deployable_twi_speech_model"

    logger.info(f"Project root: {project_root}")
    logger.info(f"Looking for deployable package at: {deployable_path}")

    if not deployable_path.exists():
        logger.error(f"Deployable package not found at: {deployable_path}")
        logger.error("Available directories:")
        for item in project_root.iterdir():
            if item.is_dir():
                logger.error(f"  - {item.name}")
        sys.exit(1)

    # Verify required files
    required_files = [
        deployable_path / "utils" / "serve.py",
        deployable_path / "requirements.txt",
        deployable_path / "model",
        deployable_path / "config"
    ]

    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        logger.error("Missing required files/directories:")
        for missing in missing_files:
            logger.error(f"  - {missing}")
        sys.exit(1)

    logger.info("All required files found")
    return port, deployable_path

def install_dependencies(deployable_path):
    """Install required dependencies"""
    logger.info("Installing dependencies...")

    requirements_path = deployable_path / "requirements.txt"

    if not requirements_path.exists():
        logger.error(f"Requirements file not found: {requirements_path}")
        sys.exit(1)

    try:
        # Read requirements to see what we're installing
        with open(requirements_path, 'r') as f:
            requirements = f.read().strip()
        logger.info(f"Installing packages: {requirements.replace(chr(10), ', ')}")

        # Install with verbose output
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_path)
        ], capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            logger.error(f"Failed to install dependencies: {result.stderr}")
            sys.exit(1)

        logger.info("Dependencies installed successfully")

    except subprocess.TimeoutExpired:
        logger.error("Dependency installation timed out")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error installing dependencies: {e}")
        sys.exit(1)

def verify_imports(deployable_path):
    """Verify that all required modules can be imported"""
    logger.info("Verifying imports...")

    # Add deployable path to Python path
    sys.path.insert(0, str(deployable_path))

    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")

        import fastapi
        logger.info(f"FastAPI version: {fastapi.__version__}")

        import uvicorn
        logger.info("Uvicorn imported successfully")

        import librosa
        logger.info(f"Librosa version: {librosa.__version__}")

        import numpy
        logger.info(f"NumPy version: {numpy.__version__}")

        # Test if we can import the serve module
        serve_path = deployable_path / "utils"
        sys.path.insert(0, str(serve_path))

        # Try importing the inference module
        try:
            from inference import ModelInference
            logger.info("ModelInference imported successfully")
        except ImportError as e:
            logger.warning(f"Could not import ModelInference: {e}")
            logger.info("Will try to continue without it")

        return True

    except ImportError as e:
        logger.error(f"Failed to import required module: {e}")
        return False

def start_server():
    """Start the FastAPI server"""
    logger.info("=== Starting Twi Speech Model on Render ===")

    port, deployable_path = check_environment()
    install_dependencies(deployable_path)

    if not verify_imports(deployable_path):
        logger.error("Import verification failed")
        sys.exit(1)

    # Set environment variables
    os.environ["PORT"] = str(port)
    os.environ["PYTHONPATH"] = str(deployable_path)

    # Change to deployable directory
    original_cwd = os.getcwd()
    serve_dir = deployable_path / "utils"

    try:
        os.chdir(serve_dir)
        logger.info(f"Changed directory to: {serve_dir}")

        # Start the server
        logger.info(f"Starting FastAPI server on port {port}...")

        # Import and run the server
        try:
            # Method 1: Try to run serve.py as a module
            serve_file = serve_dir / "serve.py"
            if serve_file.exists():
                logger.info("Executing serve.py...")

                # Read and execute the serve.py file
                with open(serve_file, 'r') as f:
                    serve_code = f.read()

                # Create a proper execution environment
                serve_globals = {
                    '__name__': '__main__',
                    '__file__': str(serve_file),
                }

                exec(serve_code, serve_globals)

            else:
                logger.error(f"serve.py not found at: {serve_file}")
                sys.exit(1)

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            logger.error("Attempting fallback method...")

            # Method 2: Direct uvicorn start
            try:
                import uvicorn
                from serve import app

                uvicorn.run(
                    app,
                    host="0.0.0.0",
                    port=int(port),
                    log_level="info"
                )

            except Exception as fallback_error:
                logger.error(f"Fallback method also failed: {fallback_error}")
                sys.exit(1)

    except Exception as e:
        logger.error(f"Critical error in server startup: {e}")
        sys.exit(1)

    finally:
        # Restore original working directory
        os.chdir(original_cwd)

def health_check():
    """Simple health check for debugging"""
    logger.info("=== Health Check ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Python path: {sys.path[:3]}...")  # Show first 3 entries

    # Show environment variables
    important_env_vars = ["PORT", "PYTHONPATH", "RENDER", "ENVIRONMENT"]
    for var in important_env_vars:
        value = os.environ.get(var, "Not set")
        logger.info(f"{var}: {value}")

if __name__ == "__main__":
    # Check if this is a health check call
    if len(sys.argv) > 1 and sys.argv[1] == "--health":
        health_check()
        sys.exit(0)

    # Normal startup
    try:
        start_server()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
