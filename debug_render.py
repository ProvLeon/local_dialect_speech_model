#!/usr/bin/env python3
"""
Debug script to check what's happening on Render deployment.
This script helps identify issues with file paths, imports, and environment.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_environment():
    """Debug the Render environment."""
    logger.info("=== RENDER DEBUG SCRIPT ===")

    # Environment info
    logger.info("=== ENVIRONMENT ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"User: {os.environ.get('USER', 'unknown')}")
    logger.info(f"Home: {os.environ.get('HOME', 'unknown')}")

    # Render-specific environment variables
    render_vars = ['RENDER', 'PORT', 'RENDER_SERVICE_ID', 'RENDER_SERVICE_NAME']
    for var in render_vars:
        value = os.environ.get(var, 'NOT SET')
        logger.info(f"{var}: {value}")

    # Project structure
    logger.info("=== PROJECT STRUCTURE ===")
    project_root = Path.cwd()
    logger.info(f"Project root: {project_root}")

    # List all items in project root
    logger.info("Contents of project root:")
    try:
        for item in sorted(project_root.iterdir()):
            if item.is_dir():
                logger.info(f"  📁 {item.name}/")
            else:
                size = item.stat().st_size
                logger.info(f"  📄 {item.name} ({size} bytes)")
    except Exception as e:
        logger.error(f"Error listing project root: {e}")

    # Check deployable package
    logger.info("=== DEPLOYABLE PACKAGE CHECK ===")
    deployable_path = project_root / "deployable_twi_speech_model"

    if deployable_path.exists():
        logger.info(f"✅ Found deployable package at: {deployable_path}")

        # Check utils directory
        utils_path = deployable_path / "utils"
        if utils_path.exists():
            logger.info(f"✅ Found utils directory at: {utils_path}")

            # List utils contents
            logger.info("Contents of utils directory:")
            for item in sorted(utils_path.iterdir()):
                logger.info(f"  📄 {item.name}")

            # Check specific files
            serve_file = utils_path / "serve.py"
            inference_file = utils_path / "inference.py"

            if serve_file.exists():
                logger.info(f"✅ Found serve.py")
                # Check first few lines for version info
                try:
                    with open(serve_file, 'r') as f:
                        lines = f.readlines()[:20]  # First 20 lines
                    for i, line in enumerate(lines, 1):
                        if 'version' in line.lower() or 'VERSION' in line:
                            logger.info(f"  Line {i}: {line.strip()}")
                except Exception as e:
                    logger.error(f"Error reading serve.py: {e}")
            else:
                logger.error("❌ serve.py not found")

            if inference_file.exists():
                logger.info(f"✅ Found inference.py")
                # Check first few lines for version info
                try:
                    with open(inference_file, 'r') as f:
                        lines = f.readlines()[:30]  # First 30 lines
                    for i, line in enumerate(lines, 1):
                        if 'version' in line.lower() or 'VERSION' in line or 'IntentOnlyModel' in line:
                            logger.info(f"  Line {i}: {line.strip()}")
                except Exception as e:
                    logger.error(f"Error reading inference.py: {e}")
            else:
                logger.error("❌ inference.py not found")
        else:
            logger.error(f"❌ Utils directory not found at: {utils_path}")

        # Check model directory
        model_path = deployable_path / "model"
        if model_path.exists():
            logger.info(f"✅ Found model directory")
            for item in sorted(model_path.iterdir()):
                size = item.stat().st_size if item.is_file() else 0
                logger.info(f"  📄 {item.name} ({size} bytes)")
        else:
            logger.error("❌ Model directory not found")

        # Check config
        config_path = deployable_path / "config" / "config.json"
        if config_path.exists():
            logger.info(f"✅ Found config.json")
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"  Model: {config.get('model_name', 'Unknown')}")
                logger.info(f"  Version: {config.get('version', 'Unknown')}")
                logger.info(f"  Type: {config.get('model_type', 'Unknown')}")
            except Exception as e:
                logger.error(f"Error reading config: {e}")
        else:
            logger.error("❌ config.json not found")
    else:
        logger.error(f"❌ Deployable package not found at: {deployable_path}")

    # Check Python path
    logger.info("=== PYTHON PATH ===")
    for i, path in enumerate(sys.path):
        logger.info(f"  {i}: {path}")

    # Test imports
    logger.info("=== IMPORT TESTS ===")

    # Test basic imports
    basic_imports = ['torch', 'numpy', 'librosa', 'fastapi', 'uvicorn']
    for module in basic_imports:
        try:
            __import__(module)
            logger.info(f"✅ {module} import successful")
        except ImportError as e:
            logger.error(f"❌ {module} import failed: {e}")

    # Test inference import
    if deployable_path.exists():
        utils_path = deployable_path / "utils"
        if utils_path.exists():
            # Add to path and try import
            sys.path.insert(0, str(utils_path))

            try:
                from inference import ModelInference
                logger.info("✅ ModelInference import successful")

                # Try to create instance
                try:
                    model = ModelInference(str(deployable_path))
                    logger.info("✅ ModelInference instance created successfully")

                    # Get model info
                    try:
                        info = model.get_model_info()
                        logger.info(f"✅ Model info: {info.get('model_name', 'Unknown')}")
                    except Exception as e:
                        logger.error(f"❌ Model info failed: {e}")

                except Exception as e:
                    logger.error(f"❌ ModelInference creation failed: {e}")

            except ImportError as e:
                logger.error(f"❌ ModelInference import failed: {e}")

    logger.info("=== DEBUG COMPLETE ===")

if __name__ == "__main__":
    debug_environment()
