#!/usr/bin/env python3
"""
Optimized Engine Configuration for Twi Speech Recognition
========================================================

This configuration uses a multi-task Whisper model for both speech-to-text
and intent classification for the Twi language.

Author: AI Assistant
Date: 2025-11-07
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)


class OptimizedConfig:
    """Configuration class for the optimized Twi speech recognition engine."""

    # ==================== WHISPER CONFIGURATION ====================
    WHISPER = {
        "model_size": "small",  # Using small as the base for multi-task model
        "custom_model_path": str(
            MODELS_DIR / "whisper_twi"
        ),  # Path to fine-tuned multi-task Whisper model
        "language": "tw",
        "task": "transcribe",
        "device": "auto",
        "compute_type": "float16",
        "beam_size": 5,
        "temperature": 0.0,
        "use_fine_tuned": True,
    }

    # ==================== SUPPORTED INTENTS ====================
    # This is now mainly for reference, as intents are learned by the model
    INTENTS = {
        "go_home": {"description": "Navigate to home page"},
        "go_back": {"description": "Go back to previous page"},
        "continue": {"description": "Continue or go forward"},
        "search": {"description": "Search for products or items"},
        "show_items": {"description": "Show available items"},
        "show_description": {"description": "Show item description"},
        "show_price": {"description": "Show item price"},
        "show_cart": {"description": "Display shopping cart"},
        "add_to_cart": {"description": "Add item to cart"},
        "remove_from_cart": {"description": "Remove item from cart"},
        "change_quantity": {"description": "Change item quantity"},
        "select_size": {"description": "Select product size"},
        "select_color": {"description": "Select product color"},
        "change_size": {"description": "Change product size"},
        "change_color": {"description": "Change product color"},
        "set_filter": {"description": "Set search filters"},
        "clear_filter": {"description": "Clear search filters"},
        "checkout": {"description": "Proceed to checkout"},
        "make_payment": {"description": "Make payment"},
        "orders": {"description": "View user orders"},
        "wishlist": {"description": "View wishlist"},
        "save_for_later": {"description": "Save item for later"},
        "help": {"description": "Get help or assistance"},
        "ask_questions": {"description": "Ask questions about products"},
        "cancel": {"description": "Cancel current action"},
        "fast_delivery": {"description": "Request fast delivery"},
        "change_order": {"description": "Change order details"},
    }

    # ==================== AUDIO PROCESSING ====================
    AUDIO = {
        "sample_rate": 16000,
        "channels": 1,
        "format": "wav",
        "max_duration": 30,
        "min_duration": 0.5,
    }

    # ==================== MODEL TRAINING ====================
    TRAINING = {
        "data_augmentation": {
            "enabled": False,  # Keep it simple for now
        },
    }

    # ==================== API CONFIGURATION ====================
    API = {
        "host": "0.0.0.0",
        "port": 8000,
        "debug": False,
        "cors_origins": ["*"],
    }

    # ==================== LOGGING CONFIGURATION ====================
    LOGGING = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file_logging": True,
        "console_logging": True,
        "log_file": str(LOGS_DIR / "optimized_engine.log"),
    }

    # ==================== PERFORMANCE OPTIMIZATION ====================
    PERFORMANCE = {
        "use_gpu": True,
        "mixed_precision": True,
    }

    # ==================== DEPLOYMENT CONFIGURATION ====================
    DEPLOYMENT = {
        "environment": os.getenv("ENVIRONMENT", "development"),
        "version": "3.0.0",  # Bump version for multi-task model
    }

    # ==================== HELPER METHODS ====================
    @classmethod
    def get_intent_list(cls) -> List[str]:
        return list(cls.INTENTS.keys())

    @classmethod
    def get_device(cls) -> str:
        import torch

        if cls.PERFORMANCE["use_gpu"] and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @classmethod
    def print_config_summary(cls):
        print("=" * 60)
        print("OPTIMIZED TWI SPEECH ENGINE CONFIGURATION")
        print("=" * 60)
        print(f"Whisper Model: {cls.WHISPER['model_size']} (multi-task)")
        print(f"Language: {cls.WHISPER['language']}")
        print(f"Supported Intents: {len(cls.INTENTS)}")
        print(f"Device: {cls.get_device()}")
        print(f"Version: {cls.DEPLOYMENT['version']}")
        print("=" * 60)


# Create default configuration instance
config = OptimizedConfig()

# Environment-specific overrides
if os.getenv("ENVIRONMENT") == "production":
    config.LOGGING["level"] = "WARNING"
    config.API["debug"] = False
elif os.getenv("ENVIRONMENT") == "development":
    config.LOGGING["level"] = "DEBUG"
    config.API["debug"] = True

# GPU availability check
try:
    import torch

    if not torch.cuda.is_available():
        config.WHISPER["compute_type"] = "float32"
        config.PERFORMANCE["mixed_precision"] = False
except ImportError:
    pass
