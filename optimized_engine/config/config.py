#!/usr/bin/env python3
"""
Optimized Engine Configuration for Twi Speech Recognition
========================================================

This configuration uses Whisper for speech-to-text and focuses on intent classification
for the Twi language. This approach is more efficient and accurate than training
a speech recognition model from scratch.

Author: AI Assistant
Date: 2025-11-05
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional

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
        "model_size": "large-v3",  # Options: tiny, base, small, medium, large, large-v2, large-v3, custom
        "custom_model_path": str(
            MODELS_DIR / "whisper_twi"
        ),  # Path to fine-tuned Whisper model
        "language": None,  # Auto-detect language (Whisper doesn't officially support 'tw')
        "task": "transcribe",  # Always transcribe (not translate)
        "device": "auto",  # auto, cuda, cpu
        "compute_type": "float16",  # float16, float32, int8
        "beam_size": 5,  # Beam search size
        "best_of": 5,  # Number of candidates to consider
        "temperature": 0.0,  # Temperature for sampling (0.0 = deterministic)
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": True,
        "use_fine_tuned": True,  # Use fine-tuned model if available
    }

    # ==================== INTENT CLASSIFICATION ====================
    INTENT_CLASSIFIER = {
        "base_model": "microsoft/DialoGPT-medium",  # Base model for fine-tuning
        "custom_model_path": str(
            MODELS_DIR / "intent_classifier"
        ),  # Path to custom trained intent model
        "max_length": 512,  # Maximum token length for text input
        "confidence_threshold": 0.3,  # Lower threshold for better recall
        "top_k": 5,  # Return top-k intent predictions
        "device": "auto",  # auto, cuda, cpu
        "batch_size": 16,  # Batch size for training/inference
    }

    # ==================== SUPPORTED INTENTS ====================
    INTENTS = {
        # Navigation intents
        "go_home": {
            "description": "Navigate to home page",
            "examples": ["Kɔ fie", "Kɔ home", "Kɔ homepage"],
            "confidence_boost": 0.1,  # Boost confidence for common intents
        },
        "go_back": {
            "description": "Go back to previous page",
            "examples": ["San kɔ", "San w'akyi"],
            "confidence_boost": 0.1,
        },
        "continue": {
            "description": "Continue or go forward",
            "examples": ["Kɔ w'anim", "Kɔ anim"],
            "confidence_boost": 0.0,
        },
        # Search and browsing
        "search": {
            "description": "Search for products or items",
            "examples": ["Hwehwɛ", "Search", "Kɔ hwehwɛ"],
            "confidence_boost": 0.15,  # High priority for e-commerce
        },
        "show_items": {
            "description": "Show available items",
            "examples": ["Kyerɛ nneɛma", "Show items"],
            "confidence_boost": 0.1,
        },
        "show_description": {
            "description": "Show item description",
            "examples": ["Kyerɛ nsɛm", "Show description"],
            "confidence_boost": 0.0,
        },
        "show_price": {
            "description": "Show item price",
            "examples": ["Kyerɛ boɔ", "Show price"],
            "confidence_boost": 0.1,
        },
        # Cart operations
        "show_cart": {
            "description": "Display shopping cart",
            "examples": ["Kɔ cart", "Kɔ shopping cart", "Kyerɛ cart"],
            "confidence_boost": 0.15,
        },
        "add_to_cart": {
            "description": "Add item to cart",
            "examples": ["De kɔ cart mu", "Add to cart"],
            "confidence_boost": 0.15,
        },
        "remove_from_cart": {
            "description": "Remove item from cart",
            "examples": ["Yi fi cart mu", "Remove from cart"],
            "confidence_boost": 0.1,
        },
        "change_quantity": {
            "description": "Change item quantity",
            "examples": ["Sesa dodow", "Change quantity"],
            "confidence_boost": 0.1,
        },
        # Product customization
        "select_size": {
            "description": "Select product size",
            "examples": ["Yi size", "Select size"],
            "confidence_boost": 0.1,
        },
        "select_color": {
            "description": "Select product color",
            "examples": ["Yi color", "Select color"],
            "confidence_boost": 0.1,
        },
        "change_size": {
            "description": "Change product size",
            "examples": ["Sesa size", "Change size"],
            "confidence_boost": 0.0,
        },
        "change_color": {
            "description": "Change product color",
            "examples": ["Sesa color", "Change color"],
            "confidence_boost": 0.0,
        },
        # Filters and sorting
        "set_filter": {
            "description": "Set search filters",
            "examples": ["Set filter", "Sesa filter"],
            "confidence_boost": 0.1,
        },
        "clear_filter": {
            "description": "Clear search filters",
            "examples": ["Clear filter", "Yi filter"],
            "confidence_boost": 0.0,
        },
        # Checkout and payment
        "checkout": {
            "description": "Proceed to checkout",
            "examples": ["Kɔ checkout", "Tua ka"],
            "confidence_boost": 0.15,
        },
        "make_payment": {
            "description": "Make payment",
            "examples": ["Tua ka", "Make payment"],
            "confidence_boost": 0.15,
        },
        # User account
        "orders": {
            "description": "View user orders",
            "examples": ["Kɔ me orders", "Kyerɛ orders"],
            "confidence_boost": 0.1,
        },
        "wishlist": {
            "description": "View wishlist",
            "examples": ["Kɔ wishlist", "Kyerɛ wishlist"],
            "confidence_boost": 0.1,
        },
        "save_for_later": {
            "description": "Save item for later",
            "examples": ["Save for later", "Kora ma akyiri"],
            "confidence_boost": 0.0,
        },
        # General actions
        "help": {
            "description": "Get help or assistance",
            "examples": ["Help", "Boa me", "Mmoa"],
            "confidence_boost": 0.1,
        },
        "ask_questions": {
            "description": "Ask questions about products",
            "examples": ["Bisa nsɛm", "Ask questions"],
            "confidence_boost": 0.0,
        },
        "cancel": {
            "description": "Cancel current action",
            "examples": ["Cancel", "Gyae", "Stop"],
            "confidence_boost": 0.1,
        },
        "fast_delivery": {
            "description": "Request fast delivery",
            "examples": ["Fast delivery", "Ntɛm delivery"],
            "confidence_boost": 0.0,
        },
        "change_order": {
            "description": "Change order details",
            "examples": ["Sesa order", "Change order"],
            "confidence_boost": 0.0,
        },
    }

    # ==================== AUDIO PROCESSING ====================
    AUDIO = {
        "sample_rate": 16000,  # Standard sample rate for speech
        "channels": 1,  # Mono audio
        "format": "wav",  # Preferred audio format
        "max_duration": 30,  # Maximum audio duration in seconds
        "min_duration": 0.5,  # Minimum audio duration in seconds
        "chunk_duration": 5,  # Chunk duration for streaming
        "overlap_duration": 0.5,  # Overlap between chunks
        "noise_reduction": True,  # Apply noise reduction
        "volume_normalization": True,  # Normalize audio volume
    }

    # ==================== MODEL TRAINING ====================
    TRAINING = {
        "intent_classifier": {
            "epochs": 10,
            "batch_size": 16,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_steps": 500,
            "max_grad_norm": 1.0,
            "eval_steps": 100,
            "save_steps": 500,
            "logging_steps": 50,
            "early_stopping_patience": 3,
            "train_test_split": 0.8,
            "validation_split": 0.1,
        },
        "data_augmentation": {
            "enabled": True,
            "techniques": {
                "synonym_replacement": True,
                "random_insertion": True,
                "random_swap": True,
                "random_deletion": False,  # Be careful with Twi
                "back_translation": False,  # Not available for Twi
            },
            "augmentation_ratio": 0.3,  # 30% augmentation
        },
    }

    # ==================== API CONFIGURATION ====================
    API = {
        "host": "0.0.0.0",
        "port": 8000,
        "debug": False,
        "cors_origins": ["*"],
        "max_file_size": 50 * 1024 * 1024,  # 50MB
        "timeout": 120,  # 2 minutes
        "rate_limiting": {
            "enabled": True,
            "requests_per_minute": 60,
            "burst_size": 10,
        },
        "response_format": {
            "include_confidence": True,
            "include_alternatives": True,
            "include_timing": True,
            "include_segments": False,  # Whisper segments
        },
    }

    # ==================== LOGGING CONFIGURATION ====================
    LOGGING = {
        "level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file_logging": True,
        "console_logging": True,
        "log_file": str(LOGS_DIR / "optimized_engine.log"),
        "max_file_size": 10 * 1024 * 1024,  # 10MB
        "backup_count": 5,
        "performance_logging": True,
    }

    # ==================== PERFORMANCE OPTIMIZATION ====================
    PERFORMANCE = {
        "use_gpu": True,
        "mixed_precision": True,
        "batch_inference": True,
        "model_caching": True,
        "result_caching": {
            "enabled": True,
            "max_size": 1000,  # Cache up to 1000 results
            "ttl": 3600,  # 1 hour TTL
        },
        "parallel_processing": {
            "enabled": True,
            "max_workers": 4,
        },
    }

    # ==================== DEPLOYMENT CONFIGURATION ====================
    DEPLOYMENT = {
        "environment": os.getenv("ENVIRONMENT", "development"),
        "version": "2.0.0",
        "health_check_interval": 30,  # seconds
        "metrics_collection": True,
        "error_reporting": True,
        "model_monitoring": {
            "enabled": True,
            "confidence_threshold_alert": 0.3,
            "error_rate_threshold": 0.1,
        },
    }

    # ==================== HELPER METHODS ====================
    @classmethod
    def get_intent_list(cls) -> List[str]:
        """Get list of all supported intents."""
        return list(cls.INTENTS.keys())

    @classmethod
    def get_intent_examples(cls, intent: str) -> List[str]:
        """Get example phrases for a specific intent."""
        return cls.INTENTS.get(intent, {}).get("examples", [])

    @classmethod
    def get_high_priority_intents(cls) -> List[str]:
        """Get list of high-priority intents (with confidence boost > 0.1)."""
        return [
            intent
            for intent, config in cls.INTENTS.items()
            if config.get("confidence_boost", 0) > 0.1
        ]

    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings."""
        # Check if required directories exist
        required_dirs = [DATA_DIR, MODELS_DIR, LOGS_DIR]
        for dir_path in required_dirs:
            if not dir_path.exists():
                print(f"Warning: Directory {dir_path} does not exist")
                return False

        # Validate intent configuration
        if len(cls.INTENTS) < 5:
            print("Warning: Less than 5 intents configured")
            return False

        # Validate audio configuration
        if cls.AUDIO["sample_rate"] not in [8000, 16000, 22050, 44100]:
            print("Warning: Unusual sample rate configured")

        print("Configuration validation passed")
        return True

    @classmethod
    def get_device(cls) -> str:
        """Get the appropriate device for model inference."""
        import torch

        if cls.WHISPER["device"] == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return cls.WHISPER["device"]

    @classmethod
    def print_config_summary(cls):
        """Print a summary of the current configuration."""
        print("=" * 60)
        print("OPTIMIZED TWI SPEECH ENGINE CONFIGURATION")
        print("=" * 60)
        print(f"Whisper Model: {cls.WHISPER['model_size']}")
        print(f"Language: {cls.WHISPER['language']}")
        print(f"Supported Intents: {len(cls.INTENTS)}")
        print(f"Device: {cls.get_device()}")
        print(f"Environment: {cls.DEPLOYMENT['environment']}")
        print(f"Version: {cls.DEPLOYMENT['version']}")
        print("=" * 60)

        # High priority intents
        high_priority = cls.get_high_priority_intents()
        if high_priority:
            print(f"High Priority Intents: {', '.join(high_priority)}")
            print("=" * 60)


# Create default configuration instance
config = OptimizedConfig()

# Environment-specific overrides
if os.getenv("ENVIRONMENT") == "production":
    config.LOGGING["level"] = "WARNING"
    config.API["debug"] = False
    config.PERFORMANCE["result_caching"]["enabled"] = True
elif os.getenv("ENVIRONMENT") == "development":
    config.LOGGING["level"] = "DEBUG"
    config.API["debug"] = True
    config.PERFORMANCE["result_caching"]["enabled"] = False

# GPU availability check
try:
    import torch

    if not torch.cuda.is_available():
        config.WHISPER["compute_type"] = "float32"
        config.PERFORMANCE["mixed_precision"] = False
except ImportError:
    pass
