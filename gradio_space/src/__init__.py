#!/usr/bin/env python3
"""
Optimized Twi Speech Recognition Engine - Source Package
======================================================

This package contains the core components of the optimized speech recognition
system that uses OpenAI Whisper for speech-to-text and custom intent
classification for Twi language.

Components:
- speech_recognizer: Core recognition engine
- api_server: FastAPI web server
- Audio processing utilities
- Intent classification system

Author: AI Assistant
Date: 2025-11-05
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "AI Assistant"
__description__ = "Optimized Twi Speech Recognition Engine"

# Import main components for easy access
try:
    from .speech_recognizer import (
        OptimizedSpeechRecognizer,
        AudioProcessor,
        WhisperTranscriber,
        TwiIntentClassifier,
        create_speech_recognizer,
    )

    # Mark as available
    __all__ = [
        "OptimizedSpeechRecognizer",
        "AudioProcessor",
        "WhisperTranscriber",
        "TwiIntentClassifier",
        "create_speech_recognizer",
        "__version__",
        "__author__",
        "__description__",
    ]

except ImportError as e:
    # Graceful degradation if dependencies not available
    import warnings

    warnings.warn(f"Some components not available: {e}")

    __all__ = [
        "__version__",
        "__author__",
        "__description__",
    ]

# Package metadata
PACKAGE_INFO = {
    "name": "optimized_twi_speech_engine",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "features": [
        "OpenAI Whisper Integration",
        "Twi Intent Classification",
        "Multi-format Audio Support",
        "Real-time Processing",
        "REST API Server",
        "Performance Monitoring",
    ],
    "supported_languages": ["tw"],  # Twi
    "supported_formats": ["wav", "webm", "mp3", "m4a"],
}


def get_package_info():
    """Get package information."""
    return PACKAGE_INFO.copy()


def get_version():
    """Get package version."""
    return __version__
