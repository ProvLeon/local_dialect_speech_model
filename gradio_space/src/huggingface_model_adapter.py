#!/usr/bin/env python3
"""
HuggingFace Model Adapter for Twi Speech Recognition
==================================================

This module provides an adapter layer to seamlessly work with both
single-task (transcription only) and multi-task (transcription + intent)
models from HuggingFace Hub.

Key Features:
1. Automatic model type detection
2. Unified interface for both model types
3. Dynamic loading and configuration
4. Fallback mechanisms for missing components
5. Performance optimization for HF models

Author: AI Assistant
Date: 2025-11-11
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HuggingFaceModelAdapter:
    """Adapter for HuggingFace models supporting both single and multi-task configurations."""

    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the HuggingFace model adapter.

        Args:
            model_path: Path to the HuggingFace model directory
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.model_path = Path(model_path)
        self.device = self._get_device(device)
        self.model_type = None  # 'single' or 'multi'

        # Model components
        self.transcription_model = None
        self.processor = None
        self.tokenizer = None
        self.intent_classifier = None

        # Configuration
        self.config = None
        self.intent_labels = {}
        self.supported_languages = ["tw", "en", "auto"]

        # Performance tracking
        self.stats = {
            "total_requests": 0,
            "transcription_requests": 0,
            "intent_requests": 0,
            "avg_transcription_time": 0.0,
            "avg_intent_time": 0.0,
        }

        self._initialize_model()

    def _get_device(self, device: str) -> str:
        """Determine the appropriate device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _initialize_model(self):
        """Initialize and load the model components."""
        logger.info(f"🚀 Initializing HuggingFace model from: {self.model_path}")

        try:
            # Load configuration
            self._load_configuration()

            # Detect model type
            self._detect_model_type()

            # Load model components based on type
            if self.model_type == "multi":
                self._load_multitask_model()
            else:
                self._load_singletask_model()

            # Apply optimizations
            self._apply_optimizations()

            logger.info(f"✅ Model initialized successfully ({self.model_type}-task)")

        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            raise

    def _load_configuration(self):
        """Load model configuration."""
        config_path = self.model_path / "config.json"

        if config_path.exists():
            with open(config_path, "r") as f:
                self.config = json.load(f)
            logger.info("✅ Configuration loaded from config.json")
        else:
            # Try to load using AutoConfig
            try:
                self.config = AutoConfig.from_pretrained(str(self.model_path))
                logger.info("✅ Configuration loaded using AutoConfig")
            except Exception as e:
                logger.warning(f"⚠️ Could not load config: {e}")
                self.config = {}

    def _detect_model_type(self):
        """Detect whether this is a single-task or multi-task model."""
        logger.info("🔍 Detecting model type...")

        # Check for multi-task indicators in config
        multi_task_indicators = [
            "num_labels",
            "custom_model",
            "task_types",
            "intent_classifier",
            "classification_head",
        ]

        config_has_multi = any(
            indicator in str(self.config) for indicator in multi_task_indicators
        )

        # Check for multi-task files
        multi_task_files = [
            "intent_labels.json",
            "label_map.json",
            "classification_head.bin",
            "pytorch_model.bin",  # Custom model weights
            "intent_config.json",
        ]

        files_have_multi = any(
            (self.model_path / file).exists() for file in multi_task_files
        )

        # Check model architecture name
        model_name = self.config.get("_name_or_path", "").lower()
        arch_name = self.config.get("architectures", [""])[0].lower()

        arch_has_multi = any(
            indicator in model_name or indicator in arch_name
            for indicator in ["multitask", "intent", "classification"]
        )

        if config_has_multi or files_have_multi or arch_has_multi:
            self.model_type = "multi"
            logger.info("🎯 Detected multi-task model (transcription + intent)")
        else:
            self.model_type = "single"
            logger.info("📝 Detected single-task model (transcription only)")

    def _load_singletask_model(self):
        """Load single-task (transcription only) model."""
        logger.info("📝 Loading single-task transcription model...")

        try:
            # Try to load as Whisper model first
            try:
                self.transcription_model = (
                    WhisperForConditionalGeneration.from_pretrained(
                        str(self.model_path),
                        torch_dtype=torch.float16
                        if self.device == "cuda"
                        else torch.float32,
                    )
                )
                self.processor = WhisperProcessor.from_pretrained(str(self.model_path))
                logger.info("✅ Loaded as Whisper model")
            except Exception:
                # Fallback to generic speech-to-text model
                self.transcription_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    str(self.model_path),
                    torch_dtype=torch.float16
                    if self.device == "cuda"
                    else torch.float32,
                )
                self.processor = AutoProcessor.from_pretrained(str(self.model_path))
                logger.info("✅ Loaded as generic speech-to-text model")

            # Move to device
            self.transcription_model = self.transcription_model.to(self.device)
            self.tokenizer = (
                self.processor.tokenizer
                if hasattr(self.processor, "tokenizer")
                else AutoTokenizer.from_pretrained(str(self.model_path))
            )

        except Exception as e:
            logger.error(f"❌ Failed to load single-task model: {e}")
            raise

    def _load_multitask_model(self):
        """Load multi-task (transcription + intent) model."""
        logger.info("🎯 Loading multi-task model...")

        try:
            # Load transcription component (same as single-task)
            self._load_singletask_model()

            # Load intent classification component
            self._load_intent_classifier()

            # Load intent labels
            self._load_intent_labels()

        except Exception as e:
            logger.error(f"❌ Failed to load multi-task model: {e}")
            raise

    def _load_intent_classifier(self):
        """Load intent classification component."""
        try:
            # Try to load as a separate classification model
            classifier_path = self.model_path / "intent_classifier"

            if classifier_path.exists():
                # Separate intent classifier directory
                self.intent_classifier = pipeline(
                    "text-classification",
                    model=str(classifier_path),
                    tokenizer=str(classifier_path),
                    device=0 if self.device == "cuda" else -1,
                )
                logger.info("✅ Loaded separate intent classifier")
            else:
                # Try to use the main model for classification
                try:
                    self.intent_classifier = pipeline(
                        "text-classification",
                        model=str(self.model_path),
                        tokenizer=str(self.model_path),
                        device=0 if self.device == "cuda" else -1,
                    )
                    logger.info("✅ Using main model for intent classification")
                except Exception:
                    logger.warning(
                        "⚠️ Could not load intent classifier, will use fallback"
                    )
                    self.intent_classifier = None

        except Exception as e:
            logger.warning(f"⚠️ Intent classifier loading failed: {e}")
            self.intent_classifier = None

    def _load_intent_labels(self):
        """Load intent label mappings."""
        label_files = ["intent_labels.json", "label_map.json", "labels.json"]

        for label_file in label_files:
            label_path = self.model_path / label_file
            if label_path.exists():
                try:
                    with open(label_path, "r") as f:
                        self.intent_labels = json.load(f)
                    logger.info(f"✅ Loaded intent labels from {label_file}")
                    break
                except Exception as e:
                    logger.warning(f"⚠️ Could not load {label_file}: {e}")

        if not self.intent_labels:
            logger.warning("⚠️ No intent labels found, using default mapping")
            self.intent_labels = {"label_to_id": {}, "id_to_label": {}}

    def _apply_optimizations(self):
        """Apply performance optimizations."""
        logger.info("⚡ Applying performance optimizations...")

        try:
            if self.transcription_model and self.device == "cuda":
                # Enable half precision if on GPU
                try:
                    self.transcription_model = self.transcription_model.half()
                    logger.info("✅ Half precision enabled")
                except Exception as e:
                    logger.warning(f"⚠️ Could not enable half precision: {e}")

                # Enable optimized attention if available
                try:
                    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                        torch.backends.cuda.enable_flash_sdp(True)
                        logger.info("✅ Flash attention enabled")
                except Exception:
                    pass

            # Set to evaluation mode
            if self.transcription_model:
                self.transcription_model.eval()

            # Compile model if PyTorch 2.0+
            try:
                if hasattr(torch, "compile") and self.transcription_model:
                    self.transcription_model = torch.compile(
                        self.transcription_model, mode="reduce-overhead"
                    )
                    logger.info("✅ Model compilation enabled")
            except Exception as e:
                logger.warning(f"⚠️ Model compilation failed: {e}")

        except Exception as e:
            logger.warning(f"⚠️ Optimization failed: {e}")

    def transcribe(self, audio_path: str, language: str = None) -> Dict[str, Any]:
        """
        Transcribe audio to text.

        Args:
            audio_path: Path to audio file
            language: Language code ('tw', 'en', 'auto')

        Returns:
            Dictionary with transcription results
        """
        start_time = time.time()
        self.stats["transcription_requests"] += 1

        try:
            if not self.transcription_model:
                raise ValueError("Transcription model not loaded")

            # Load and preprocess audio
            import librosa

            audio, sr = librosa.load(audio_path, sr=16000, mono=True)

            # Process with model
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = inputs.to(self.device)

            # Generate transcription
            with torch.no_grad():
                if language and language != "auto":
                    # Force specific language if specified
                    forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                        language=language, task="transcribe"
                    )
                    predicted_ids = self.transcription_model.generate(
                        inputs.input_features,
                        forced_decoder_ids=forced_decoder_ids,
                        max_length=448,
                        num_beams=1,
                        do_sample=False,
                    )
                else:
                    # Auto-detect language
                    predicted_ids = self.transcription_model.generate(
                        inputs.input_features,
                        max_length=448,
                        num_beams=1,
                        do_sample=False,
                    )

            # Decode transcription
            transcription = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]

            processing_time = time.time() - start_time
            self._update_transcription_stats(processing_time)

            return {
                "text": transcription.strip(),
                "language": language or "auto",
                "confidence": 0.95,  # HF models typically have good confidence
                "processing_time": processing_time,
                "model_type": "huggingface",
                "segments": [],
            }

        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            return {
                "text": "",
                "language": language or "auto",
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "error": str(e),
            }

    def classify_intent(self, text: str) -> Dict[str, Any]:
        """
        Classify intent from text.

        Args:
            text: Input text to classify

        Returns:
            Dictionary with intent classification results
        """
        start_time = time.time()
        self.stats["intent_requests"] += 1

        try:
            if self.model_type == "single":
                # Fallback to simple keyword-based classification for single-task models
                return self._fallback_intent_classification(text)

            if not self.intent_classifier:
                return self._fallback_intent_classification(text)

            # Use trained intent classifier
            results = self.intent_classifier(text, top_k=5)

            # Format results
            alternatives = []
            for result in results:
                alternatives.append(
                    {
                        "label": result["label"],
                        "score": float(result["score"]),
                    }
                )

            processing_time = time.time() - start_time
            self._update_intent_stats(processing_time)

            return {
                "intent": alternatives[0]["label"] if alternatives else "unknown",
                "confidence": alternatives[0]["score"] if alternatives else 0.0,
                "alternatives": alternatives,
                "processing_time": processing_time,
                "method": "huggingface_classifier",
            }

        except Exception as e:
            logger.error(f"❌ Intent classification failed: {e}")
            return self._fallback_intent_classification(text)

    def _fallback_intent_classification(self, text: str) -> Dict[str, Any]:
        """Fallback intent classification using simple keyword matching."""

        # Twi e-commerce intent keywords based on prompts_lean.csv
        intent_keywords = {
            # Navigation
            "go_home": ["kɔ fie", "kɔ home", "kɔ homepage"],
            "go_back": ["san w'akyi", "san kɔ"],
            "continue": ["kɔ w'anim"],
            "show_cart": ["kɔ cart", "cart no mu"],
            "open_account": ["kɔ me account", "kɔ me akawnt"],
            "open_orders": ["kɔ me orders"],
            "open_wishlist": ["kɔ wishlist"],
            # Search & Discovery
            "search": ["hwehwɛ", "hwɛ"],
            "apply_filter": ["fa filter", "fa to so"],
            "clear_filter": ["yi filter"],
            "sort_items": ["sort by", "fa di kan"],
            # Product Info
            "show_description": ["kyerɛ", "ho nsɛm", "kenkan"],
            "show_price": ["kyerɛ", "boɔ"],
            "show_reviews": ["kyerɛ reviews"],
            "show_similar_items": ["kyerɛ nea ɛte sɛ", "te sɛ yei"],
            # Cart Operations
            "add_to_cart": ["fa yei to cart", "fa yei ka me cart"],
            "remove_from_cart": ["yi yei firi cart", "yi yei firi me cart"],
            "save_for_later": ["fa yei to wishlist", "fa yei ka me wishlist"],
            "change_quantity": ["fa baako ka ho", "yi baako", "fa baako bio"],
            "set_quantity": ["hyɛ dodow no yɛ"],
            "clear_cart": ["pepa cart", "pepa me cart"],
            # Checkout & Payment
            "checkout": ["fa me kɔ checkout", "kɔ checkout"],
            "confirm_order": ["pintim me order"],
            "make_payment": [
                "fa card",
                "fa mobile money",
                "fa momo",
                "tua ka",
                "tua seesei",
            ],
            "cancel_order": ["gyae order", "twa order no mu"],
            # Post-Purchase
            "show_orders": ["kyerɛ me orders", "hwɛ me orders"],
            "show_order_status": ["kyerɛ me order status"],
            "track_order": ["trake me order"],
            "return_item": ["mepɛ sɛ mesan de adeɛ no ba"],
            "exchange_item": ["mepɛ sɛ mesesa adeɛ no"],
            # Addresses
            "show_addresses": ["kyerɛ me addresses"],
            "add_address": ["fa address foforɔ ka ho"],
            "remove_address": ["yi address"],
            "set_default_address": ["fa default address"],
            # Promotions
            "apply_coupon": ["fa coupon to so"],
            "remove_coupon": ["yi coupon"],
            # Notifications
            "enable_order_updates": ["sɔ order updates"],
            "disable_order_updates": ["gyae order updates"],
            "enable_price_alert": ["sɔ price alert"],
            "disable_price_alert": ["gyae price alert"],
            # Support
            "help": ["boa me", "sɔ mmoa"],
            "start_live_chat": ["frɛ live chat", "kɔ live chat"],
            "show_faqs": ["kyerɛ faqs"],
            # Variant Selection
            "select_color": ["fa color"],
            "change_color": ["sesa color"],
            "select_size": ["fa size"],
            "change_size": ["sesa size"],
        }

        text_lower = text.lower()
        scores = {}

        for intent, keywords in intent_keywords.items():
            score = 0.0
            matches = 0
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    # Give higher weight to exact matches
                    if keyword.lower() == text_lower.strip():
                        score += 2.0 / len(keywords)
                    else:
                        score += 1.0 / len(keywords)
                    matches += 1

            # Boost score for multiple keyword matches
            if matches > 1:
                score *= 1.5

            if score > 0:
                scores[intent] = score

        # Sort by score
        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if sorted_intents:
            best_intent, best_score = sorted_intents[0]
            alternatives = [
                {"label": intent, "score": min(0.9, score)}  # Cap at 90%
                for intent, score in sorted_intents[:5]
            ]
        else:
            best_intent = "search"  # Default fallback for unknown commands
            best_score = 0.1
            alternatives = [{"label": "search", "score": 0.1}]

        return {
            "intent": best_intent,
            "confidence": min(0.8, best_score),  # Cap at 80% for fallback
            "alternatives": alternatives,
            "processing_time": 0.01,
            "method": "fallback_keywords",
        }

    def recognize(self, audio_path: str, language: str = None) -> Dict[str, Any]:
        """
        Complete recognition pipeline (transcription + intent if available).

        Args:
            audio_path: Path to audio file
            language: Language code

        Returns:
            Dictionary with complete recognition results
        """
        start_time = time.time()
        self.stats["total_requests"] += 1

        try:
            # Step 1: Transcription
            transcription_result = self.transcribe(audio_path, language)

            if not transcription_result.get("text"):
                return {
                    "status": "failed",
                    "error": "Transcription failed",
                    "processing_time": time.time() - start_time,
                }

            # Step 2: Intent classification (if available)
            intent_result = self.classify_intent(transcription_result["text"])

            # Step 3: Combine results
            return {
                "status": "success",
                "transcription": transcription_result,
                "intent": intent_result,
                "processing_time": time.time() - start_time,
                "model_info": {
                    "model_path": str(self.model_path),
                    "model_type": self.model_type,
                    "device": self.device,
                },
            }

        except Exception as e:
            logger.error(f"❌ Recognition failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    def _update_transcription_stats(self, processing_time: float):
        """Update transcription performance statistics."""
        current_avg = self.stats["avg_transcription_time"]
        count = self.stats["transcription_requests"]
        self.stats["avg_transcription_time"] = (
            current_avg * (count - 1) + processing_time
        ) / count

    def _update_intent_stats(self, processing_time: float):
        """Update intent classification performance statistics."""
        current_avg = self.stats["avg_intent_time"]
        count = self.stats["intent_requests"]
        self.stats["avg_intent_time"] = (
            current_avg * (count - 1) + processing_time
        ) / count

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and capabilities."""
        return {
            "model_path": str(self.model_path),
            "model_type": self.model_type,
            "device": self.device,
            "capabilities": {
                "transcription": self.transcription_model is not None,
                "intent_classification": self.intent_classifier is not None,
                "multi_task": self.model_type == "multi",
            },
            "supported_languages": self.supported_languages,
            "intent_labels": list(self.intent_labels.get("id_to_label", {}).values()),
            "statistics": self.stats,
        }

    def get_supported_intents(self) -> List[Dict[str, str]]:
        """Get list of supported intents."""
        if self.model_type == "multi" and self.intent_labels:
            intents = []
            id_to_label = self.intent_labels.get("id_to_label", {})
            for intent_id, intent_name in id_to_label.items():
                intents.append(
                    {
                        "intent": intent_name,
                        "id": intent_id,
                        "description": f"Twi e-commerce intent: {intent_name}",
                    }
                )
            return intents
        else:
            # Fallback intents for single-task models based on e-commerce functionality
            return [
                {"intent": "search", "description": "Search for products"},
                {"intent": "add_to_cart", "description": "Add items to shopping cart"},
                {"intent": "show_cart", "description": "View shopping cart"},
                {"intent": "checkout", "description": "Proceed to checkout"},
                {"intent": "show_orders", "description": "View order history"},
                {"intent": "track_order", "description": "Track order status"},
                {"intent": "help", "description": "Request for assistance"},
                {"intent": "go_home", "description": "Navigate to homepage"},
                {"intent": "go_back", "description": "Go back to previous page"},
                {"intent": "apply_filter", "description": "Apply product filters"},
                {"intent": "show_description", "description": "Show product details"},
                {"intent": "make_payment", "description": "Process payment"},
                {"intent": "return_item", "description": "Return purchased items"},
                {"intent": "add_address", "description": "Add delivery address"},
                {"intent": "apply_coupon", "description": "Apply discount coupon"},
            ]

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the model."""
        return {
            "status": "healthy" if self.transcription_model else "unhealthy",
            "model_type": self.model_type,
            "device": self.device,
            "components": {
                "transcription": "healthy" if self.transcription_model else "missing",
                "intent_classification": "healthy"
                if self.intent_classifier
                else "fallback",
                "processor": "healthy" if self.processor else "missing",
            },
            "statistics": self.stats,
        }


def create_huggingface_adapter(
    model_path: str, device: str = "auto"
) -> HuggingFaceModelAdapter:
    """
    Create and initialize a HuggingFace model adapter.

    Args:
        model_path: Path to the HuggingFace model
        device: Device to use ('auto', 'cpu', 'cuda')

    Returns:
        Initialized HuggingFace model adapter
    """
    return HuggingFaceModelAdapter(model_path, device)
