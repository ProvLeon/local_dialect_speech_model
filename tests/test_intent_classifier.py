# tests/test_intent_classifier.py
import pytest
import torch
import numpy as np
import os
import tempfile
# import json
from src.models.speech_model import EnhancedTwiSpeechModel as TwiSpeechModel, IntentClassifier
# from src.preprocessing.audio_processor import AudioProcessor

class TestIntentClassifier:
    @pytest.fixture
    def mock_model_path(self):
        """Create a mock model for testing"""
        # Create a temporary model file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            # Create a small test model
            model = TwiSpeechModel(input_dim=39, hidden_dim=64, num_classes=5)

            # Create checkpoint dictionary
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': {},  # Empty optimizer state
                'history': {}  # Empty history
            }

            # Save model
            torch.save(checkpoint, tmp.name)
            tmp_path = tmp.name

        yield tmp_path

        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    @pytest.fixture
    def mock_label_map(self):
        """Create a mock label map for testing"""
        # Create a temporary label map file
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            label_map = {
                "add_to_cart": 0,
                "checkout": 1,
                "search": 2,
                "purchase": 3,
                "cancel": 4
            }
            np.save(tmp.name, label_map)
            tmp_path = tmp.name

        yield tmp_path

        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    def test_classifier_initialization(self, mock_model_path, mock_label_map):
        """Test classifier initialization"""
        device = torch.device("cpu")
        classifier = IntentClassifier(
            model_path=mock_model_path,
            device=device,
            label_map_path=mock_label_map
        )

        assert classifier.model is not None
        assert classifier.label_map is not None
        assert len(classifier.label_map) == 5
        assert classifier.idx_to_label is not None
        assert classifier.idx_to_label[0] == "add_to_cart"
