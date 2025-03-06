# tests/test_audio_processor.py
import pytest
import numpy as np
import os
import tempfile
from src.preprocessing.audio_processor import AudioProcessor

class TestAudioProcessor:
    @pytest.fixture
    def audio_processor(self):
        """Create an AudioProcessor instance for testing"""
        return AudioProcessor()

    @pytest.fixture
    def temp_audio_file(self):
        """Create a temporary audio file for testing"""
        import scipy.io.wavfile as wav

        # Create a simple sine wave
        sample_rate = 16000
        duration = 1  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav.write(tmp.name, sample_rate, (audio * 32767).astype(np.int16))
            tmp_path = tmp.name

        yield tmp_path

        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    def test_load_audio(self, audio_processor, temp_audio_file):
        """Test audio loading"""
        audio, sr = audio_processor.load_audio(temp_audio_file)

        assert sr == audio_processor.sample_rate
        assert isinstance(audio, np.ndarray)
        assert len(audio.shape) == 1  # Mono audio
        assert audio.dtype == np.float32

    def test_extract_mfcc(self, audio_processor, temp_audio_file):
        """Test MFCC extraction"""
        audio, _ = audio_processor.load_audio(temp_audio_file)
        features = audio_processor.extract_mfcc(audio)

        assert isinstance(features, np.ndarray)
        assert features.shape[0] == 3 * audio_processor.n_mfcc  # MFCCs + deltas + delta-deltas

    def test_preprocess(self, audio_processor, temp_audio_file):
        """Test full preprocessing pipeline"""
        features = audio_processor.preprocess(temp_audio_file)

        assert isinstance(features, np.ndarray)
        assert features.shape[0] == 3 * audio_processor.n_mfcc  # MFCCs + deltas + delta-deltas

        # Test with max_length
        max_length = 100
        features = audio_processor.preprocess(temp_audio_file, max_length=max_length)
        assert features.shape[1] == max_length
