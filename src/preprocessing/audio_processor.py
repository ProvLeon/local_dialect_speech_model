# src/preprocessing/audio_processor.py
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
# import os
from typing import Tuple, Optional

class AudioProcessor:
    def __init__(self, sample_rate=16000, n_mfcc=13, n_fft=2048, hop_length=512):
        """
        Initialize audio processor with parameters

        Args:
            sample_rate: Target sample rate
            n_mfcc: Number of MFCCs to extract
            n_fft: FFT window size
            hop_length: Hop length for feature extraction
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and resample if necessary

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of audio data and sample rate
        """
        # Handle different file formats
        if file_path.endswith('.mp3'):
            audio = AudioSegment.from_mp3(file_path)
            audio = audio.set_channels(1)  # Convert to mono
            audio = audio.set_frame_rate(self.sample_rate)
            samples = np.array(audio.get_array_of_samples())
            return samples.astype(np.float32) / 32768.0, self.sample_rate
        else:
            # Use librosa for wav and other formats
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            return audio, int(sr)

    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio

        Args:
            audio: Audio signal

        Returns:
            MFCC features
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        # Add delta and delta-delta features
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # Stack features
        features = np.vstack((mfcc, mfcc_delta, mfcc_delta2))

        return features

    def preprocess(self, file_path: str, max_length: Optional[int] = None) -> np.ndarray:
        """
        Complete preprocessing pipeline

        Args:
            file_path: Path to audio file
            max_length: Maximum length of features (for padding/truncation)

        Returns:
            Preprocessed features
        """
        # Load audio
        audio, _ = self.load_audio(file_path)

        # Apply noise reduction
        audio = self._reduce_noise(audio)

        # Extract MFCCs
        features = self.extract_mfcc(audio)

        # Normalize features
        features = self._normalize_features(features)

        # Pad or truncate if max_length is specified
        if max_length is not None:
            if features.shape[1] > max_length:
                features = features[:, :max_length]
            elif features.shape[1] < max_length:
                padding = np.zeros((features.shape[0], max_length - features.shape[1]))
                features = np.hstack((features, padding))

        return features

    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Simple noise reduction using spectral gating

        Args:
            audio: Audio signal

        Returns:
            Noise-reduced audio
        """
        # Simple implementation - can be enhanced with more complex methods
        n_grad = 2

        # Compute spectrogram
        D = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)

        # Compute magnitude
        mag = np.abs(D)

        # Estimate noise floor
        noise_floor = np.mean(np.sort(mag, axis=1)[:, :int(mag.shape[1]*0.1)], axis=1)
        noise_floor = noise_floor.reshape(-1, 1)

        # Apply soft mask
        mask = 1 - 1 / (1 + np.power(mag / (noise_floor * n_grad), 2))

        # Apply mask to spectrogram
        D_denoised = D * mask

        # Convert back to time domain
        audio_denoised = librosa.istft(D_denoised, hop_length=self.hop_length)

        return audio_denoised

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using mean and std

        Args:
            features: Feature matrix

        Returns:
            Normalized features
        """
        # Normalize each feature independently
        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True) + 1e-5
        normalized = (features - mean) / std

        return normalized

    def save_processed_audio(self, audio: np.ndarray, output_path: str):
        """
        Save processed audio to file

        Args:
            audio: Audio data
            output_path: Path to save file
        """
        sf.write(output_path, audio, self.sample_rate)
