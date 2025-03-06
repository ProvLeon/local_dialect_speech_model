import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import torch
from typing import Tuple, Optional, Union, Dict

class EnhancedAudioProcessor:
    def __init__(self, sample_rate=16000, n_mfcc=13, n_mels=40, n_fft=2048, hop_length=512,
                augment=False, feature_type='combined'):
        """
        Enhanced audio processor with improved feature extraction

        Args:
            sample_rate: Target sample rate
            n_mfcc: Number of MFCCs to extract
            n_mels: Number of Mel bands
            n_fft: FFT window size
            hop_length: Hop length for feature extraction
            augment: Whether to apply augmentation during preprocessing
            feature_type: Type of features to extract ('mfcc', 'melspec', 'combined')
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.augment = augment
        self.feature_type = feature_type

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load and normalize audio file"""
        # Handle different file formats
        if file_path.endswith('.mp3'):
            audio = AudioSegment.from_mp3(file_path)
            audio = audio.set_channels(1)  # Convert to mono
            audio = audio.set_frame_rate(self.sample_rate)
            samples = np.array(audio.get_array_of_samples())
            # Normalize to [-1, 1]
            audio_data = samples.astype(np.float32) / 32768.0
            return audio_data, self.sample_rate
        else:
            # Use librosa for wav and other formats
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            return audio, int(sr)

    def apply_augmentation(self, audio: np.ndarray) -> np.ndarray:
        """Apply random augmentations to the audio signal"""
        # Only apply augmentation if enabled
        if not self.augment:
            return audio

        # Randomly select augmentations
        augmentations = np.random.choice([
            'time_shift', 'pitch_shift', 'time_stretch', 'add_noise'
        ], size=np.random.randint(0, 3), replace=False)

        # Apply selected augmentations
        if 'time_shift' in augmentations:
            shift = np.random.randint(-self.sample_rate // 4, self.sample_rate // 4)
            audio = np.roll(audio, shift)
            if shift > 0:
                audio[:shift] = 0
            else:
                audio[shift:] = 0

        if 'pitch_shift' in augmentations:
            pitch_shift = np.random.uniform(-2, 2)
            audio = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=pitch_shift)

        if 'time_stretch' in augmentations:
            stretch_factor = np.random.uniform(0.8, 1.2)
            audio = librosa.effects.time_stretch(audio, rate=stretch_factor)
            # Ensure the same length
            if len(audio) > len(audio):
                audio = audio[:len(audio)]
            elif len(audio) < len(audio):
                audio = np.pad(audio, (0, len(audio) - len(audio)))

        if 'add_noise' in augmentations:
            noise_factor = np.random.uniform(0.001, 0.02)
            noise = np.random.randn(len(audio))
            audio = audio + noise_factor * noise

        return audio

    def extract_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract multiple feature types from audio signal

        Returns a dictionary of different feature representations
        """
        # Dictionary to store different feature types
        features = {}

        # Calculate MFCCs (baseline features)
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # Add deltas and delta-deltas (acceleration coefficients)
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

        # Stack MFCC features
        mfcc_features = np.vstack((mfccs, mfcc_delta, mfcc_delta2))
        features['mfcc'] = mfcc_features

        # Calculate Mel spectrogram features
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features['melspec'] = mel_spec_db

        # Calculate spectral contrast
        contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        features['contrast'] = contrast

        # Calculate chroma features (useful for tonal content)
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        features['chroma'] = chroma

        # Combined features for better performance
        if self.feature_type == 'combined':
            # Combine MFCC and selected additional features
            combined = np.vstack([
                features['mfcc'],
                features['contrast'],
                features['chroma']
            ])
            features['combined'] = combined

        return features

    def preprocess(self, file_path: str, max_length: Optional[int] = None, target_channels: Optional[int] = None) -> np.ndarray:
        """
        Complete preprocessing pipeline with optional channel adjustment

        Args:
            file_path: Path to audio file
            max_length: Maximum length of features (for padding/truncation)
            target_channels: Target number of feature channels (for model compatibility)

        Returns:
            Preprocessed features
        """
        # Load audio
        audio, _ = self.load_audio(file_path)

        # Apply optional augmentations
        audio = self.apply_augmentation(audio)

        # Apply noise reduction
        audio = self._reduce_noise(audio)

        # Extract multiple types of features
        feature_dict = self.extract_features(audio)

        # Select the desired feature type
        if self.feature_type in feature_dict:
            features = feature_dict[self.feature_type]
        else:
            # Default to combined features
            features = feature_dict.get('combined', feature_dict['mfcc'])

        # Normalize features
        features = self._normalize_features(features)

        # Pad or truncate if max_length is specified
        if max_length is not None:
            if features.shape[1] > max_length:
                features = features[:, :max_length]
            elif features.shape[1] < max_length:
                padding = np.zeros((features.shape[0], max_length - features.shape[1]))
                features = np.hstack((features, padding))

        # If target_channels is specified, adjust feature dimensions
        if target_channels is not None and features.shape[0] != target_channels:
            print(f"Adjusting feature channels from {features.shape[0]} to {target_channels}")
            if features.shape[0] > target_channels:
                # Truncate features
                features = features[:target_channels, :]
            else:
                # Pad with zeros
                padding = np.zeros((target_channels - features.shape[0], features.shape[1]))
                features = np.vstack((features, padding))

        return features
        # return features

    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """Enhanced noise reduction using spectral gating with refinements"""
        # Compute spectrogram
        D = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)

        # Compute magnitude
        mag = np.abs(D)

        # Improved noise floor estimation
        # Use the lowest 10% of energy frames as noise reference
        noise_floor = np.mean(np.sort(mag, axis=1)[:, :int(mag.shape[1]*0.1)], axis=1)
        noise_floor = noise_floor.reshape(-1, 1)

        # Apply refined soft mask with smoother transition
        mask = 1 - 1 / (1 + np.power(mag / (noise_floor * 2), 2))

        # Apply time-frequency smoothing to the mask
        mask = librosa.decompose.nn_filter(
            mask,
            aggregate=np.median,
            metric='cosine',
            width=1
        )

        # Apply mask to spectrogram
        D_denoised = D * mask

        # Convert back to time domain
        audio_denoised = librosa.istft(D_denoised, hop_length=self.hop_length)

        # Trim silent parts
        audio_denoised, _ = librosa.effects.trim(audio_denoised, top_db=20)

        return audio_denoised

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using mean and standard deviation"""
        # Normalize each feature independently
        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True) + 1e-5
        normalized = (features - mean) / std

        return normalized

    def save_processed_audio(self, audio: np.ndarray, output_path: str):
        """Save processed audio to file"""
        sf.write(output_path, audio, self.sample_rate)
