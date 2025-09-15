# src/preprocessing/audio_processor.py
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import random
import warnings
from typing import Tuple, Optional, Callable, Dict, Any

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    warnings.warn("Torch not available: waveform-domain augmentations disabled.")

class AudioProcessor:
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        enable_deltas: bool = True,
        enable_audio_augment: bool = False,
        enable_spec_augment: bool = False,
        spec_time_masks: int = 2,
        spec_time_width: int = 50,
        spec_freq_masks: int = 2,
        spec_freq_width: int = 6,
        augment_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize audio processor with parameters & optional augmentation.

        Args:
            sample_rate: Target sample rate
            n_mfcc: Number of MFCCs to extract
            n_fft: FFT window size
            hop_length: Hop length for feature extraction
            enable_deltas: If True include delta + delta-delta (triples channels)
            enable_audio_augment: Apply waveform-domain augmentation (time stretch, noise, gain)
            enable_spec_augment: Apply SpecAugment masking on feature matrix
            spec_time_masks: Number of time masks
            spec_time_width: Max width in frames for each time mask
            spec_freq_masks: Number of frequency masks
            spec_freq_width: Max width in bins for each frequency mask
            augment_config: Dict overriding default augmentation probabilities / ranges
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.enable_deltas = enable_deltas
        self.enable_audio_augment = enable_audio_augment
        self.enable_spec_augment = enable_spec_augment
        self.spec_time_masks = spec_time_masks
        self.spec_time_width = spec_time_width
        self.spec_freq_masks = spec_freq_masks
        self.spec_freq_width = spec_freq_width

        # Default augmentation config
        default_aug = {
            "p_time_stretch": 0.35,
            "time_stretch_min": 0.9,
            "time_stretch_max": 1.1,
            "p_pitch_shift": 0.30,
            "pitch_semitones": 2,
            "p_noise": 0.50,
            "noise_min": 0.003,
            "noise_max": 0.02,
            "p_gain": 0.30,
            "gain_min": 0.7,
            "gain_max": 1.4
        }
        if augment_config:
            default_aug.update(augment_config)
        self.augment_config = default_aug

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
        Extract MFCC (with optional delta / delta-delta) features.
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        if not self.enable_deltas:
            return mfcc
        # Add delta and delta-delta features
        try:
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            features = np.vstack((mfcc, mfcc_delta, mfcc_delta2))
        except Exception:
            # Fallback silently if delta computation fails
            features = mfcc
        return features

    def preprocess(self, file_path: str, max_length: Optional[int] = None) -> np.ndarray:
        """
        Complete preprocessing pipeline:
          1. Load
          2. Optional waveform augmentation
          3. Noise reduction
          4. MFCC (+ optional deltas)
          5. Normalize
          6. Optional SpecAugment (masking)
          7. Pad / truncate
        """
        # Load audio
        audio, _ = self.load_audio(file_path)

        # Waveform augment (only if enabled)
        if self.enable_audio_augment:
            audio = self._augment_waveform(audio)

        # Apply noise reduction (lightweight)
        audio = self._reduce_noise(audio)

        # Extract MFCC-based features
        features = self.extract_mfcc(audio)

        # Normalize features (channel-wise)
        features = self._normalize_features(features)

        # SpecAugment (mask on time / freq axes) - works in-place on numpy
        if self.enable_spec_augment:
            features = self._spec_augment(features)

        # Pad or truncate
        if max_length is not None:
            T = features.shape[1]
            if T > max_length:
                features = features[:, :max_length]
            elif T < max_length:
                padding = np.zeros((features.shape[0], max_length - T))
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
        Per-channel standardization.
        """
        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True) + 1e-5
        return (features - mean) / std

    def save_processed_audio(self, audio: np.ndarray, output_path: str):
        """
        Save processed audio to file
        """
        sf.write(output_path, audio, self.sample_rate)

    # ---------------- Internal Augmentation Helpers ---------------- #

    def _augment_waveform(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply simple waveform-domain augmentations (time stretch, pitch shift, noise, gain).
        Executed with probabilities defined in self.augment_config.
        """
        cfg = self.augment_config
        # Time stretch
        if random.random() < cfg["p_time_stretch"]:
            rate = random.uniform(cfg["time_stretch_min"], cfg["time_stretch_max"])
            try:
                audio = librosa.effects.time_stretch(audio, rate)
            except Exception:
                pass
        # Pitch shift
        if random.random() < cfg["p_pitch_shift"]:
            steps = random.uniform(-cfg["pitch_semitones"], cfg["pitch_semitones"])
            try:
                audio = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=steps)
            except Exception:
                pass
        # Add noise
        if random.random() < cfg["p_noise"]:
            noise_amp = random.uniform(cfg["noise_min"], cfg["noise_max"])
            noise = np.random.randn(len(audio)) * noise_amp
            audio = audio + noise
        # Gain
        if random.random() < cfg["p_gain"]:
            gain = random.uniform(cfg["gain_min"], cfg["gain_max"])
            audio = audio * gain
        # Clip to safe range
        if audio.size:
            mx = np.max(np.abs(audio))
            if mx > 1.0:
                audio = audio / (mx + 1e-6)
        return audio

    def _spec_augment(self, feats: np.ndarray) -> np.ndarray:
        """
        Apply SpecAugment-style masking (in-place safe copy).
        feats: (C, T)
        """
        C, T = feats.shape
        out = feats.copy()
        # Time masks
        for _ in range(self.spec_time_masks):
            w = random.randint(5, self.spec_time_width)
            if w >= T:
                continue
            t0 = random.randint(0, T - w)
            out[:, t0:t0 + w] = 0
        # Frequency masks
        for _ in range(self.spec_freq_masks):
            w = random.randint(2, self.spec_freq_width)
            if w >= C:
                continue
            f0 = random.randint(0, C - w)
            out[f0:f0 + w, :] = 0
        return out
