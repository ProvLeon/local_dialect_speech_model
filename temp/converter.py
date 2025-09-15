#!/usr/bin/env python3
"""
Robust Audio Format Converter for Akan Speech API

This module provides comprehensive audio format conversion capabilities
to ensure all incoming audio is properly converted to WAV format compatible
with the backend processing pipeline.
"""

import os
import tempfile
import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    sf = None

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    AudioSegment = None

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None

logger = logging.getLogger(__name__)


class AudioFormatConverter:
    """
    Robust audio format converter that handles multiple input formats
    and ensures consistent WAV output for backend processing.
    """

    def __init__(self,
                 target_sample_rate: int = 16000,
                 target_channels: int = 1,
                 target_dtype: str = 'float32'):
        """
        Initialize converter with target audio specifications.

        Args:
            target_sample_rate: Target sample rate in Hz
            target_channels: Target number of channels (1 for mono)
            target_dtype: Target data type ('float32', 'int16', etc.)
        """
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self.target_dtype = target_dtype

        # Check available libraries
        self._check_dependencies()

    def _check_dependencies(self):
        """Check which audio processing libraries are available."""
        available = []
        if SOUNDFILE_AVAILABLE:
            available.append("soundfile")
        if PYDUB_AVAILABLE:
            available.append("pydub")
        if LIBROSA_AVAILABLE:
            available.append("librosa")

        logger.info(f"Available audio libraries: {available}")

        if not available:
            logger.warning("No audio processing libraries available!")

    def detect_format(self, file_path: str) -> Optional[str]:
        """
        Detect audio format from file extension and content.

        Args:
            file_path: Path to audio file

        Returns:
            Detected format string or None
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        # Map extensions to formats
        format_map = {
            '.wav': 'wav',
            '.mp3': 'mp3',
            '.webm': 'webm',
            '.ogg': 'ogg',
            '.m4a': 'm4a',
            '.aac': 'aac',
            '.flac': 'flac'
        }

        detected_format = format_map.get(extension)
        logger.debug(f"Detected format from extension: {detected_format}")

        return detected_format

    def convert_with_pydub(self, input_path: str, output_path: str) -> bool:
        """
        Convert audio using pydub library.

        Args:
            input_path: Input audio file path
            output_path: Output WAV file path

        Returns:
            Success status
        """
        if not PYDUB_AVAILABLE:
            return False

        try:
            # Detect format
            detected_format = self.detect_format(input_path)

            # Load audio with appropriate method
            if detected_format == 'mp3':
                audio = AudioSegment.from_mp3(input_path)
            elif detected_format == 'webm':
                audio = AudioSegment.from_file(input_path, format='webm')
            elif detected_format == 'ogg':
                audio = AudioSegment.from_ogg(input_path)
            elif detected_format == 'm4a':
                audio = AudioSegment.from_file(input_path, format='m4a')
            else:
                # Try generic loader
                audio = AudioSegment.from_file(input_path)

            # Convert to target specifications
            audio = audio.set_channels(self.target_channels)
            audio = audio.set_frame_rate(self.target_sample_rate)

            # Export as WAV
            audio.export(output_path, format='wav')

            logger.info(f"Converted {input_path} to {output_path} using pydub")
            return True

        except Exception as e:
            logger.warning(f"Pydub conversion failed: {e}")
            return False

    def convert_with_ffmpeg(self, input_path: str, output_path: str) -> bool:
        """
        Convert audio using ffmpeg command line tool.

        Args:
            input_path: Input audio file path
            output_path: Output WAV file path

        Returns:
            Success status
        """
        try:
            # Check if ffmpeg is available
            subprocess.run(['ffmpeg', '-version'],
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.debug("FFmpeg not available")
            return False

        try:
            # Build ffmpeg command
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', str(self.target_sample_rate),  # Sample rate
                '-ac', str(self.target_channels),  # Channels
                '-y',  # Overwrite output
                output_path
            ]

            # Run conversion
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"Converted {input_path} to {output_path} using ffmpeg")
                return True
            else:
                logger.warning(f"FFmpeg conversion failed: {result.stderr}")
                return False

        except Exception as e:
            logger.warning(f"FFmpeg conversion error: {e}")
            return False

    def convert_with_librosa(self, input_path: str, output_path: str) -> bool:
        """
        Convert audio using librosa library.

        Args:
            input_path: Input audio file path
            output_path: Output WAV file path

        Returns:
            Success status
        """
        if not LIBROSA_AVAILABLE or not SOUNDFILE_AVAILABLE:
            return False

        try:
            # Load audio with librosa
            audio, sr = librosa.load(input_path,
                                   sr=self.target_sample_rate,
                                   mono=(self.target_channels == 1))

            # Save as WAV using soundfile
            sf.write(output_path, audio, self.target_sample_rate)

            logger.info(f"Converted {input_path} to {output_path} using librosa")
            return True

        except Exception as e:
            logger.warning(f"Librosa conversion failed: {e}")
            return False

    def validate_audio(self, file_path: str) -> bool:
        """
        Validate that the converted audio file is readable and has reasonable content.

        Args:
            file_path: Path to audio file to validate

        Returns:
            True if audio is valid
        """
        try:
            if SOUNDFILE_AVAILABLE:
                # Try reading with soundfile
                data, sr = sf.read(file_path)

                # Check basic properties
                if len(data) == 0:
                    logger.warning("Audio file is empty")
                    return False

                if np.all(data == 0):
                    logger.warning("Audio contains only silence")
                    return False

                if not np.isfinite(data).all():
                    logger.warning("Audio contains non-finite values")
                    return False

                logger.debug(f"Audio validation passed: {len(data)} samples at {sr} Hz")
                return True

            elif LIBROSA_AVAILABLE:
                # Try reading with librosa
                audio, sr = librosa.load(file_path, sr=None)

                if len(audio) == 0 or np.all(audio == 0) or not np.isfinite(audio).all():
                    return False

                return True

            else:
                # No validation library available, assume valid
                logger.warning("No audio validation library available")
                return True

        except Exception as e:
            logger.warning(f"Audio validation failed: {e}")
            return False

    def convert_to_wav(self,
                      input_path: str,
                      output_path: Optional[str] = None) -> Optional[str]:
        """
        Convert any supported audio format to WAV.

        Args:
            input_path: Path to input audio file
            output_path: Path for output WAV file (optional)

        Returns:
            Path to converted WAV file or None if conversion failed
        """
        if output_path is None:
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                output_path = tmp.name

        # Try conversion methods in order of preference
        conversion_methods = [
            ('pydub', self.convert_with_pydub),
            ('ffmpeg', self.convert_with_ffmpeg),
            ('librosa', self.convert_with_librosa)
        ]

        for method_name, method in conversion_methods:
            logger.debug(f"Trying conversion with {method_name}")

            try:
                if method(input_path, output_path):
                    # Validate the converted file
                    if self.validate_audio(output_path):
                        logger.info(f"Successfully converted using {method_name}")
                        return output_path
                    else:
                        logger.warning(f"Conversion with {method_name} produced invalid audio")
                        # Clean up invalid file
                        if os.path.exists(output_path):
                            os.unlink(output_path)
            except Exception as e:
                logger.warning(f"Conversion method {method_name} failed: {e}")
                continue

        logger.error(f"All conversion methods failed for {input_path}")
        return None

    def convert_bytes_to_wav(self,
                           audio_bytes: bytes,
                           filename: str,
                           output_path: Optional[str] = None) -> Optional[str]:
        """
        Convert audio bytes to WAV format.

        Args:
            audio_bytes: Raw audio data as bytes
            filename: Original filename (used for format detection)
            output_path: Path for output WAV file (optional)

        Returns:
            Path to converted WAV file or None if conversion failed
        """
        # Create temporary input file
        input_suffix = Path(filename).suffix or '.wav'

        with tempfile.NamedTemporaryFile(suffix=input_suffix, delete=False) as tmp_input:
            tmp_input.write(audio_bytes)
            tmp_input_path = tmp_input.name

        try:
            # Convert the temporary file
            result = self.convert_to_wav(tmp_input_path, output_path)
            return result
        finally:
            # Clean up temporary input file
            if os.path.exists(tmp_input_path):
                os.unlink(tmp_input_path)


# Global converter instance
_converter = None

def get_converter() -> AudioFormatConverter:
    """Get global converter instance."""
    global _converter
    if _converter is None:
        _converter = AudioFormatConverter()
    return _converter

def convert_audio_to_wav(input_path: str, output_path: Optional[str] = None) -> Optional[str]:
    """
    Convenience function to convert audio to WAV format.

    Args:
        input_path: Path to input audio file
        output_path: Path for output WAV file (optional)

    Returns:
        Path to converted WAV file or None if conversion failed
    """
    converter = get_converter()
    return converter.convert_to_wav(input_path, output_path)

def convert_bytes_to_wav(audio_bytes: bytes,
                        filename: str,
                        output_path: Optional[str] = None) -> Optional[str]:
    """
    Convenience function to convert audio bytes to WAV format.

    Args:
        audio_bytes: Raw audio data as bytes
        filename: Original filename (used for format detection)
        output_path: Path for output WAV file (optional)

    Returns:
        Path to converted WAV file or None if conversion failed
    """
    converter = get_converter()
    return converter.convert_bytes_to_wav(audio_bytes, filename, output_path)

def validate_audio_file(file_path: str) -> bool:
    """
    Convenience function to validate audio file.

    Args:
        file_path: Path to audio file

    Returns:
        True if audio is valid
    """
    converter = get_converter()
    return converter.validate_audio(file_path)
