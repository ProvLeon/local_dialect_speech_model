#!/usr/bin/env python3
"""
Simple Audio Format Converter for Akan Speech API
"""

import os
import tempfile
import logging
import subprocess
import signal
from pathlib import Path
from typing import Optional
import numpy as np
from contextlib import contextmanager

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

def convert_audio_to_wav(input_path: str, output_path: Optional[str] = None, timeout_seconds: int = 30) -> Optional[str]:
    """Convert any supported audio format to WAV with timeout handling."""
    if output_path is None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name

    # Try ffmpeg first for WebM and other complex formats
    if _try_ffmpeg_conversion(input_path, output_path):
        return output_path

    # Try pydub second
    if PYDUB_AVAILABLE:
        try:
            # Handle WebM specifically
            if input_path.lower().endswith('.webm'):
                audio = AudioSegment.from_file(input_path, format="webm")
            else:
                audio = AudioSegment.from_file(input_path)

            audio = audio.set_channels(1)  # mono
            audio = audio.set_frame_rate(16000)  # 16kHz
            audio.export(output_path, format="wav")
            logger.debug(f"Pydub conversion successful: {input_path}")
            return output_path
        except TimeoutError:
            logger.error(f"Pydub conversion timed out after {timeout_seconds}s: {input_path}")
        except Exception as e:
            logger.debug(f"Pydub conversion failed: {e}")

    # Try librosa fallback (avoid for WebM as it often fails)
    if LIBROSA_AVAILABLE and SOUNDFILE_AVAILABLE and not input_path.lower().endswith('.webm'):
        try:
            logger.debug(f"Trying librosa conversion for: {input_path}")
            with timeout_handler(timeout_seconds):
                audio, sr = librosa.load(input_path, sr=16000, mono=True)
                sf.write(output_path, audio, 16000)
            logger.debug(f"Librosa conversion successful: {input_path}")
            return output_path
        except TimeoutError:
            logger.error(f"Librosa conversion timed out after {timeout_seconds}s: {input_path}")
        except Exception as e:
            logger.debug(f"Librosa conversion failed: {e}")

    # If all methods fail, return the original path if it is WAV
    if input_path.lower().endswith(".wav"):
        logger.debug(f"Input is already WAV: {input_path}")
        return input_path

    logger.error(f"All conversion methods failed for: {input_path}")
    return None

@contextmanager
def timeout_handler(seconds):
    """Context manager for timeout handling"""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    # Set the signal handler and a timeout alarm
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Reset the alarm and handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def _try_ffmpeg_conversion(input_path: str, output_path: str, timeout_seconds: int = 30) -> bool:
    """Try converting using ffmpeg command line tool with timeout."""
    try:
        # Check if ffmpeg is available with timeout
        with timeout_handler(10):
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, timeout=5)
    except (subprocess.CalledProcessError, FileNotFoundError, TimeoutError):
        logger.debug("FFmpeg not available or timed out")
        return False

    try:
        cmd = [
            'ffmpeg', '-i', input_path,
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',  # overwrite
            output_path
        ]

        logger.debug(f"Starting FFmpeg conversion: {input_path}")
        with timeout_handler(timeout_seconds):
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)

        if result.returncode == 0:
            logger.debug(f"FFmpeg conversion successful: {input_path}")
            return True
        else:
            logger.debug(f"FFmpeg failed: {result.stderr}")
            return False
    except (subprocess.TimeoutExpired, TimeoutError) as e:
        logger.error(f"FFmpeg conversion timed out after {timeout_seconds}s: {input_path}")
        return False
    except Exception as e:
        logger.debug(f"FFmpeg conversion error: {e}")
        return False

def convert_bytes_to_wav(audio_bytes: bytes, filename: str, output_path: Optional[str] = None, timeout_seconds: int = 30) -> Optional[str]:
    """Convert audio bytes to WAV format with timeout handling."""
    input_suffix = Path(filename).suffix or ".wav"

    with tempfile.NamedTemporaryFile(suffix=input_suffix, delete=False) as tmp_input:
        tmp_input.write(audio_bytes)
        tmp_input_path = tmp_input.name

    try:
        logger.debug(f"Temporary input file created: {tmp_input_path}")
        result = convert_audio_to_wav(tmp_input_path, output_path, timeout_seconds)
        if result:
            logger.debug(f"Successfully converted bytes to WAV: {result}")
        else:
            logger.error(f"Failed to convert bytes to WAV for: {filename}")
        return result
    finally:
        if os.path.exists(tmp_input_path):
            os.unlink(tmp_input_path)
            logger.debug(f"Cleaned up temporary file: {tmp_input_path}")

def validate_audio_file(file_path: str, timeout_seconds: int = 10) -> bool:
    """Validate that the audio file is readable and has reasonable content with timeout."""
    try:
        if not os.path.exists(file_path):
            logger.debug(f"Audio file does not exist: {file_path}")
            return False

        if os.path.getsize(file_path) == 0:
            logger.debug(f"Audio file is empty: {file_path}")
            return False

        if SOUNDFILE_AVAILABLE:
            try:
                with timeout_handler(timeout_seconds):
                    data, sr = sf.read(file_path)
                if len(data) == 0:
                    logger.debug(f"Audio data is empty: {file_path}")
                    return False
                if not np.isfinite(data).all():
                    logger.debug(f"Audio contains non-finite values: {file_path}")
                    return False
                if np.all(data == 0):
                    logger.debug(f"Audio contains only silence: {file_path}")
                    return False
                logger.debug(f"Audio validation passed: {len(data)} samples at {sr} Hz")
                return True
            except TimeoutError:
                logger.error(f"Audio validation timed out after {timeout_seconds}s: {file_path}")
                return False
        elif LIBROSA_AVAILABLE:
            try:
                with timeout_handler(timeout_seconds):
                    audio, sr = librosa.load(file_path, sr=None)
                if len(audio) == 0 or not np.isfinite(audio).all() or np.all(audio == 0):
                    return False
                return True
            except TimeoutError:
                logger.error(f"Audio validation timed out after {timeout_seconds}s: {file_path}")
                return False
        else:
            # No validation library available, assume valid if file exists and has size
            return True
    except Exception as e:
        logger.debug(f"Audio validation failed for {file_path}: {e}")
        return False
