import sys
import tempfile
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from speech_recognizer import create_speech_recognizer

def record_audio(duration=5, sample_rate=16000):
    """Records audio from the microphone."""
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    return audio_data

def main():
    """Main function for live audio testing."""
    print("=" * 60)
    print("    LIVE TWI SPEECH RECOGNITION TEST")
    print("=" * 60)
    print("This script will record audio from your microphone and process it.")
    
    try:
        recognizer = create_speech_recognizer()
        sample_rate = recognizer.config.AUDIO["sample_rate"]
    except Exception as e:
        print(f"❌ Failed to initialize speech recognizer: {e}")
        return

    while True:
        try:
            input("Press Enter to start recording for 5 seconds (or Ctrl+C to exit)...")
            
            # Record audio
            audio_data = record_audio(duration=5, sample_rate=sample_rate)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                write(tmp_file.name, sample_rate, audio_data)
                temp_file_path = tmp_file.name
            
            print(f"🔄 Processing audio...")
            
            # Recognize speech
            result = recognizer.recognize(temp_file_path)
            
            # Print results
            if result.get("status") == "success":
                print("\n--- Recognition Result ---")
                print(f"📝 Transcription: '{result['transcription']['text']}'")
                print(f"🎯 Intent: {result['intent']['intent']}")
                print(f"📊 Confidence: {result['intent']['confidence']:.3f}")
                print("--------------------------\n")
            else:
                print(f"❌ Recognition failed: {result.get('error', 'Unknown error')}")

        except KeyboardInterrupt:
            print("\n👋 Exiting live test.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    main()
