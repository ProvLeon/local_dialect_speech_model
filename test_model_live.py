import os
import torch
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import argparse
from src.models.speech_model import IntentClassifier
from src.preprocessing.enhanced_audio_processor import EnhancedAudioProcessor

def record_audio(duration=3, sample_rate=16000):
    """Record audio from microphone"""
    print("Recording in 3...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        sd.sleep(1000)  # Sleep for 1 second

    print("Speak now!")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Recording complete!")

    # Normalize
    audio_data = audio_data.flatten()
    audio_data = audio_data / np.max(np.abs(audio_data))

    return audio_data, sample_rate

def save_audio(audio_data, sample_rate, output_file):
    """Save audio data to a file"""
    sf.write(output_file, audio_data, sample_rate)
    print(f"Audio saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Test the speech-to-intent model with live audio")
    parser.add_argument("--model", type=str, default="data/models_improved/best_model.pt", help="Path to model file")
    parser.add_argument("--label-map", type=str, default="data/processed_augmented/label_map.npy", help="Path to label map")
    parser.add_argument("--duration", type=int, default=3, help="Recording duration in seconds")
    parser.add_argument("--model-type", type=str, default="improved", choices=["standard", "improved"], help="Type of model to use")
    parser.add_argument("--save-audio", action="store_true", help="Save recorded audio")

    args = parser.parse_args()

    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        return

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use the enhanced audio processor
    processor = EnhancedAudioProcessor(feature_type="combined")

    # Initialize classifier
    classifier = IntentClassifier(
        model_path=args.model,
        device=device,
        processor=processor,
        label_map_path=args.label_map,
        model_type=args.model_type
    )

    print("Model loaded successfully!")
    print("\nAvailable intents:")
    for intent in classifier.label_map.keys():
        print(f"- {intent}")

    while True:
        try:
            input("\nPress Enter to record audio (or Ctrl+C to exit)...")

            # Record audio
            audio_data, sample_rate = record_audio(duration=args.duration)

            # Save audio to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=not args.save_audio) as tmp:
                save_audio(audio_data, sample_rate, tmp.name)

                # Classify
                intent, confidence = classifier.classify(tmp.name)

                print(f"\nPredicted intent: {intent}")
                print(f"Confidence: {confidence:.4f}")

                if args.save_audio:
                    save_path = f"test_recording_{int(time.time())}.wav"
                    os.rename(tmp.name, save_path)
                    print(f"Audio saved to {save_path}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
