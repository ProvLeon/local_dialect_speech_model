import os
import torch
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import argparse
import time
from src.models.speech_model import IntentOnlyModel
from src.preprocessing.enhanced_audio_processor import EnhancedAudioProcessor
from src.utils.model_utils import load_label_map, get_model_input_dim

def record_audio(duration=3, sample_rate=16000):
    """Record audio from microphone"""
    print("Recording in 3...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    print("Speak now!")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Recording complete!")

    # Normalize
    audio_data = audio_data.flatten()
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))

    return audio_data, sample_rate

def save_audio(audio_data, sample_rate, output_file):
    """Save audio data to a file"""
    sf.write(output_file, audio_data, sample_rate)
    print(f"Audio saved to {output_file}")

def preprocess_audio(audio_path, processor, expected_input_dim):
    features = processor.preprocess(audio_path)
    if features.shape[0] != expected_input_dim:
        if features.shape[0] > expected_input_dim:
            features = features[:expected_input_dim, :]
        else:
            padding = np.zeros((expected_input_dim - features.shape[0], features.shape[1]))
            features = np.vstack((features, padding))
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    return features_tensor

def classify_audio(model, audio_tensor, label_map, device):
    audio_tensor = audio_tensor.to(device)
    with torch.no_grad():
        outputs = model(audio_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = probabilities.max(1)
    idx_to_label = {v: k for k, v in label_map.items()}
    predicted_intent = idx_to_label[predicted_idx.item()]
    confidence_value = confidence.item()
    return predicted_intent, confidence_value, probabilities.cpu().numpy()[0]

def main():
    parser = argparse.ArgumentParser(description="Test the speech-to-intent model with live audio")
    parser.add_argument("--model", type=str, default="data/models/47_class_results/balanced_20250828_074437/best_model.pt", help="Path to model file")
    parser.add_argument("--label-map", type=str, default="data/processed/label_map.json", help="Path to label map")
    parser.add_argument("--duration", type=int, default=3, help="Recording duration in seconds")
    parser.add_argument("--save-audio", action="store_true", help="Save recorded audio")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor = EnhancedAudioProcessor(feature_type="combined")
    label_map = load_label_map(args.label_map)
    input_dim = 39

    model = IntentOnlyModel(
        input_dim=input_dim,
        hidden_dim=128,
        num_classes=len(label_map)
    )
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    print("Model loaded successfully!")
    print("\nAvailable intents:")
    for intent in sorted(label_map.keys()):
        print(f"- {intent}")

    while True:
        try:
            input("\nPress Enter to record audio (or Ctrl+C to exit)...")
            audio_data, sample_rate = record_audio(duration=args.duration)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=not args.save_audio) as tmp:
                save_audio(audio_data, sample_rate, tmp.name)
                features_tensor = preprocess_audio(tmp.name, processor, input_dim)
                intent, confidence, _ = classify_audio(model, features_tensor, label_map, device)
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
