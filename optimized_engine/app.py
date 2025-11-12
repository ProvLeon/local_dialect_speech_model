import os
import sys
from pathlib import Path

from src.speech_recognizer import create_speech_recognizer

# Check PyTorch availability first
try:
    import torch

    print(f"✅ PyTorch {torch.__version__} is available")
    print(f"🔧 PyTorch CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"❌ PyTorch not found: {e}")
    print("🔄 Attempting to install PyTorch...")
    import subprocess

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "torch", "torchaudio", "--upgrade"]
        )
        import torch

        print(f"✅ PyTorch {torch.__version__} installed successfully")
    except Exception as install_error:
        print(f"❌ Failed to install PyTorch: {install_error}")

import gradio as gr

# Add src to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))
sys.path.insert(0, str(current_dir))


# --- Configuration ---
# It's recommended to use a specific model from the Hub.
# If the model is in this repository, you can use a local path.
# For this example, we'll assume the model needs to be downloaded.
HUGGINGFACE_REPO = os.environ.get("HUGGINGFACE_REPO", "TwiWhisperModel/TwiWhisperModel")

# Set environment variables for the speech recognizer
# This mimics the logic from main.py
if HUGGINGFACE_REPO:
    model_path = (
        current_dir / "models" / "huggingface" / HUGGINGFACE_REPO.replace("/", "_")
    )
    os.environ["HUGGINGFACE_MODEL_PATH"] = str(model_path)
    # The model type detection will happen inside the recognizer
    # os.environ["HUGGINGFACE_MODEL_TYPE"] = "single" # or "multi"

# --- Model Loading ---
# This can take a while, so it's good to have a clear message.
print("Initializing the speech recognizer...")
print(f"🔍 Python executable: {sys.executable}")
print(f"🔍 Python path: {sys.path}")

# Check critical imports
try:
    import transformers

    print(f"✅ Transformers {transformers.__version__} available")
except ImportError as e:
    print(f"❌ Transformers not available: {e}")

try:
    # This is where the model is downloaded and loaded.
    # In a Hugging Face Space, this happens during startup.
    # The create_speech_recognizer function should handle the download if the model is not found.
    # We need to ensure the logic from main.py's download_huggingface_model is somehow triggered.
    # For simplicity, we'll rely on the existing logic inside OptimizedSpeechRecognizer.
    # A better approach would be to refactor the download logic to be more reusable.

    # A simplified version of the manager from main.py
    from main import OptimizedEngineManager

    manager = OptimizedEngineManager(huggingface_repo=HUGGINGFACE_REPO)
    if not manager.download_huggingface_model():
        raise RuntimeError("Failed to download HuggingFace model.")

    # Copy intent labels from trained models or use fallback
    model_path = os.environ.get("HUGGINGFACE_MODEL_PATH")
    if model_path:
        intent_labels_path = Path(model_path) / "intent_labels.json"

        # Check for intent labels from various sources
        intent_sources = [
            # Local intent classifier model
            current_dir / "models" / "intent_classifier" / "intent_labels.json",
            # Local fallback file
            current_dir / "intent_labels.json",
            # Other trained models
            current_dir / "models" / "whisper_twi_multitask" / "intent_labels.json",
        ]

        # Find first available intent labels file
        source_file = None
        for source in intent_sources:
            if source.exists():
                source_file = source
                print(f"✅ Found intent labels at: {source}")
                break

        # Copy if found and not already in model directory
        if source_file and not intent_labels_path.exists():
            try:
                import shutil

                shutil.copy2(source_file, intent_labels_path)
                print(f"✅ Copied intent labels to {intent_labels_path}")
            except Exception as e:
                print(f"⚠️ Could not copy intent labels: {e}")
        elif not source_file:
            # Create minimal intent labels as fallback
            print("⚠️ No trained intent labels found, creating minimal fallback")
            minimal_labels = {
                "label_to_id": {"unknown": 0, "general": 1},
                "id_to_label": {"0": "unknown", "1": "general"},
                "num_labels": 2,
            }
            try:
                import json

                with open(intent_labels_path, "w") as f:
                    json.dump(minimal_labels, f, indent=2)
                print(f"✅ Created fallback intent labels at {intent_labels_path}")
            except Exception as e:
                print(f"⚠️ Could not create fallback intent labels: {e}")

    recognizer = create_speech_recognizer()
    print("Speech recognizer initialized successfully.")
except Exception as e:
    print(f"Error initializing speech recognizer: {e}")
    import traceback

    traceback.print_exc()
    # We'll create a dummy recognizer to allow the UI to load and show an error.
    recognizer = None

# --- Gradio Interface ---


def recognize_speech(audio):
    """
    The main function for the Gradio interface.
    Takes an audio file path and returns the results.
    """
    if recognizer is None:
        return (
            "Error: Speech recognizer not initialized.",
            "Please check the logs.",
            "Error",
        )

    if audio is None:
        return "No audio provided.", "", ""

    # Gradio provides the audio as a tuple (sample_rate, numpy_array)
    # We need to save it to a temporary file to pass to the recognizer.
    sample_rate, audio_data = audio

    import tempfile

    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        sf.write(tmpfile.name, audio_data, sample_rate)
        temp_audio_path = tmpfile.name

    try:
        result = recognizer.recognize(temp_audio_path)

        if result["status"] == "success":
            transcription = result["transcription"]["text"]
            intent = result["intent"]["intent"]
            confidence = result["intent"]["confidence"]
            return (
                transcription,
                f"Intent: {intent} (Confidence: {confidence:.2f})",
                "Success",
            )
        else:
            return "Recognition failed.", result.get("error", "Unknown error"), "Failed"
    except Exception as e:
        return "An error occurred during recognition.", str(e), "Error"
    finally:
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)


# --- UI Definition ---

title = "Optimized Twi Speech Recognition Engine"
description = """
This is a demo of the optimized Twi speech recognition engine.
Upload an audio file (in Twi) and the model will transcribe it and classify the intent.
The model is based on OpenAI's Whisper and a custom intent classifier.
"""

article = """
<div style='text-align: center;'>
    <p>Model based on <a href='https://huggingface.co/openai/whisper-large-v3' target='_blank'>Whisper</a> and fine-tuned for Twi.</p>
    <p>Developed by Orlixis LTD.</p>
</div>
"""

iface = gr.Interface(
    fn=recognize_speech,
    inputs=gr.Audio(
        sources=["upload", "microphone"],
        type="numpy",
        label="Upload Audio File or Record",
    ),
    outputs=[
        gr.Textbox(label="Transcription"),
        gr.Textbox(label="Intent Classification"),
        gr.Textbox(label="Status"),
    ],
    title=title,
    description=description,
    article=article,
    allow_flagging="never",
)

# --- Launch the App ---

if __name__ == "__main__":
    print(
        "Launching Gradio interface... The API will be available at the /api endpoint."
    )

    # Check if running on HuggingFace Spaces
    is_hf_space = os.getenv("SPACE_ID") is not None
    space_id = os.getenv("SPACE_ID", "unknown")

    # Debug environment info
    print("🔍 Environment Detection:")
    print(f"   - SPACE_ID: {space_id}")
    print(f"   - Is HF Space: {is_hf_space}")
    print(f"   - Model Repo: {HUGGINGFACE_REPO}")

    if is_hf_space:
        print(f"🚀 Running on HuggingFace Spaces: {space_id}")
        print(f"🌐 Public URL: https://huggingface.co/spaces/{space_id}")
        iface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,  # Not needed on HF Spaces - already public
        )
    else:
        print("💻 Running locally - creating public share link for development")
        iface.launch(
            share=True,  # Create public ngrok link for local development
            server_name="0.0.0.0",
            server_port=7860,
        )
