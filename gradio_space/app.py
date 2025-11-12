
import gradio as gr
import os
import sys
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))
sys.path.insert(0, str(current_dir))

from src.speech_recognizer import create_speech_recognizer

# --- Configuration ---
# It's recommended to use a specific model from the Hub.
# If the model is in this repository, you can use a local path.
# For this example, we'll assume the model needs to be downloaded.
HUGGINGFACE_REPO = os.environ.get("HUGGINGFACE_REPO", "TwiWhisperModel/TwiWhisperModel")

# Set environment variables for the speech recognizer
# This mimics the logic from main.py
if HUGGINGFACE_REPO:
    model_path = current_dir / "models" / "huggingface" / HUGGINGFACE_REPO.replace("/", "_")
    os.environ["HUGGINGFACE_MODEL_PATH"] = str(model_path)
    # The model type detection will happen inside the recognizer
    # os.environ["HUGGINGFACE_MODEL_TYPE"] = "single" # or "multi"

# --- Model Loading ---
# This can take a while, so it's good to have a clear message.
print("Initializing the speech recognizer...")
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

    recognizer = create_speech_recognizer()
    print("Speech recognizer initialized successfully.")
except Exception as e:
    print(f"Error initializing speech recognizer: {e}")
    # We'll create a dummy recognizer to allow the UI to load and show an error.
    recognizer = None

# --- Gradio Interface ---

def recognize_speech(audio):
    """
    The main function for the Gradio interface.
    Takes an audio file path and returns the results.
    """
    if recognizer is None:
        return "Error: Speech recognizer not initialized.", "Please check the logs.", "Error"

    if audio is None:
        return "No audio provided.", "", ""

    # Gradio provides the audio as a tuple (sample_rate, numpy_array)
    # We need to save it to a temporary file to pass to the recognizer.
    sample_rate, audio_data = audio
    
    import soundfile as sf
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        sf.write(tmpfile.name, audio_data, sample_rate)
        temp_audio_path = tmpfile.name

    try:
        result = recognizer.recognize(temp_audio_path)

        if result["status"] == "success":
            transcription = result["transcription"]["text"]
            intent = result["intent"]["intent"]
            confidence = result["intent"]["confidence"]
            return transcription, f"Intent: {intent} (Confidence: {confidence:.2f})", "Success"
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
    <p>Developed by an AI Assistant.</p>
</div>
"""

iface = gr.Interface(
    fn=recognize_speech,
    inputs=gr.Audio(type="numpy", label="Upload Audio File"),
    outputs=[
        gr.Textbox(label="Transcription"),
        gr.Textbox(label="Intent Classification"),
        gr.Textbox(label="Status"),
    ],
    title=title,
    description=description,
    article=article,
    examples=[
        ["addr_add_1.wav"], # Assuming this file exists in the repo
    ],
    allow_flagging="never",
)

# --- Launch the App ---

if __name__ == "__main__":
    print("Launching Gradio interface... The API will be available at the /api endpoint.")
    iface.launch(share=True)
