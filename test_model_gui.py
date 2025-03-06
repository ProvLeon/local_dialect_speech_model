# test_model_gui.py
import os
import torch
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import time
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from src.models.speech_model import IntentClassifier
from src.preprocessing.enhanced_audio_processor import EnhancedAudioProcessor
import json

class SpeechIntentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Twi Speech Intent Recognizer")
        self.root.geometry("500x400")

        # Load model
        self.load_model()

        # Main frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="Twi Speech Intent Recognizer",
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        # Duration selector
        duration_frame = ttk.Frame(main_frame)
        duration_frame.pack(pady=10)

        ttk.Label(duration_frame, text="Recording Duration:").pack(side=tk.LEFT)
        self.duration_var = tk.IntVar(value=3)
        ttk.Spinbox(duration_frame, from_=1, to=10, textvariable=self.duration_var,
                  width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(duration_frame, text="seconds").pack(side=tk.LEFT)

        # Record button
        self.record_btn = ttk.Button(main_frame, text="Record Audio",
                                 command=self.start_recording)
        self.record_btn.pack(pady=10)

        # Status message
        self.status_var = tk.StringVar(value="Ready to record")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.pack(pady=5)

        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Recognition Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Intent result
        intent_frame = ttk.Frame(results_frame)
        intent_frame.pack(fill=tk.X, pady=5)
        ttk.Label(intent_frame, text="Intent:", width=10).pack(side=tk.LEFT)
        self.intent_var = tk.StringVar()
        ttk.Label(intent_frame, textvariable=self.intent_var, font=("Arial", 12, "bold")).pack(side=tk.LEFT)

        # Confidence result
        conf_frame = ttk.Frame(results_frame)
        conf_frame.pack(fill=tk.X, pady=5)
        ttk.Label(conf_frame, text="Confidence:", width=10).pack(side=tk.LEFT)
        self.conf_var = tk.StringVar()
        ttk.Label(conf_frame, textvariable=self.conf_var).pack(side=tk.LEFT)

        # Available intents
        intents_frame = ttk.LabelFrame(main_frame, text="Available Intents")
        intents_frame.pack(fill=tk.BOTH, pady=10)

        self.intents_text = tk.Text(intents_frame, height=5, wrap=tk.WORD)
        self.intents_text.pack(fill=tk.BOTH)

        if hasattr(self, 'classifier') and self.classifier and hasattr(self.classifier, 'label_map'):
            intents_str = ", ".join(self.classifier.label_map.keys())
            self.intents_text.insert(tk.END, intents_str)

    def load_model(self):
        try:
            model_path = os.environ.get("MODEL_PATH", "data/models_improved/best_model.pt")
            label_map_path = os.environ.get("LABEL_MAP_PATH", "data/processed_augmented/label_map.npy")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            processor = EnhancedAudioProcessor(feature_type="combined")

            self.classifier = IntentClassifier(
                model_path=model_path,
                device=device,
                processor=processor,
                label_map_path=label_map_path,
                model_type="improved"
            )
            print("Model loaded successfully!")

        except Exception as e:
            print(f"Error loading model: {e}")
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.classifier = None

    def start_recording(self):
        # Disable button during recording
        self.record_btn.config(state=tk.DISABLED)

        # Start recording in a separate thread to avoid UI freeze
        threading.Thread(target=self.record_and_process, daemon=True).start()

    def record_and_process(self):
        try:
            duration = self.duration_var.get()
            self.status_var.set("Preparing to record in 3...")
            self.root.update()
            time.sleep(1)

            self.status_var.set("Preparing to record in 2...")
            self.root.update()
            time.sleep(1)

            self.status_var.set("Preparing to record in 1...")
            self.root.update()
            time.sleep(1)

            self.status_var.set("Recording now... Speak!")
            self.root.update()

            # Record audio
            sample_rate = 16000
            audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
            sd.wait()

            # Process audio
            self.status_var.set("Processing audio...")
            self.root.update()

            # Save to temp file
            audio_data = audio_data.flatten()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, audio_data, sample_rate)

                # Classify
                if self.classifier:
                    intent, confidence = self.classifier.classify(tmp.name)

                    # Update results
                    self.intent_var.set(intent)
                    self.conf_var.set(f"{confidence:.4f}")
                    self.status_var.set("Recognition complete!")
                else:
                    self.status_var.set("Error: Model not loaded")

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            print(f"Error: {e}")

        # Re-enable button
        self.record_btn.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechIntentApp(root)
    root.mainloop()
