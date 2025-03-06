# src/utils/dataset_builder.py
import os
import pandas as pd
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
from tqdm import tqdm
import time
from ..preprocessing.audio_processor import AudioProcessor

class TwiDatasetBuilder:
    def __init__(self, output_dir="data/raw", processor=None):
        """
        Initialize dataset builder

        Args:
            output_dir: Directory to store recordings
            processor: AudioProcessor instance for preprocessing
        """
        self.output_dir = output_dir
        self.processor = processor if processor else AudioProcessor()
        self.metadata = []
        self.common_commands = [
            # Shopping commands in Twi
            "Tɔ adeɛ" ,               # Buy item
            "Fa gu me cart mu",        # Add to cart
            "Hwehwɛ...",              # Search for...
            "Fa firi me cart mu",      # Remove from cart
            "Gyina me deɛ so",         # Checkout
            "Me pɛ sɛ me tɔ",          # I want to buy
            "Di kan",                  # Continue
            "San wɔ w'akyi",           # Go back
            "Kyerɛ me nneɛma no",      # Show me items
            "Kyerɛ me nneɛma a ɛwɔ cart mu", # Show cart
            "Pintim me nhyehyɛeɛ",     # Confirm order
            "Twa me ka",               # Make payment
            "Mmisa nsɛm",              # Ask questions
            "Mesrɛ wo, boa me",        # Please help me
            "Dae",                    # Cancel
            "Kyerɛ me ɛboɔ ne mfoni",  # Show price and pictures
            "Sesa dodoɔ no",           # Change quantity
            "Kyerɛ me adaka a ɛwɔ hɔ", # Show available categories
            "Kyerɛ me nsɛnkyerɛnne",   # Show description
            "Fa ma me",                # Save for later
        ]
        self.intent_map = {
            "Tɔ adeɛ": "purchase",
            "Fa gu me cart mu": "add_to_cart",
            "Hwehwɛ...": "search",
            "Fa firi me cart mu": "remove_from_cart",
            "Gyina me deɛ so": "checkout",
            "Me pɛ sɛ me tɔ": "intent_to_buy",
            "Di kan": "continue",
            "San wɔ w'akyi": "go_back",
            "Kyerɛ me nneɛma no": "show_items",
            "Kyerɛ me nneɛma a ɛwɔ cart mu": "show_cart",
            "Pintim me nhyehyɛeɛ": "confirm_order",
            "Twa me ka": "make_payment",
            "Mmisa nsɛm": "ask_questions",
            "Mesrɛ wo, boa me": "help",
            "Dae": "cancel",
            "Kyerɛ me ɛboɔ ne mfoni": "show_price_images",
            "Sesa dodoɔ no": "change_quantity",
            "Kyerɛ me adaka a ɛwɔ hɔ": "show_categories",
            "Kyerɛ me nsɛnkyerɛnne": "show_description",
            "Fa ma me": "save_for_later",
        }

        os.makedirs(self.output_dir, exist_ok=True)

    def record_sample(self, command, duration=3, sample_rate=16000):
        """
        Record a voice sample for a given command

        Args:
            command: Command to record
            duration: Recording duration in seconds
            sample_rate: Sample rate for recording

        Returns:
            Path to saved recording
        """
        print(f"Prepare to say: '{command}'")
        time.sleep(1)
        print("Recording in 3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)

        print("Recording... speak now")
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        print("Recording complete!")

        # Generate filename
        timestamp = int(time.time())
        safe_command = command.replace(" ", "_").replace("...", "").lower()
        filename = f"{safe_command}_{timestamp}.wav"
        filepath = os.path.join(self.output_dir, filename)

        # Save the recording
        sf.write(filepath, audio_data, sample_rate)

        # Add metadata
        self.metadata.append({
            'file': filepath,
            'command': command,
            'intent': self.intent_map.get(command, "unknown"),
            'timestamp': timestamp
        })

        return filepath

    def collect_dataset(self, num_samples_per_command=5):
        """
        Collect a dataset by recording multiple samples for each command

        Args:
            num_samples_per_command: Number of samples to record per command
        """
        try:
            for command in tqdm(self.common_commands, desc="Recording commands"):
                print(f"\n--- Command: {command} ({self.intent_map.get(command, 'unknown')}) ---")
                for i in range(num_samples_per_command):
                    print(f"\nSample {i+1}/{num_samples_per_command}")
                    self.record_sample(command)
                    time.sleep(1)

            self.save_metadata()
            print(f"Dataset collection complete! {len(self.metadata)} samples recorded.")
        except KeyboardInterrupt:
            print("\nRecording interrupted. Saving collected samples...")
            self.save_metadata()
            print(f"Saved {len(self.metadata)} samples.")

    def load_existing_dataset(self, metadata_path):
        """
        Load an existing dataset metadata file

        Args:
            metadata_path: Path to metadata CSV
        """
        self.metadata = pd.read_csv(metadata_path).to_dict('records')
        print(f"Loaded {len(self.metadata)} sample records from {metadata_path}")

    def save_metadata(self):
        """Save dataset metadata to CSV"""
        metadata_df = pd.DataFrame(self.metadata)
        output_path = os.path.join(self.output_dir, "metadata.csv")
        metadata_df.to_csv(output_path, index=False)
        print(f"Metadata saved to {output_path}")

    def augment_data(self, noise_factor=0.05, pitch_shift_range=2, time_stretch_range=(0.9, 1.1)):
        """
        Augment data to create more training samples

        Args:
            noise_factor: Factor for noise addition
            pitch_shift_range: Range for pitch shifting (semitones)
            time_stretch_range: Range for time stretching

        Returns:
            List of new metadata entries
        """
        new_metadata = []

        for entry in tqdm(self.metadata, desc="Augmenting data"):
            audio_path = entry['file']
            y, sr = librosa.load(audio_path, sr=None)

            # 1. Add noise
            noise = np.random.randn(len(y))
            augmented_noise = y + noise_factor * noise
            noise_filename = os.path.join(self.output_dir, f"aug_noise_{os.path.basename(audio_path)}")
            sf.write(noise_filename, augmented_noise, sr)
            new_metadata.append({
                'file': noise_filename,
                'command': entry['command'],
                'intent': entry['intent'],
                'timestamp': int(time.time()),
                'augmentation': 'noise'
            })

            # 2. Pitch shift
            shift = np.random.uniform(-pitch_shift_range, pitch_shift_range)
            augmented_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift)
            pitch_filename = os.path.join(self.output_dir, f"aug_pitch_{os.path.basename(audio_path)}")
            sf.write(pitch_filename, augmented_pitch, sr)
            new_metadata.append({
                'file': pitch_filename,
                'command': entry['command'],
                'intent': entry['intent'],
                'timestamp': int(time.time()),
                'augmentation': 'pitch_shift'
            })

            # 3. Time stretch
            stretch_factor = np.random.uniform(*time_stretch_range)
            augmented_tempo = librosa.effects.time_stretch(y, rate=stretch_factor)
            tempo_filename = os.path.join(self.output_dir, f"aug_tempo_{os.path.basename(audio_path)}")
            sf.write(tempo_filename, augmented_tempo, sr)
            new_metadata.append({
                'file': tempo_filename,
                'command': entry['command'],
                'intent': entry['intent'],
                'timestamp': int(time.time()),
                'augmentation': 'time_stretch'
            })

        # Add new entries to metadata
        self.metadata.extend(new_metadata)
        self.save_metadata()

        return new_metadata
