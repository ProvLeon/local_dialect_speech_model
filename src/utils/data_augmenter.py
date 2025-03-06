import os
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import torchaudio
from tqdm import tqdm
import time
import random
from ..preprocessing.audio_processor import AudioProcessor

class AdvancedDataAugmenter:
    def __init__(self, metadata_path, output_dir="data/augmented", processor=None):
        """
        Advanced data augmentation for audio commands

        Args:
            metadata_path: Path to metadata CSV
            output_dir: Directory to store augmented data
            processor: Audio processor instance
        """
        self.metadata = pd.read_csv(metadata_path)
        self.output_dir = output_dir
        self.processor = processor if processor else AudioProcessor()
        self.new_metadata = []

        os.makedirs(self.output_dir, exist_ok=True)

    def analyze_class_distribution(self):
        """Analyze and print class distribution"""
        counts = self.metadata['intent'].value_counts()
        print("\n=== Class Distribution ===")
        for intent, count in counts.items():
            print(f"{intent}: {count} samples")

        # Calculate target counts for balanced dataset
        max_count = counts.max()
        targets = {intent: max_count for intent in counts.index}

        # Check if dataset is already balanced
        if len(set(counts)) == 1:  # Fixed: removed .values()
            print("\nThe dataset is already balanced (all classes have the same number of samples).")
        else:
            print("\n=== Augmentation Targets for Balancing ===")
            for intent, target in targets.items():
                current = counts[intent]
                needed = target - current
                print(f"{intent}: {current} → {target} (need {needed} more)")

        return counts, targets

    def time_shift(self, audio, sr, shift_limit=0.4):
        """Apply time shifting"""
        shift = int(random.random() * shift_limit * sr)
        direction = 1 if random.random() > 0.5 else -1
        shift = shift * direction
        audio_shifted = np.roll(audio, shift)
        # Set the rolled part to zero
        if shift > 0:
            audio_shifted[:shift] = 0
        else:
            audio_shifted[shift:] = 0
        return audio_shifted

    def pitch_shift(self, audio, sr, pitch_limit=5):
        """Apply pitch shifting"""
        pitch_factor = random.uniform(-pitch_limit, pitch_limit)
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_factor)

    def time_stretch(self, audio, stretch_limit=(0.8, 1.2)):
        """Apply time stretching"""
        stretch_factor = random.uniform(*stretch_limit)
        audio_stretched = librosa.effects.time_stretch(audio, rate=stretch_factor)

        # Handle length differences
        if len(audio_stretched) > len(audio):
            audio_stretched = audio_stretched[:len(audio)]
        elif len(audio_stretched) < len(audio):
            pad_length = len(audio) - len(audio_stretched)
            audio_stretched = np.pad(audio_stretched, (0, pad_length), mode='constant')

        return audio_stretched

    def add_noise(self, audio, noise_factor_range=(0.001, 0.03)):
        """Add random noise"""
        noise_factor = random.uniform(*noise_factor_range)
        noise = np.random.randn(len(audio))
        augmented_audio = audio + noise_factor * noise
        return augmented_audio

    def add_background(self, audio, background_path, snr_db_range=(5, 15)):
        """Add background noise at a specified SNR"""
        try:
            background, sr = librosa.load(background_path, sr=None)

            # Make sure background is long enough
            if len(background) < len(audio):
                num_repeats = int(np.ceil(len(audio) / len(background)))
                background = np.tile(background, num_repeats)
                background = background[:len(audio)]
            else:
                start = random.randint(0, len(background) - len(audio))
                background = background[start:start + len(audio)]

            # Calculate power of signal and background
            audio_power = np.mean(audio ** 2)
            bg_power = np.mean(background ** 2)

            # Convert SNR to linear scale
            snr_db = random.uniform(*snr_db_range)
            snr_linear = 10 ** (snr_db / 10)

            # Scale background accordingly
            bg_scale = np.sqrt(audio_power / (bg_power * snr_linear))
            bg_scaled = background * bg_scale

            # Mix signal with background
            mixed_audio = audio + bg_scaled

            # Normalize
            max_val = np.max(np.abs(mixed_audio))
            if max_val > 1.0:
                mixed_audio = mixed_audio / max_val

            return mixed_audio

        except Exception as e:
            print(f"Error adding background: {e}")
            return audio

    def apply_bandpass_filter(self, audio, sr, low_cut=300, high_cut=3400):
        """Apply bandpass filter to simulate telephone audio"""
        from scipy.signal import butter, lfilter

        def butter_bandpass(lowcut, highcut, sr, order=5):
            nyq = 0.5 * sr
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            return b, a

        def bandpass_filter(data, lowcut, highcut, sr, order=5):
            b, a = butter_bandpass(lowcut, highcut, sr, order=order)
            y = lfilter(b, a, data)
            return y

        return bandpass_filter(audio, low_cut, high_cut, sr)

    def apply_room_reverb(self, audio, sr, room_scale=0.9, reverberance=50):
        """Apply room reverberation effect"""
        try:
            import pyroomacoustics as pra

            # Create a shoebox room
            room_dim = [8*room_scale, 6*room_scale, 3*room_scale]  # meters
            room = pra.ShoeBox(room_dim, fs=sr, absorption=0.9*(1-reverberance/100), max_order=17)

            # Add source and microphone
            source_pos = np.array([room_dim[0]/2, room_dim[1]/2, room_dim[2]/2])
            room.add_source(source_pos, signal=audio)
            mic_pos = np.array([room_dim[0]/2+1, room_dim[1]/2+0.5, room_dim[2]/2])
            room.add_microphone(mic_pos)

            # Run the simulation
            room.simulate()

            # Get the reverberant signal
            reverb_audio = room.mic_array.signals[0, :]

            # Normalize
            max_val = np.max(np.abs(reverb_audio))
            if max_val > 0:
                reverb_audio = reverb_audio / max_val

            return reverb_audio

        except ImportError:
            print("pyroomacoustics not installed, skipping reverb")
            return audio
        except Exception as e:
            print(f"Error applying reverb: {e}")
            return audio

    def augment_audio(self, audio_path, augmentation_type, output_path):
        """Apply specified augmentation and save result"""
        # Debug: Check if file exists
        if not os.path.exists(audio_path):
            print(f"ERROR: File does not exist: {audio_path}")
            return False

        try:
            # Load audio
            print(f"Loading audio file: {audio_path}")
            audio, sr = librosa.load(audio_path, sr=None)
            print(f"Successfully loaded audio with sample rate {sr}, length {len(audio)}")

            # Apply augmentation
            if augmentation_type == 'time_shift':
                augmented = self.time_shift(audio, sr)
            elif augmentation_type == 'pitch_shift':
                augmented = self.pitch_shift(audio, sr)
            elif augmentation_type == 'time_stretch':
                augmented = self.time_stretch(audio)
            elif augmentation_type == 'add_noise':
                augmented = self.add_noise(audio)
            elif augmentation_type == 'telephony':
                augmented = self.apply_bandpass_filter(audio, sr)
            elif augmentation_type == 'reverb':
                augmented = self.apply_room_reverb(audio, sr)
            elif augmentation_type == 'combined_1':
                # Multiple augmentations in sequence
                augmented = self.time_shift(audio, sr)
                augmented = self.add_noise(augmented, noise_factor_range=(0.001, 0.01))
            elif augmentation_type == 'combined_2':
                augmented = self.pitch_shift(audio, sr, pitch_limit=3)
                augmented = self.time_stretch(augmented, stretch_limit=(0.9, 1.1))
            else:
                augmented = audio

            # Save augmented audio
            sf.write(output_path, augmented, sr)
            return True

        except Exception as e:
            print(f"Error augmenting audio {audio_path}: {e}")
            return False

    def augment_dataset(self, augmentation_factor=3, balanced=True):
        """
        Augment the entire dataset

        Args:
            augmentation_factor: Multiplier for existing data
            balanced: Whether to balance classes through augmentation
        """
        # Analyze current distribution
        counts, targets = self.analyze_class_distribution()

        # If dataset is already balanced but we want augmentations anyway
        all_equal = len(set(counts)) == 1
        if balanced and all_equal:
            print("\nDetected perfectly balanced dataset (all classes have the same number of samples).")
            print(f"Since you requested balanced augmentation but classes are already balanced, using augmentation_factor={augmentation_factor} instead.")
            balanced = False  # Switch to non-balanced mode to ensure we generate some samples

        # Available augmentation types
        augmentation_types = [
            'time_shift',
            'pitch_shift',
            'time_stretch',
            'add_noise',
            'telephony',
            'reverb',
            'combined_1',
            'combined_2'
        ]

        # Track generated files
        generated_files = []

        # Group entries by intent
        grouped = self.metadata.groupby('intent')

        # Process each intent
        for intent, group in grouped:
            print(f"\nProcessing intent: {intent}")
            files = group['file'].tolist()

            # Determine how many augmentations to create
            if balanced:
                current_count = len(files)
                target_count = targets[intent]
                needed = max(0, target_count - current_count)
                # Calculate how many augmentations per file
                augs_per_file = int(np.ceil(needed / current_count)) if current_count > 0 else 0
            else:
                augs_per_file = augmentation_factor

            print(f"Creating {augs_per_file} augmentations per file")

            # Process each file
            for file_path in tqdm(files, desc=f"Augmenting {intent}"):
                for i in range(augs_per_file):
                    # Select random augmentation type
                    aug_type = random.choice(augmentation_types)

                    # Generate output filename
                    basename = os.path.splitext(os.path.basename(file_path))[0]
                    timestamp = int(time.time() * 1000) % 100000
                    filename = f"aug_{aug_type}_{basename}_{timestamp}.wav"
                    output_path = os.path.join(self.output_dir, filename)

                    # Apply augmentation
                    success = self.augment_audio(file_path, aug_type, output_path)

                    if success:
                        # Add to metadata
                        file_metadata = group[group['file'] == file_path].iloc[0].to_dict()
                        new_entry = {
                            'file': output_path,
                            'command': file_metadata.get('command', ''),
                            'intent': intent,
                            'timestamp': int(time.time()),
                            'augmentation': aug_type,
                            'original_file': file_path
                        }
                        self.new_metadata.append(new_entry)
                        generated_files.append(output_path)

        # Save metadata if new entries were generated
        if self.new_metadata:
            self.save_metadata()

            print(f"\nAugmentation complete! Generated {len(generated_files)} new audio files.")

            # Report final distribution
            combined_df = pd.concat([
                self.metadata[['file', 'intent']],
                pd.DataFrame(self.new_metadata)[['file', 'intent']]
            ])
            final_counts = combined_df['intent'].value_counts()

            print("\n=== Final Class Distribution ===")
            for intent, count in final_counts.items():
                print(f"{intent}: {count} samples")
        else:
            print("\nNo files were augmented. Check the following:")
            print("1. Make sure the input metadata file exists and has entries")
            print("2. Check that the audio files referenced in metadata exist")
            print("3. Ensure the augmentation parameters allow for file generation")

            # Return empty results
            return [], []

        return generated_files, self.new_metadata

    def save_metadata(self):
        """Save augmented metadata to CSV"""
        # Check if new metadata exists
        if not self.new_metadata:
            print("No new metadata to save.")
            return

        # Create DataFrame for new entries
        new_df = pd.DataFrame(self.new_metadata)

        # Save to CSV
        output_path = os.path.join(self.output_dir, "augmented_metadata.csv")
        new_df.to_csv(output_path, index=False)
        print(f"Augmented metadata saved to {output_path}")

        # Also create combined metadata
        combined_df = pd.concat([
            self.metadata,
            new_df
        ], ignore_index=True)

        combined_path = os.path.join(self.output_dir, "combined_metadata.csv")
        combined_df.to_csv(combined_path, index=False)
        print(f"Combined metadata saved to {combined_path}")
