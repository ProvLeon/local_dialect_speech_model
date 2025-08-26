# src/utils/enhanced_dataset_builder.py
import os
import pandas as pd
import numpy as np
import json
import logging
import sounddevice as sd
import soundfile as sf
import tempfile
import time
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from .prompts_processor import TwiPromptsProcessor
from ..preprocessing.enhanced_audio_processor import EnhancedAudioProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedTwiDatasetBuilder:
    """
    Enhanced dataset builder that uses the comprehensive Twi prompts
    and creates a more robust training dataset
    """

    def __init__(self, prompts_csv="twi_prompts.csv", output_dir="data/enhanced_raw",
                 processed_prompts_dir="data/processed_prompts"):
        """
        Initialize the enhanced dataset builder

        Args:
            prompts_csv: Path to the CSV file with prompts
            output_dir: Directory to store recordings
            processed_prompts_dir: Directory with processed prompts data
        """
        self.prompts_csv = prompts_csv
        self.output_dir = output_dir
        self.processed_prompts_dir = processed_prompts_dir
        self.processor = EnhancedAudioProcessor()
        self.metadata = []

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load processed prompts data
        self.load_prompts_data()

    def load_prompts_data(self):
        """Load processed prompts data"""
        try:
            # Check if processed data exists, if not create it
            if not os.path.exists(os.path.join(self.processed_prompts_dir, 'training_metadata.json')):
                logger.info("Processed prompts not found. Processing prompts CSV...")
                processor = TwiPromptsProcessor(
                    csv_path=self.prompts_csv,
                    output_dir=self.processed_prompts_dir
                )
                processor.process()

            # Load training metadata
            metadata_path = os.path.join(self.processed_prompts_dir, 'training_metadata.json')
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.prompts_data = json.load(f)

            # Load intent mapping
            intent_mapping_path = os.path.join(self.processed_prompts_dir, 'intent_mapping.json')
            with open(intent_mapping_path, 'r', encoding='utf-8') as f:
                self.intent_mapping = json.load(f)

            # Load text-to-intent mapping
            text_intent_path = os.path.join(self.processed_prompts_dir, 'text_to_intent.json')
            with open(text_intent_path, 'r', encoding='utf-8') as f:
                self.text_to_intent = json.load(f)

            logger.info(f"Loaded {len(self.prompts_data)} prompts with {len(self.intent_mapping)} intents")

        except Exception as e:
            logger.error(f"Error loading prompts data: {e}")
            raise

    def get_prompts_by_section(self, section: str) -> List[Dict]:
        """Get all prompts for a specific section"""
        return [p for p in self.prompts_data if p.get('section') == section]

    def get_prompts_by_intent(self, intent: str) -> List[Dict]:
        """Get all prompts for a specific intent"""
        return [p for p in self.prompts_data if p.get('intent') == intent]

    def display_prompts_summary(self):
        """Display a summary of available prompts"""
        # Group by section
        sections = {}
        intents = {}

        for prompt in self.prompts_data:
            section = prompt.get('section', 'Unknown')
            intent = prompt.get('intent', 'Unknown')

            if section not in sections:
                sections[section] = []
            sections[section].append(prompt)

            if intent not in intents:
                intents[intent] = 0
            intents[intent] += 1

        print("\n" + "="*60)
        print("AVAILABLE PROMPTS SUMMARY")
        print("="*60)

        print(f"\nTotal prompts: {len(self.prompts_data)}")
        print(f"Total intents: {len(intents)}")
        print(f"Total sections: {len(sections)}")

        print("\nPROMPTS BY SECTION:")
        print("-" * 40)
        for section, prompts in sections.items():
            print(f"{section}: {len(prompts)} prompts")
            # Show first few examples
            for i, prompt in enumerate(prompts[:3]):
                print(f"  • {prompt['text']} -> {prompt['intent']}")
            if len(prompts) > 3:
                print(f"  ... and {len(prompts) - 3} more")
            print()

        print("INTENT DISTRIBUTION:")
        print("-" * 40)
        sorted_intents = sorted(intents.items(), key=lambda x: x[1], reverse=True)
        for intent, count in sorted_intents:
            print(f"{intent}: {count}")

        print("="*60)

    def record_sample(self, text: str, intent: str, duration: int = 3,
                     sample_rate: int = 16000) -> Optional[str]:
        """
        Record a voice sample for a given text and intent

        Args:
            text: Text to record
            intent: Associated intent
            duration: Recording duration in seconds
            sample_rate: Sample rate for recording

        Returns:
            Path to saved recording or None if failed
        """
        try:
            print(f"\nPrepare to say: '{text}'")
            print(f"Intent: {intent}")
            print(f"English meaning: {self.get_meaning_for_text(text)}")

            # Countdown
            for i in range(3, 0, -1):
                print(f"Recording in {i}...")
                time.sleep(1)

            print("🎤 Recording... Speak now!")
            audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
            sd.wait()
            print("✅ Recording complete!")

            # Generate filename
            timestamp = int(time.time())
            safe_text = self.make_safe_filename(text)
            filename = f"{intent}_{safe_text}_{timestamp}.wav"
            filepath = os.path.join(self.output_dir, filename)

            # Save the recording
            sf.write(filepath, audio_data, sample_rate)

            # Add metadata
            self.metadata.append({
                'file': filepath,
                'text': text,
                'intent': intent,
                'timestamp': timestamp,
                'duration': duration,
                'sample_rate': sample_rate,
                'meaning': self.get_meaning_for_text(text)
            })

            logger.info(f"Recorded: {text} -> {intent}")
            return filepath

        except Exception as e:
            logger.error(f"Error recording sample: {e}")
            return None

    def get_meaning_for_text(self, text: str) -> str:
        """Get English meaning for a Twi text"""
        for prompt in self.prompts_data:
            if prompt['text'] == text:
                return prompt.get('meaning', 'No meaning available')
        return 'No meaning available'

    def make_safe_filename(self, text: str) -> str:
        """Create a safe filename from text"""
        import re
        # Remove or replace unsafe characters
        safe = re.sub(r'[^\w\s-]', '', text)
        safe = re.sub(r'[-\s]+', '_', safe)
        return safe[:30]  # Limit length

    def interactive_recording_session(self):
        """Run an interactive recording session with menu choices"""
        while True:
            try:
                print("\n" + "="*50)
                print("ENHANCED TWI DATASET BUILDER")
                print("="*50)
                print("1. Record by section")
                print("2. Record by intent")
                print("3. Record specific prompts")
                print("4. Record all prompts")
                print("5. Show prompts summary")
                print("6. Save and exit")
                print("7. Exit without saving")

                choice = input("\nSelect an option (1-7): ").strip()

                if choice == '1':
                    self.record_by_section()
                elif choice == '2':
                    self.record_by_intent()
                elif choice == '3':
                    self.record_specific_prompts()
                elif choice == '4':
                    self.record_all_prompts()
                elif choice == '5':
                    self.display_prompts_summary()
                elif choice == '6':
                    self.save_metadata()
                    print("✅ Metadata saved. Exiting...")
                    break
                elif choice == '7':
                    print("Exiting without saving...")
                    break
                else:
                    print("Invalid option. Please try again.")

            except KeyboardInterrupt:
                print("\n\n⚠️ Recording interrupted.")
                save_choice = input("Save recorded data? (y/n): ").strip().lower()
                if save_choice == 'y':
                    self.save_metadata()
                    print("✅ Data saved.")
                break
            except Exception as e:
                logger.error(f"Error in interactive session: {e}")

    def record_by_section(self):
        """Record prompts organized by section"""
        # Get available sections
        sections = set(p['section'] for p in self.prompts_data)
        sections = sorted(list(sections))

        print("\nAvailable sections:")
        for i, section in enumerate(sections, 1):
            section_prompts = self.get_prompts_by_section(section)
            print(f"{i}. {section} ({len(section_prompts)} prompts)")

        try:
            choice = int(input("\nSelect section number: ")) - 1
            if 0 <= choice < len(sections):
                selected_section = sections[choice]
                section_prompts = self.get_prompts_by_section(selected_section)

                samples_per_prompt = int(input(f"Samples per prompt (default 3): ") or "3")

                print(f"\nRecording {len(section_prompts)} prompts from {selected_section} section...")
                print(f"Each prompt will be recorded {samples_per_prompt} times.")

                for prompt in tqdm(section_prompts, desc=f"Recording {selected_section}"):
                    print(f"\n--- Prompt: {prompt['text']} ---")
                    for i in range(samples_per_prompt):
                        print(f"Sample {i+1}/{samples_per_prompt}")
                        self.record_sample(prompt['text'], prompt['intent'])

                        if i < samples_per_prompt - 1:
                            input("Press Enter for next sample...")

                print(f"✅ Completed recording {selected_section} section!")
            else:
                print("Invalid section number.")
        except ValueError:
            print("Invalid input.")

    def record_by_intent(self):
        """Record prompts organized by intent"""
        # Get available intents
        intents = set(p['intent'] for p in self.prompts_data)
        intents = sorted(list(intents))

        print("\nAvailable intents:")
        for i, intent in enumerate(intents, 1):
            intent_prompts = self.get_prompts_by_intent(intent)
            print(f"{i}. {intent} ({len(intent_prompts)} prompts)")

        try:
            choice = int(input("\nSelect intent number: ")) - 1
            if 0 <= choice < len(intents):
                selected_intent = intents[choice]
                intent_prompts = self.get_prompts_by_intent(selected_intent)

                samples_per_prompt = int(input(f"Samples per prompt (default 3): ") or "3")

                print(f"\nRecording {len(intent_prompts)} prompts for '{selected_intent}' intent...")

                for prompt in tqdm(intent_prompts, desc=f"Recording {selected_intent}"):
                    print(f"\n--- Prompt: {prompt['text']} ---")
                    print(f"Meaning: {prompt['meaning']}")
                    for i in range(samples_per_prompt):
                        print(f"Sample {i+1}/{samples_per_prompt}")
                        self.record_sample(prompt['text'], prompt['intent'])

                        if i < samples_per_prompt - 1:
                            input("Press Enter for next sample...")

                print(f"✅ Completed recording '{selected_intent}' intent!")
            else:
                print("Invalid intent number.")
        except ValueError:
            print("Invalid input.")

    def record_specific_prompts(self):
        """Record specific selected prompts"""
        print("\nEnter prompt numbers to record (comma-separated):")

        # Display all prompts with numbers
        for i, prompt in enumerate(self.prompts_data, 1):
            print(f"{i}. {prompt['text']} -> {prompt['intent']} ({prompt['meaning']})")

        try:
            selections = input("\nPrompt numbers: ").strip()
            if not selections:
                return

            indices = [int(x.strip()) - 1 for x in selections.split(',')]
            samples_per_prompt = int(input(f"Samples per prompt (default 3): ") or "3")

            for idx in indices:
                if 0 <= idx < len(self.prompts_data):
                    prompt = self.prompts_data[idx]
                    print(f"\n--- Prompt: {prompt['text']} ---")
                    print(f"Meaning: {prompt['meaning']}")
                    print(f"Intent: {prompt['intent']}")

                    for i in range(samples_per_prompt):
                        print(f"Sample {i+1}/{samples_per_prompt}")
                        self.record_sample(prompt['text'], prompt['intent'])

                        if i < samples_per_prompt - 1:
                            input("Press Enter for next sample...")
                else:
                    print(f"Invalid prompt number: {idx + 1}")

        except ValueError:
            print("Invalid input.")

    def record_all_prompts(self):
        """Record all available prompts"""
        samples_per_prompt = int(input(f"Samples per prompt (default 2): ") or "2")

        print(f"\nRecording ALL {len(self.prompts_data)} prompts...")
        print(f"Each prompt will be recorded {samples_per_prompt} times.")
        print("This will take a while. Press Ctrl+C to stop.")

        input("Press Enter to start or Ctrl+C to cancel...")

        for prompt in tqdm(self.prompts_data, desc="Recording all prompts"):
            print(f"\n--- Prompt: {prompt['text']} ---")
            print(f"Meaning: {prompt['meaning']}")
            print(f"Intent: {prompt['intent']}")

            for i in range(samples_per_prompt):
                print(f"Sample {i+1}/{samples_per_prompt}")
                self.record_sample(prompt['text'], prompt['intent'])

                if i < samples_per_prompt - 1:
                    time.sleep(1)  # Brief pause between samples

        print("✅ Completed recording all prompts!")

    def save_metadata(self):
        """Save dataset metadata to CSV and JSON"""
        if not self.metadata:
            logger.warning("No metadata to save")
            return

        # Save as CSV
        metadata_df = pd.DataFrame(self.metadata)
        csv_path = os.path.join(self.output_dir, "metadata.csv")
        metadata_df.to_csv(csv_path, index=False, encoding='utf-8')

        # Save as JSON
        json_path = os.path.join(self.output_dir, "metadata.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        # Save intent mapping for compatibility
        intent_map_path = os.path.join(self.output_dir, "intent_mapping.json")
        with open(intent_map_path, 'w', encoding='utf-8') as f:
            json.dump(self.intent_mapping, f, ensure_ascii=False, indent=2)

        # Save statistics
        stats = self.generate_recording_statistics()
        stats_path = os.path.join(self.output_dir, "recording_statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved metadata: {len(self.metadata)} recordings")
        logger.info(f"CSV: {csv_path}")
        logger.info(f"JSON: {json_path}")
        logger.info(f"Intent mapping: {intent_map_path}")
        logger.info(f"Statistics: {stats_path}")

    def generate_recording_statistics(self):
        """Generate statistics about recorded data"""
        if not self.metadata:
            return {}

        stats = {
            'total_recordings': len(self.metadata),
            'unique_intents': len(set(m['intent'] for m in self.metadata)),
            'unique_texts': len(set(m['text'] for m in self.metadata)),
            'total_duration': sum(m.get('duration', 0) for m in self.metadata),
            'intent_distribution': {},
            'text_distribution': {}
        }

        # Intent distribution
        for metadata in self.metadata:
            intent = metadata['intent']
            stats['intent_distribution'][intent] = stats['intent_distribution'].get(intent, 0) + 1

        # Text distribution
        for metadata in self.metadata:
            text = metadata['text']
            stats['text_distribution'][text] = stats['text_distribution'].get(text, 0) + 1

        return stats

    def load_existing_metadata(self):
        """Load existing metadata if available"""
        csv_path = os.path.join(self.output_dir, "metadata.csv")
        json_path = os.path.join(self.output_dir, "metadata.json")

        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded existing metadata: {len(self.metadata)} recordings")
        elif os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            self.metadata = df.to_dict('records')
            logger.info(f"Loaded existing metadata from CSV: {len(self.metadata)} recordings")


def main():
    """Main function to run the enhanced dataset builder"""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Twi Dataset Builder")
    parser.add_argument("--prompts", type=str, default="twi_prompts.csv",
                       help="Path to prompts CSV file")
    parser.add_argument("--output", type=str, default="data/enhanced_raw",
                       help="Output directory for recordings")
    parser.add_argument("--processed-prompts", type=str, default="data/processed_prompts",
                       help="Directory with processed prompts data")
    parser.add_argument("--load-existing", action="store_true",
                       help="Load existing metadata if available")

    args = parser.parse_args()

    # Initialize builder
    builder = EnhancedTwiDatasetBuilder(
        prompts_csv=args.prompts,
        output_dir=args.output,
        processed_prompts_dir=args.processed_prompts
    )

    # Load existing metadata if requested
    if args.load_existing:
        builder.load_existing_metadata()

    # Start interactive session
    builder.interactive_recording_session()


if __name__ == "__main__":
    main()
