# src/utils/prompts_processor.py
import pandas as pd
import numpy as np
import json
import os
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TwiPromptsProcessor:
    """
    Process Twi prompts CSV file and prepare data for training
    """

    def __init__(self, csv_path="twi_prompts.csv", output_dir="data/processed_prompts"):
        """
        Initialize the prompts processor

        Args:
            csv_path: Path to the CSV file with prompts
            output_dir: Directory to save processed data
        """
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.df = None
        self.intent_mapping = {}
        self.text_to_intent = {}

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load and validate CSV
        self.load_csv()

    def load_csv(self):
        """Load and validate the CSV file"""
        try:
            self.df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded CSV with {len(self.df)} rows")

            # Validate required columns
            required_columns = ['section_id', 'text', 'meaning', 'intent']
            missing_columns = [col for col in required_columns if col not in self.df.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Clean data
            self.df = self.df.dropna(subset=['text'])  # Remove rows without text
            logger.info(f"After cleaning: {len(self.df)} rows")

        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise

    def analyze_intents(self):
        """Analyze the intent distribution in the data"""
        logger.info("Analyzing intent distribution...")

        # Get all non-empty intents
        valid_intents = self.df[self.df['intent'].notna() & (self.df['intent'] != '')]
        intent_counts = valid_intents['intent'].value_counts()

        logger.info(f"Found {len(intent_counts)} unique intents:")
        for intent, count in intent_counts.items():
            logger.info(f"  {intent}: {count} samples")

        # Check for missing intents
        missing_intents = self.df[self.df['intent'].isna() | (self.df['intent'] == '')]
        if len(missing_intents) > 0:
            logger.warning(f"Found {len(missing_intents)} rows with missing intents:")
            for _, row in missing_intents.iterrows():
                logger.warning(f"  {row['id']}: {row['text']} -> {row['meaning']}")

        return intent_counts

    def create_comprehensive_intent_mapping(self):
        """Create a comprehensive intent mapping including all possible intents"""

        # Base intents from CSV
        csv_intents = set(self.df[self.df['intent'].notna() & (self.df['intent'] != '')]['intent'].unique())

        # Add missing intents for complete e-commerce functionality
        additional_intents = {
            'purchase',
            'intent_to_buy',
            'confirm_order',
            'show_categories',
            'select_color',
            'select_size',
            'change_color',
            'change_size',
            'change_order',
            'manage_address',
            'manage_profile',
            'manage_notifications',
            'apply_coupon',
            'remove_coupon',
            'track_order',
            'return_item',
            'exchange_item',
            'view_wishlist',
            'clear_cart'
        }

        # Combine all intents
        all_intents = csv_intents.union(additional_intents)

        # Create mapping
        self.intent_mapping = {intent: idx for idx, intent in enumerate(sorted(all_intents))}

        logger.info(f"Created intent mapping with {len(self.intent_mapping)} intents:")
        for intent, idx in sorted(self.intent_mapping.items(), key=lambda x: x[1]):
            logger.info(f"  {idx}: {intent}")

        return self.intent_mapping

    def assign_missing_intents(self):
        """Assign intents to rows that don't have them based on section and meaning"""

        # Rules for assigning intents based on section and keywords
        intent_assignment_rules = {
            'Nav': {
                'back|akyi': 'go_back',
                'forward|anim|continue': 'continue',
                'cart': 'show_cart',
                'account|profile': 'manage_profile',
                'home|fie': 'continue',
                'app': 'continue'
            },
            'Search': {
                'search|hwehwɛ': 'search'
            },
            'FilterSort': {
                'filter|sort': 'search'
            },
            'Product': {
                'details|description|nsɛm': 'show_description',
                'price|boɔ': 'show_price_images',
                'reviews': 'show_description',
                'similar|items': 'show_items',
                'wishlist': 'save_for_later',
                'add.*cart|cart': 'add_to_cart',
                'remove.*cart': 'remove_from_cart',
                'add.*one|more': 'change_quantity',
                'remove.*one': 'change_quantity',
                'clear.*cart': 'remove_from_cart'
            },
            'Cart': {
                'view.*cart|hwɛ.*cart': 'show_cart',
                'checkout': 'checkout',
                'address': 'manage_address',
                'payment|tua': 'make_payment',
                'mobile.*money|momo': 'make_payment',
                'card': 'make_payment',
                'cancel|gyae': 'cancel',
                'delivery': 'checkout'
            },
            'Orders': {
                'show.*orders|orders': 'show_items',
                'status': 'show_items',
                'track': 'track_order',
                'cancel.*order': 'cancel',
                'return': 'return_item',
                'exchange': 'exchange_item',
                'refund': 'ask_questions',
                'details': 'show_description',
                'receipt': 'show_description'
            },
            'Account': {
                'profile': 'manage_profile',
                'password': 'manage_profile',
                'address': 'manage_address',
                'notifications': 'manage_notifications',
                'support|help|boa': 'help',
                'logout|fi.*mu': 'continue'
            },
            'Deals': {
                'deals|discounts|promo|flash|voucher': 'search',
                'coupon': 'apply_coupon',
                'apply': 'apply_coupon',
                'remove|yi': 'remove_coupon'
            },
            'Brand': {
                'search|hwehwɛ': 'search'
            },
            'Attributes': {
                'select.*color|fa.*color': 'select_color',
                'select.*size|fa.*size': 'select_size',
                'change.*color|sesa.*color': 'change_color',
                'change.*size|sesa.*size': 'change_size'
            },
            'Address': {
                'show.*address|address': 'manage_address',
                'add.*address': 'manage_address',
                'remove.*address': 'manage_address',
                'default': 'manage_address'
            },
            'Notify': {
                'turn.*on|sɔ': 'manage_notifications',
                'turn.*off|gyae': 'manage_notifications',
                'alert': 'manage_notifications'
            },
            'Support': {
                'chat|ticket|support|help|faq': 'help',
                'photo|video|upload': 'help'
            }
        }

        import re
        assigned_count = 0

        for idx, row in self.df.iterrows():
            if pd.isna(row['intent']) or row['intent'] == '':
                section = row['section_id']
                text_lower = str(row['text']).lower()
                meaning_lower = str(row['meaning']).lower()
                combined_text = f"{text_lower} {meaning_lower}"

                if section in intent_assignment_rules:
                    for pattern, intent in intent_assignment_rules[section].items():
                        if re.search(pattern, combined_text):
                            self.df.at[idx, 'intent'] = intent
                            assigned_count += 1
                            logger.info(f"Assigned intent '{intent}' to: {row['text']}")
                            break

        logger.info(f"Assigned intents to {assigned_count} previously missing entries")

    def create_text_to_intent_mapping(self):
        """Create mapping from text commands to intents"""
        self.text_to_intent = {}

        for _, row in self.df.iterrows():
            if pd.notna(row['intent']) and row['intent'] != '':
                text = str(row['text']).strip()
                intent = str(row['intent']).strip()
                self.text_to_intent[text] = intent

        logger.info(f"Created text-to-intent mapping with {len(self.text_to_intent)} entries")
        return self.text_to_intent

    def generate_training_metadata(self):
        """Generate metadata file for training"""
        # Filter rows with valid intents
        valid_data = self.df[self.df['intent'].notna() & (self.df['intent'] != '')]

        metadata = []
        for _, row in valid_data.iterrows():
            metadata.append({
                'text': row['text'],
                'intent': row['intent'],
                'meaning': row['meaning'],
                'section': row['section_id'],
                'id': row['id']
            })

        return metadata

    def save_processed_data(self):
        """Save all processed data to files"""

        # Save intent mapping
        intent_mapping_path = os.path.join(self.output_dir, 'intent_mapping.json')
        with open(intent_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(self.intent_mapping, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved intent mapping to {intent_mapping_path}")

        # Save text-to-intent mapping
        text_intent_path = os.path.join(self.output_dir, 'text_to_intent.json')
        with open(text_intent_path, 'w', encoding='utf-8') as f:
            json.dump(self.text_to_intent, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved text-to-intent mapping to {text_intent_path}")

        # Save training metadata
        metadata = self.generate_training_metadata()
        metadata_path = os.path.join(self.output_dir, 'training_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved training metadata to {metadata_path}")

        # Save processed CSV
        processed_csv_path = os.path.join(self.output_dir, 'processed_prompts.csv')
        self.df.to_csv(processed_csv_path, index=False, encoding='utf-8')
        logger.info(f"Saved processed CSV to {processed_csv_path}")

        # Save label map (compatible with existing model format)
        label_map = {intent: idx for intent, idx in self.intent_mapping.items()}
        label_map_path = os.path.join(self.output_dir, 'label_map.json')
        with open(label_map_path, 'w', encoding='utf-8') as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved label map to {label_map_path}")

        # Also save as .npy for compatibility
        label_map_npy_path = os.path.join(self.output_dir, 'label_map.npy')
        np.save(label_map_npy_path, label_map)
        logger.info(f"Saved label map (numpy) to {label_map_npy_path}")

        return {
            'intent_mapping': intent_mapping_path,
            'text_to_intent': text_intent_path,
            'metadata': metadata_path,
            'processed_csv': processed_csv_path,
            'label_map': label_map_path,
            'label_map_npy': label_map_npy_path
        }

    def generate_statistics(self):
        """Generate statistics about the processed data"""
        stats = {}

        # Basic statistics
        stats['total_prompts'] = len(self.df)
        stats['prompts_with_intents'] = len(self.df[self.df['intent'].notna() & (self.df['intent'] != '')])
        stats['prompts_without_intents'] = stats['total_prompts'] - stats['prompts_with_intents']

        # Intent distribution
        valid_intents = self.df[self.df['intent'].notna() & (self.df['intent'] != '')]
        intent_counts = valid_intents['intent'].value_counts().to_dict()
        stats['intent_distribution'] = {k: int(v) for k, v in intent_counts.items()}
        stats['unique_intents'] = len(intent_counts)

        # Section distribution
        section_counts = self.df['section_id'].value_counts().to_dict()
        stats['section_distribution'] = {k: int(v) for k, v in section_counts.items()}
        stats['unique_sections'] = len(section_counts)

        # Languages and text length
        stats['avg_text_length'] = float(self.df['text'].str.len().mean())
        stats['max_text_length'] = int(self.df['text'].str.len().max())
        stats['min_text_length'] = int(self.df['text'].str.len().min())

        # Save statistics
        stats_path = os.path.join(self.output_dir, 'statistics.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved statistics to {stats_path}")

        return stats

    def process(self):
        """Main processing pipeline"""
        logger.info("Starting Twi prompts processing...")

        # Step 1: Analyze current intents
        intent_counts = self.analyze_intents()

        # Step 2: Create comprehensive intent mapping
        self.create_comprehensive_intent_mapping()

        # Step 3: Assign missing intents
        self.assign_missing_intents()

        # Step 4: Create text-to-intent mapping
        self.create_text_to_intent_mapping()

        # Step 5: Save processed data
        saved_files = self.save_processed_data()

        # Step 6: Generate statistics
        stats = self.generate_statistics()

        logger.info("Processing complete!")
        logger.info(f"Processed {stats['total_prompts']} prompts")
        logger.info(f"Found {stats['unique_intents']} unique intents")
        logger.info(f"Prompts with intents: {stats['prompts_with_intents']}")
        logger.info(f"Prompts without intents: {stats['prompts_without_intents']}")

        return saved_files, stats


def main():
    """Main function to run the prompts processor"""
    import argparse

    parser = argparse.ArgumentParser(description="Process Twi prompts CSV for training")
    parser.add_argument("--csv", type=str, default="twi_prompts.csv", help="Path to CSV file")
    parser.add_argument("--output", type=str, default="data/processed_prompts", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Process prompts
    processor = TwiPromptsProcessor(csv_path=args.csv, output_dir=args.output)
    saved_files, stats = processor.process()

    print("\n" + "="*50)
    print("PROCESSING COMPLETE")
    print("="*50)
    print(f"Total prompts: {stats['total_prompts']}")
    print(f"Unique intents: {stats['unique_intents']}")
    print(f"Prompts with intents: {stats['prompts_with_intents']}")
    print(f"Average text length: {stats['avg_text_length']:.1f} characters")
    print("\nTop intents:")
    for intent, count in sorted(stats['intent_distribution'].items(),
                               key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {intent}: {count}")
    print(f"\nFiles saved to: {args.output}")
    print("="*50)


if __name__ == "__main__":
    main()
