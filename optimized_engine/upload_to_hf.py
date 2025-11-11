#!/usr/bin/env python3
"""
HuggingFace Model Upload Script for Twi Whisper Model
===================================================

This script uploads the fine-tuned Twi Whisper model to HuggingFace Hub.
It handles common issues with dependency conflicts and upload errors.

Usage:
    python upload_to_hf.py --model_dir ./models/whisper_twi_multitask --repo_name TwiWhisper_multiTask_tiny
"""

import argparse
import os
import sys
import time
from pathlib import Path


def install_requirements():
    """Install/upgrade required packages with specific versions."""
    print("📦 Installing/upgrading required packages...")

    # Use specific compatible versions
    packages = [
        "huggingface_hub==0.19.4",  # Compatible with transformers 4.36.0
        "requests>=2.25.0",
        "tqdm>=4.64.0",
    ]

    for package in packages:
        try:
            os.system(f"pip install -q {package}")
            print(f"✅ Installed: {package}")
        except Exception as e:
            print(f"⚠️  Warning: Could not install {package}: {e}")


def upload_model(
    model_dir: str,
    repo_name: str,
    username: str = "TwiWhisperModel",
    hf_token: str = None,
    private: bool = False,
):
    """
    Upload model to HuggingFace Hub.

    Args:
        model_dir: Path to the model directory
        repo_name: Name of the repository
        username: HuggingFace username
        hf_token: HuggingFace token
        private: Whether to make repo private
    """

    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Installing huggingface_hub...")
        install_requirements()
        time.sleep(2)
        try:
            from huggingface_hub import HfApi, create_repo, upload_folder
        except ImportError:
            print("❌ Failed to import huggingface_hub after installation")
            return False

    # Validate inputs
    if not os.path.exists(model_dir):
        print(f"❌ Model directory not found: {model_dir}")
        return False

    if not hf_token:
        print("❌ HuggingFace token is required")
        return False

    repo_id = f"{username}/{repo_name}"
    print(f"🚀 Uploading model to: {repo_id}")
    print(f"📂 From directory: {model_dir}")

    try:
        # Initialize API
        api = HfApi()

        # Create repository (if it doesn't exist)
        print("🔨 Creating repository...")
        try:
            create_repo(
                repo_id=repo_id,
                token=hf_token,
                private=private,
                exist_ok=True,
                repo_type="model",
            )
            print("✅ Repository created/verified")
        except Exception as e:
            print(f"⚠️  Repository creation warning: {e}")

        # Upload folder with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"📤 Uploading files (attempt {attempt + 1}/{max_retries})...")

                upload_folder(
                    folder_path=model_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=hf_token,
                    commit_message="Upload fine-tuned Twi Whisper model",
                    ignore_patterns=["*.git*", "*.DS_Store", "__pycache__", "*.pyc"],
                )

                print(f"✅ Successfully uploaded model!")
                print(f"🔗 Model URL: https://huggingface.co/{repo_id}")
                return True

            except Exception as e:
                print(f"❌ Upload attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"⏳ Retrying in 10 seconds...")
                    time.sleep(10)
                else:
                    print("❌ All upload attempts failed")
                    return False

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False


def create_model_card(model_dir: str, repo_name: str):
    """Create a README.md model card for the repository."""

    model_card_content = f"""---
license: apache-2.0
language:
- tw
tags:
- whisper
- speech-recognition
- twi
- ghana
- automatic-speech-recognition
- transformers
datasets:
- custom
metrics:
- wer
- cer
pipeline_tag: automatic-speech-recognition
---

# {repo_name}

This is a fine-tuned Whisper model for Twi (Akan) speech recognition and intent classification.

## Model Description

This model is based on OpenAI's Whisper and has been fine-tuned on Twi audio data for:
- **Speech-to-text transcription** in Twi language
- **Intent classification** for common phrases and commands

## Usage

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa

# Load model and processor
processor = WhisperProcessor.from_pretrained("TwiWhisperModel/{repo_name}")
model = WhisperForConditionalGeneration.from_pretrained("TwiWhisperModel/{repo_name}")

# Load audio
audio, sr = librosa.load("your_audio.wav", sr=16000)

# Process audio
input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features

# Generate transcription
with torch.no_grad():
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print(f"Transcription: {{transcription}}")
```

## Training Data

The model was trained on custom Twi audio recordings with corresponding transcriptions and intent labels.

## Performance

- **Language**: Twi (tw)
- **Task**: Speech Recognition + Intent Classification
- **Base Model**: OpenAI Whisper
- **Training Framework**: HuggingFace Transformers

## Limitations

- Optimized for Twi language
- Performance may vary on different dialects or accents
- Trained on limited domain-specific data

## Citation

If you use this model, please cite:

```
@misc{{twi_whisper_model,
  title={{Fine-tuned Whisper Model for Twi Speech Recognition}},
  author={{TwiWhisperModel}},
  year={{2024}},
  url={{https://huggingface.co/TwiWhisperModel/{repo_name}}}
}}
```
"""

    readme_path = os.path.join(model_dir, "README.md")

    try:
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(model_card_content)
        print(f"✅ Created model card: {readme_path}")
        return True
    except Exception as e:
        print(f"⚠️  Could not create model card: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload Twi Whisper model to HuggingFace Hub"
    )
    parser.add_argument(
        "--model_dir",
        default="./models/whisper_twi_multitask",
        help="Path to model directory",
    )
    parser.add_argument(
        "--repo_name",
        default="TwiWhisper_multiTask_tiny",
        help="Repository name on HuggingFace",
    )
    parser.add_argument(
        "--username", default="TwiWhisperModel", help="HuggingFace username"
    )
    parser.add_argument(
        "--token", help="HuggingFace token (or set HF_TOKEN environment variable)"
    )
    parser.add_argument(
        "--private", action="store_true", help="Make repository private"
    )
    parser.add_argument(
        "--create_model_card",
        action="store_true",
        default=True,
        help="Create README.md model card",
    )

    args = parser.parse_args()

    # Get token
    hf_token = args.token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("❌ HuggingFace token required!")
        print("Provide token via --token argument or HF_TOKEN environment variable")
        return False

    # Resolve model directory path
    model_dir = os.path.abspath(args.model_dir)

    print("=" * 60)
    print("🤗 HUGGINGFACE MODEL UPLOAD SCRIPT")
    print("=" * 60)
    print(f"📂 Model Directory: {model_dir}")
    print(f"👤 Username: {args.username}")
    print(f"📝 Repository: {args.repo_name}")
    print(f"🔒 Private: {args.private}")
    print("=" * 60)

    # Create model card if requested
    if args.create_model_card:
        create_model_card(model_dir, args.repo_name)

    # Upload model
    success = upload_model(
        model_dir=model_dir,
        repo_name=args.repo_name,
        username=args.username,
        hf_token=hf_token,
        private=args.private,
    )

    if success:
        print("\n🎉 Upload completed successfully!")
        print(f"🔗 Visit: https://huggingface.co/{args.username}/{args.repo_name}")
        return True
    else:
        print("\n❌ Upload failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
