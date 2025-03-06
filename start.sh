#!/bin/bash

# Download models if they don't exist
if [ ! -f "data/models_improved/best_model.pt" ]; then
    echo "Downloading model files..."
    mkdir -p data/models_improved
    mkdir -p data/processed_augmented

    # Replace these URLs with your actual model file URLs
    # For example from an S3 bucket or other storage
    curl -L "https://your-model-storage-url/best_model.pt" -o data/models_improved/best_model.pt
    curl -L "https://your-model-storage-url/label_map.npy" -o data/processed_augmented/label_map.npy
fi

# Start the application
# python app.py api
uvicorn src.api.speech_api:app --host 0.0.0.0 --port 8000 --reload
