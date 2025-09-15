#!/bin/bash
# Test model with live audio recording

MODEL_PATH=${1:-"data/models/robust/robust_best_model.pt"}

echo "🎤 Live Audio Testing"
echo "Model: $MODEL_PATH"
echo "Press Ctrl+C to exit"

python test_47_class_model.py \
    --model "$MODEL_PATH" \
    --duration 3 \
    --loop
