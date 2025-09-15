#!/bin/bash
# Start Speech Recognition API Server

MODEL_PATH=${1:-"data/models/robust/robust_best_model.pt"}
PORT=${2:-8000}

echo "🚀 Starting Speech Recognition API..."
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "Access: http://localhost:$PORT"

python app.py api \
    --host 0.0.0.0 \
    --port $PORT \
    --model "$MODEL_PATH"
