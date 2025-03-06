# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories required by the app
RUN mkdir -p data/models data/processed

# Copy model files
COPY models/best_model.pt data/models_improved/
COPY models/label_map.npy data/processed_augmented/

# Expose port
EXPOSE 8000

# Command to run the application
# CMD ["python", "app.py", "api"]
# CMD ["./start.sh"]
CMD ["uvicorn", "src.api.speech_api:app", "--reload"]
