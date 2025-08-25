# Akan (Twi) Speech-to-Action System

A deep learning system that recognizes speech commands in the Akan language (specifically Twi) and translates them into actionable e-commerce requests.

## Features

- Speech recognition for Akan (Twi) language
- Intent classification for e-commerce actions
- RESTful API for integration with e-commerce platforms
- Support for multiple e-commerce actions (purchase, add to cart, search, etc.)

## Project Structure

```
akan_speech_to_action/
├── data/
│   ├── raw/             # Raw audio recordings
│   ├── processed/       # Preprocessed audio features
│   └── models/          # Trained models
├── src/
│   ├── preprocessing/   # Audio preprocessing modules
│   ├── features/        # Feature extraction
│   ├── models/          # ML/DL models
│   ├── api/             # API implementation
│   └── utils/           # Utility functions
├── notebooks/           # Jupyter notebooks for exploration
├── tests/               # Unit and integration tests
├── config/              # Configuration files
├── app.py               # Main application entry point
├── Dockerfile           # For containerization
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch
- Libraries listed in requirements.txt

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/akan-speech-to-action.git
cd akan-speech-to-action
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (or create a .env file):
```
ECOMMERCE_API_KEY=your_api_key
ECOMMERCE_API_URL=https://api.yourecommerce.com
```

### Data Collection

To record your own Twi speech commands:

```bash
python app.py collect
```

Follow the prompts to record speech samples for each command.

### Feature Extraction

Process the recorded audio files:

```bash
python app.py extract --metadata data/raw/metadata.csv --output-dir data/processed
```

### Training

Train the model:

```bash
python app.py train --data-dir data/processed --model-dir data/models --epochs 20
```

### Running the API Server

Start the API server:

```bash
python app.py api --host 0.0.0.0 --port 8000
```

### Docker Deployment

Build and run using Docker:

```bash
docker build -t akan-speech-to-action .
docker run -p 8000:8000 --env-file .env akan-speech-to-action
```

## API Endpoints

- `POST /recognize`: Recognize speech from audio file
- `POST /action`: Recognize speech and execute corresponding action
- `GET /intents`: List available intents
- `GET /health`: Health check endpoint

## Testing

Run tests:

```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [librosa](https://librosa.org/) for audio processing
- [PyTorch](https://pytorch.org/) for deep learning
- [FastAPI](https://fastapi.tiangolo.com/) for API implementation
```

### 13. Final Integration and Project Overview

The Akan (Twi) Speech-to-Action system is now fully implemented. Here's a summary of what we've built:

1. **Audio Processing Pipeline**
   - Audio loading and format conversion
   - Feature extraction with MFCCs
   - Noise reduction
   - Feature normalization

2. **Dataset Creation Utilities**
   - Recording tool for Twi speech commands
   - Data augmentation techniques for improved model generalization
   - Feature extraction pipeline

3. **Deep Learning Model**
   - BiLSTM-based architecture with attention mechanism
   - Training pipeline with early stopping
   - Model evaluation and saving

4. **E-commerce Integration**
   - Support for 20 different intents related to e-commerce actions
   - Flexible action execution framework
   - API integration with e-commerce platforms

5. **RESTful API**
   - Speech recognition endpoint
   - Action execution endpoint
   - Available intents listing
   - Health check

6. **Testing and Deployment**
   - Unit tests for core components
   - Docker configuration for containerized deployment
   - Documentation and usage examples

### Running the Complete System

To use this system in a real-world scenario:

1. **Data Collection**:
   - Collect speech samples from native Twi speakers
   - Ensure diverse pronunciation patterns
   - Include background noise variations for robustness

2. **Model Training**:
   - Train the model on the collected dataset
   - Fine-tune hyperparameters for optimal performance
   - Evaluate on held-out test set

3. **API Deployment**:
   - Deploy using Docker on preferred cloud platform
   - Set up SSL for secure communication
   - Configure rate limiting to prevent abuse

4. **Integration**:
   - Connect to e-commerce platform API
   - Implement proper authentication
   - Add monitoring and logging

5. **User Experience**:
   - Create a mobile app or web interface for end users
   - Implement feedback mechanisms to improve recognition
   - Add contextual awareness for better intent understanding

This implementation provides a complete foundation for building a production-ready Speech-to-Action system specifically tailored for the Akan language, focusing on e-commerce applications.
