# config/api_config.py
"""Configuration for API settings"""

API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": False,
    "workers": 4,
    "confidence_threshold": 0.7,
    "allowed_origins": [
        "http://localhost:3000",  # Example frontend origin
        "https://example-ecommerce.com"  # Example production frontend
    ],
    "max_upload_size": 10 * 1024 * 1024,  # 10 MB
    "timeouts": {
        "audio_conversion": 30,  # seconds
        "audio_validation": 15,  # seconds
        "audio_preprocessing": 60,  # seconds
        "model_inference": 30,  # seconds
        "total_request": 120,  # seconds
        "ffmpeg_check": 10,  # seconds
        "subprocess": 30,  # seconds
    }
}
