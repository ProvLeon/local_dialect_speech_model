MODEL_CONFIG = {
    # Audio processing
    "sample_rate": 16000,
    "n_mfcc": 13,
    "n_mels": 40,
    "n_fft": 2048,
    "hop_length": 512,
    "max_audio_length": 5,  # seconds

    # Feature extraction
    "feature_max_length": 500,  # Max sequence length for padding/truncation
    "feature_type": "combined",  # Can be 'mfcc', 'melspec', or 'combined'

    # Model architecture
    "input_dim": 94,  # Will be overridden by actual feature dimension
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.5,

    # Training
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 50,  # Increased epochs for better learning
    "early_stopping_patience": 10,  # More patience
    "weight_decay": 0.01,  # L2 regularization strength

    # Paths
    "model_dir": "data/models",
    "best_model_path": "data/models/best_model.pt",
    "label_map_path": "data/processed/label_map.npy"
}
