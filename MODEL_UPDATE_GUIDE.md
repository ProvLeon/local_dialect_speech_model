# Akan (Twi) Speech Model Update Guide

This guide walks you through updating your Akan speech-to-action model with the comprehensive prompts from `twi_prompts.csv`.

## Overview

The update process adds support for:
- **25+ intents** covering comprehensive e-commerce functionality
- **170+ Twi prompts** across different categories
- **Enhanced audio processing** with improved feature extraction
- **Better model architecture** with attention mechanisms
- **Extended API support** for all new intents

## Quick Start

### 1. Process the Prompts (Required)
```bash
# Process the CSV file and create intent mappings
python update_model_with_prompts.py --step process-prompts
```

### 2. View Available Prompts
```bash
# See what prompts and intents are available
python update_model_with_prompts.py --step show-summary
```

### 3. Complete Pipeline (Recommended)
```bash
# Run everything including audio collection and training
python update_model_with_prompts.py --complete --collect-audio --epochs 100
```

## Step-by-Step Process

### Step 1: Install Dependencies
```bash
# Make sure you have the required packages
pip install -r requirements.txt
```

### Step 2: Process Prompts
```bash
python update_model_with_prompts.py --step process-prompts
```

**What this does:**
- Reads `twi_prompts.csv`
- Creates comprehensive intent mapping (25+ intents)
- Assigns missing intents using intelligent rules
- Saves processed data to `data/processed_prompts/`

**Output files:**
- `data/processed_prompts/intent_mapping.json` - Intent to index mapping
- `data/processed_prompts/training_metadata.json` - Training data
- `data/processed_prompts/label_map.json` - Label mappings
- `data/processed_prompts/statistics.json` - Data statistics

### Step 3: Audio Data Collection
```bash
python update_model_with_prompts.py --step collect-audio
```

**Interactive Options:**
1. **Record by section** - Record all prompts in a category (Nav, Search, etc.)
2. **Record by intent** - Record all prompts for specific intents
3. **Record specific prompts** - Choose individual prompts
4. **Record all prompts** - Record everything (170+ prompts)

**Recording Process:**
- Each prompt shows the Twi text and English meaning
- 3-second countdown before recording
- Default 3 samples per prompt (customizable)
- Audio saved as WAV files in `data/enhanced_raw/`

### Step 4: Feature Extraction
```bash
python update_model_with_prompts.py --step extract-features
```

**What this does:**
- Processes recorded audio files
- Extracts MFCC + delta features
- Applies noise reduction and normalization
- Creates training-ready feature matrices
- Saves to `data/enhanced_processed/`

### Step 5: Model Training
```bash
python update_model_with_prompts.py --step train-model --epochs 100
```

**Training Features:**
- Enhanced BiLSTM with attention mechanism
- Squeeze-and-excitation blocks for better feature learning
- Advanced training with learning rate scheduling
- Early stopping and model checkpointing
- Class balancing for imbalanced datasets

**Training Outputs:**
- `data/models_enhanced/best_model.pt` - Best model checkpoint
- `data/models_enhanced/model_info.json` - Model metadata
- `data/models_enhanced/training_history.png` - Training curves
- `data/models_enhanced/confusion_matrix.png` - Classification results

### Step 6: Update API Configuration
```bash
python update_model_with_prompts.py --step update-config
```

**What this does:**
- Updates `.env` file with new model paths
- Creates model summary documentation
- Prepares API for new intents

## New Intent Categories

### Navigation (7 intents)
- `go_back` - Return to previous screen
- `continue` - Proceed to next step
- `show_cart` - Display shopping cart

### Shopping & Search (8 intents)
- `search` - Search for products
- `add_to_cart` - Add items to cart
- `remove_from_cart` - Remove items from cart
- `checkout` - Proceed to checkout
- `purchase` - Direct purchase

### Product Management (6 intents)
- `show_items` - Display product lists
- `show_description` - Show product details
- `show_price_images` - Display prices and images
- `change_quantity` - Modify item quantities
- `select_color` - Choose product colors
- `select_size` - Choose product sizes

### Order Management (5 intents)
- `make_payment` - Process payments
- `confirm_order` - Confirm purchases
- `track_order` - Track order status
- `return_item` - Return products
- `exchange_item` - Exchange products

### Account & Support (4 intents)
- `help` - Customer support
- `ask_questions` - Ask product questions
- `manage_profile` - Account management
- `manage_address` - Address management

## Testing the Updated Model

### 1. Test with Live Audio
```bash
python test_enhanced_model.py --model data/models_enhanced/best_model.pt
```

### 2. Test with GUI
```bash
python test_model_gui.py
```

### 3. Test API Endpoints
```bash
# Start the API server
python app.py api

# Test recognition endpoint
curl -X POST "http://localhost:8000/recognize" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_audio.wav"

# List available intents
curl "http://localhost:8000/intents"
```

## File Structure After Update

```
data/
├── processed_prompts/          # Processed CSV data
│   ├── intent_mapping.json     # Intent mappings
│   ├── training_metadata.json  # Training data
│   ├── label_map.json         # Label mappings
│   └── statistics.json        # Data statistics
├── enhanced_raw/              # Recorded audio files
│   ├── metadata.csv           # Audio file metadata
│   └── *.wav                  # Audio recordings
├── enhanced_processed/        # Extracted features
│   ├── features.npy           # Feature matrices
│   ├── labels.npy            # Intent labels
│   └── label_map.npy         # Label mappings
└── models_enhanced/           # Trained models
    ├── best_model.pt          # Best model checkpoint
    ├── model_info.json        # Model metadata
    ├── training_history.png   # Training visualization
    └── confusion_matrix.png   # Classification results
```

## Troubleshooting

### Missing Dependencies
```bash
pip install torch torchaudio librosa soundfile sounddevice pandas numpy scikit-learn matplotlib seaborn tqdm
```

### Audio Recording Issues
- Check microphone permissions
- Ensure `sounddevice` package is installed
- Test with: `python -c "import sounddevice as sd; print(sd.query_devices())"`

### Model Training Issues
- Ensure sufficient disk space (>2GB recommended)
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Reduce batch size if getting memory errors

### API Issues
- Update `.env` file with correct model paths
- Restart API server after model updates
- Check model file permissions

## Performance Expectations

### Dataset Size
- **170+ prompts** from CSV
- **3-5 samples per prompt** = 500+ audio files
- **25+ unique intents**

### Training Time
- **CPU only**: 30-60 minutes
- **GPU**: 10-20 minutes
- **Memory usage**: 2-4GB RAM

### Expected Accuracy
- **Validation**: 85-95%
- **Test**: 80-90%
- **Real-world**: 75-85% (depends on recording quality)

## Next Steps

1. **Collect More Data**: Record additional samples for low-performing intents
2. **Data Augmentation**: Use the built-in augmentation features
3. **Fine-tuning**: Adjust hyperparameters for better performance
4. **Deployment**: Deploy the API to production environment
5. **Monitoring**: Set up logging and performance monitoring

## API Integration

The updated model supports all new intents in the API:

### New Endpoints
- All existing endpoints work with new intents
- `GET /intents` - Lists all 25+ supported intents
- `POST /recognize` - Recognizes any of the new intents
- `POST /action` - Executes actions for new intents

### Intent Examples
```json
{
  "intents": [
    {"intent": "search", "description": "Search for products"},
    {"intent": "add_to_cart", "description": "Add items to cart"},
    {"intent": "select_color", "description": "Select product color"},
    {"intent": "track_order", "description": "Track order status"},
    {"intent": "manage_profile", "description": "Manage user profile"}
  ]
}
```

## Support

For issues or questions:
1. Check the logs in each step's output
2. Review error messages and stack traces
3. Ensure all dependencies are installed
4. Verify file paths and permissions

---

**Happy training! 🎉**
