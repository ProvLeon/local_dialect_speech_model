# HuggingFace Model Integration Guide

## Overview

The Twi Speech Recognition Engine now supports seamless integration with HuggingFace models, allowing you to use both single-task (transcription only) and multi-task (transcription + intent classification) models from the HuggingFace Hub.

## Features

### ✅ **Automatic Model Detection**
- Detects whether a model is single-task or multi-task
- Adapts functionality based on model capabilities
- Fallback mechanisms for missing components

### ✅ **Complete E-commerce Intent Support**
- All 30+ intents from `prompts_lean.csv` supported
- Navigation: `go_home`, `go_back`, `show_cart`, `open_account`, etc.
- Search & Discovery: `search`, `apply_filter`, `sort_items`, etc.
- Cart Operations: `add_to_cart`, `remove_from_cart`, `change_quantity`, etc.
- Checkout & Payment: `checkout`, `make_payment`, `cancel_order`, etc.
- Post-Purchase: `track_order`, `return_item`, `exchange_item`, etc.
- And more...

### ✅ **Performance Optimizations**
- GPU acceleration with FP16 precision
- Model compilation for PyTorch 2.0+
- Efficient memory management
- Async processing support

## Quick Start

### 1. **Basic Usage**

Start the server with a HuggingFace model:

```bash
# Using your trained multi-task model
python main.py server --huggingface your_username/twi-whisper-multitask

# Using any Whisper model from HuggingFace
python main.py server --huggingface openai/whisper-small

# With custom port
python main.py server --port 9000 --huggingface your_username/model-name
```

### 2. **Supported Model Types**

#### **Multi-Task Models** 🎯
Models that support both transcription and intent classification:

```bash
python main.py server --huggingface your_username/twi-whisper-intent-model
```

**Expected files in repository:**
- `config.json` (with `num_labels` or `task_types`)
- `pytorch_model.bin` or model weights
- `intent_labels.json` or `label_map.json` (intent mappings)
- `tokenizer.json` and related files

#### **Single-Task Models** 📝
Standard Whisper or speech-to-text models:

```bash
python main.py server --huggingface openai/whisper-small
python main.py server --huggingface openai/whisper-medium
python main.py server --huggingface facebook/wav2vec2-large-960h
```

## Model Configuration

### **Multi-Task Model Structure**

Your HuggingFace repository should contain:

```
your-repo/
├── config.json              # Model configuration
├── pytorch_model.bin        # Model weights
├── tokenizer.json          # Tokenizer
├── tokenizer_config.json   # Tokenizer config
├── intent_labels.json      # Intent label mappings
├── preprocessor_config.json # Audio preprocessing
└── README.md               # Model documentation
```

### **Example `intent_labels.json`**

```json
{
  "label_to_id": {
    "search": 0,
    "add_to_cart": 1,
    "show_cart": 2,
    "checkout": 3,
    "go_home": 4,
    "go_back": 5,
    "apply_filter": 6,
    "show_description": 7,
    "make_payment": 8,
    "track_order": 9
  },
  "id_to_label": {
    "0": "search",
    "1": "add_to_cart",
    "2": "show_cart",
    "3": "checkout",
    "4": "go_home",
    "5": "go_back",
    "6": "apply_filter",
    "7": "show_description",
    "8": "make_payment",
    "9": "track_order"
  }
}
```

### **Example `config.json` for Multi-Task**

```json
{
  "_name_or_path": "openai/whisper-small",
  "model_type": "whisper",
  "architectures": ["WhisperForConditionalGeneration"],
  "num_labels": 30,
  "custom_model": "WhisperForMultiTask",
  "task_types": ["transcription", "classification"],
  "d_model": 768,
  "max_source_positions": 1500,
  "vocab_size": 51865
}
```

## API Usage

### **Standard API Calls**

Once the server is running, the API works exactly the same:

```javascript
// Frontend JavaScript
const formData = new FormData();
formData.append('file', audioBlob);

const response = await fetch('/test-intent?top_k=5', {
    method: 'POST',
    body: formData
});

const result = await response.json();
```

### **Response Format**

```json
{
  "filename": "audio.wav",
  "intent": "add_to_cart",
  "confidence": 0.87,
  "transcription": "Fa yei to cart no mu",
  "processing_time_ms": 2340,
  "top_predictions": [
    {"intent": "add_to_cart", "confidence": 0.87, "index": 0},
    {"intent": "save_for_later", "confidence": 0.12, "index": 1},
    {"intent": "show_cart", "confidence": 0.08, "index": 2}
  ],
  "model_type": "optimized_whisper_intent_fast",
  "whisper_info": {
    "model_size": "small",
    "language": "tw",
    "transcription_confidence": 0.92
  }
}
```

## Model Performance

### **Expected Performance by Model Type**

| Model Type | Response Time | Memory Usage | Accuracy |
|------------|---------------|--------------|----------|
| Multi-Task HF | 2-4s | 2-4GB | 85-95% |
| Single-Task HF | 1-3s | 1-3GB | 80-90% |
| Whisper Small | 2-5s | 2GB | 75-85% |
| Whisper Medium | 3-8s | 5GB | 80-90% |

### **Optimization Tips**

1. **Use GPU**: 3-5x faster processing
2. **Enable FP16**: 50% memory reduction on GPU
3. **Model Size**: Balance accuracy vs speed
4. **Cache Results**: Repeated audio processed instantly

## Monitoring & Debugging

### **Performance Stats**

```bash
curl http://localhost:8000/performance-stats
```

```json
{
  "performance_metrics": {
    "total_requests": 150,
    "successful_requests": 147,
    "avg_processing_time": 2.34,
    "cache_hits": 23
  },
  "optimization_status": {
    "torch_optimized": true,
    "cuda_available": true,
    "cache_enabled": true
  },
  "model_info": {
    "model_path": "/path/to/downloaded/model",
    "model_type": "multi",
    "device": "cuda",
    "model_source": "huggingface"
  }
}
```

### **Health Check**

```bash
curl http://localhost:8000/health
```

### **Debug Logs**

Monitor the server logs for model loading and processing information:

```bash
# Start server with verbose logging
LOG_LEVEL=DEBUG python main.py server --huggingface your_username/model-name

# Watch logs
tail -f logs/api_server.log
```

## Troubleshooting

### **Common Issues**

| Issue | Cause | Solution |
|-------|-------|----------|
| Model not found | Invalid repo ID | Check HuggingFace repo exists |
| Authentication error | Private repo | Set HF_TOKEN environment variable |
| Out of memory | Model too large | Use smaller model or enable FP16 |
| Slow download | Large model | Use `git lfs` or smaller variant |
| Intent fallback | Missing intent files | Add `intent_labels.json` to repo |

### **Environment Variables**

```bash
# HuggingFace token for private repos
export HF_TOKEN="your_huggingface_token"

# Force CPU usage
export CUDA_VISIBLE_DEVICES=""

# Debug mode
export LOG_LEVEL="DEBUG"
```

## Model Development

### **Creating Your Own Multi-Task Model**

1. **Train your model** using the training scripts
2. **Save with proper structure**:
   ```python
   # Save model
   model.save_pretrained("./my-twi-model")

   # Save intent labels
   import json
   with open("./my-twi-model/intent_labels.json", "w") as f:
       json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, f)
   ```

3. **Upload to HuggingFace**:
   ```bash
   # Install HuggingFace CLI
   pip install huggingface_hub

   # Login
   huggingface-cli login

   # Upload model
   huggingface-cli upload your_username/twi-whisper-model ./my-twi-model
   ```

4. **Test integration**:
   ```bash
   python main.py server --huggingface your_username/twi-whisper-model
   ```

### **Model Card Template**

Include this information in your model's README.md:

```markdown
# Twi Speech Recognition Model

## Model Description
- **Language**: Twi (Akan)
- **Task**: Speech Recognition + Intent Classification
- **Base Model**: openai/whisper-small
- **Fine-tuned on**: Twi e-commerce voice commands

## Supported Intents
Navigation, Search, Cart Operations, Checkout, Orders, etc.

## Usage
\`\`\`bash
python main.py server --huggingface your_username/model-name
\`\`\`

## Performance
- Response Time: 2-4 seconds
- Intent Accuracy: 87%
- Transcription WER: 15%
```

## Advanced Usage

### **Custom Model Paths**

You can also use local models by setting environment variables:

```bash
# Download model first
export HUGGINGFACE_MODEL_PATH="/path/to/local/model"
export HUGGINGFACE_MODEL_TYPE="multi"

# Start server
python main.py server
```

### **Batch Processing**

Process multiple audio files:

```bash
curl -X POST http://localhost:8000/batch-intent \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav" \
  -F "files=@audio3.wav"
```

### **Integration with Training Pipeline**

After training a new model:

```bash
# Upload to HuggingFace
python upload_to_huggingface.py --model-path ./trained_model

# Test immediately
python main.py server --huggingface your_username/new-model-name
```

## Examples

### **Example 1: E-commerce Store Integration**

```python
import requests
import soundfile as sf

# Record customer voice command
audio_data, sample_rate = sf.read("customer_command.wav")

# Send to API
files = {"file": open("customer_command.wav", "rb")}
response = requests.post("http://localhost:8000/test-intent", files=files)
result = response.json()

# Handle intent
if result["intent"] == "add_to_cart":
    # Add product to cart
    add_to_cart(product_id)
elif result["intent"] == "search":
    # Perform search
    search_products(result["transcription"])
```

### **Example 2: Voice Assistant Integration**

```python
class TwiVoiceAssistant:
    def __init__(self):
        self.api_url = "http://localhost:8000/test-intent"

    async def process_voice_command(self, audio_file):
        # Send to Twi speech recognition
        response = await self.send_audio(audio_file)

        # Execute appropriate action
        intent = response["intent"]
        confidence = response["confidence"]

        if confidence > 0.7:
            return await self.execute_intent(intent, response)
        else:
            return "Mɛntumi ntie wo yiye. San ka bio." # Could not understand
```

## Support

For issues or questions:

1. **Check logs**: Review server logs for error details
2. **Performance stats**: Monitor `/performance-stats` endpoint
3. **Model validation**: Ensure model files are complete
4. **Community**: Check HuggingFace model discussions

## Version Compatibility

- **Python**: 3.8+
- **PyTorch**: 2.0+
- **Transformers**: 4.36+
- **HuggingFace Hub**: Latest

This integration provides seamless support for both your custom trained models and any compatible models from the HuggingFace ecosystem, maintaining full compatibility with your existing Twi e-commerce intents while adding the flexibility to use community models.
