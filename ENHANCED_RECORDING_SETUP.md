# Enhanced Recording Setup Guide

This guide provides comprehensive instructions for setting up and using the refined participant-based recording system for the Akan (Twi) speech dataset.

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
# Install core dependencies
pip install -r requirements.txt

# Install enhanced recording features
pip install webrtcvad keyboard

# Verify installation
python -c "import webrtcvad, keyboard; print('✅ Enhanced recording dependencies installed')"
```

### 2. Process Prompts (One-time setup)
```bash
# Process the comprehensive prompts from CSV
python update_model_with_prompts.py --step process-prompts
```

### 3. Start Recording
```bash
# Enhanced recording with participant management
python -m src.utils.prompt_recorder --participant P01 --auto-stop --vad webrtc --vad-aggressiveness 2 --silence-ms 400 --allow-early-stop --stop-key s --no-countdown
```

## 📋 Recording System Overview

### Key Features
- **Participant Management**: Organized recording sessions by participant ID
- **Voice Activity Detection (VAD)**: Automatic speech detection and silence stopping
- **Flexible Recording Modes**: By section, intent, or custom selection
- **Quality Control**: Real-time audio validation and session tracking
- **Auto-stop**: Intelligent silence detection with configurable thresholds
- **Manual Controls**: Early stopping, pause between recordings, retry options

### System Architecture
```
Enhanced Recording System
├── Participant Management
│   ├── Unique participant IDs (P01, P02, P03...)
│   ├── Session tracking and metadata
│   └── Organized file structure
├── Voice Activity Detection
│   ├── WebRTC VAD (recommended)
│   ├── Energy-based fallback
│   └── Configurable aggressiveness levels
├── Recording Modes
│   ├── Interactive menu-driven
│   ├── Section-specific recording
│   ├── Intent-specific recording
│   └── Bulk recording options
└── Integration Pipeline
    ├── Automatic format conversion
    ├── Training data preparation
    └── Model training integration
```

## 🎯 Recording Commands Reference

### Basic Recording
```bash
# Start interactive recording session
python -m src.utils.prompt_recorder --participant P01

# Record specific section with auto-stop
python -m src.utils.prompt_recorder --participant P01 --section Search --samples 3

# Record specific intent
python -m src.utils.prompt_recorder --participant P02 --intent add_to_cart --samples 5
```

### Advanced Recording Settings
```bash
# High-quality recording with strict VAD
python -m src.utils.prompt_recorder --participant P03 \
    --vad webrtc \
    --vad-aggressiveness 3 \
    --silence-ms 300 \
    --no-countdown \
    --allow-early-stop \
    --stop-key s

# Manual control mode (no auto-stop)
python -m src.utils.prompt_recorder --participant P04 \
    --no-auto-stop \
    --allow-early-stop \
    --stop-key space

# Energy VAD fallback (if WebRTC unavailable)
python -m src.utils.prompt_recorder --participant P05 \
    --vad energy \
    --silence-ms 800
```

### Batch Recording
```bash
# Record entire section systematically
for section in Nav Search Product Cart Orders; do
    python -m src.utils.prompt_recorder --participant P01 --section $section --samples 3
done

# Record all high-frequency intents
for intent in search help make_payment show_items; do
    python -m src.utils.prompt_recorder --participant P02 --intent $intent --samples 5
done
```

## 📊 Parameter Configuration

### Voice Activity Detection (VAD)
| Parameter | Options | Description | Recommended |
|-----------|---------|-------------|-------------|
| `--vad` | `webrtc`, `energy` | VAD method | `webrtc` |
| `--vad-aggressiveness` | `0-3` | Sensitivity level | `2` |
| `--silence-ms` | `200-1000` | Auto-stop threshold | `400-500` |

**VAD Aggressiveness Levels:**
- `0`: Least aggressive (misses some speech)
- `1`: Low aggressive (good for quiet environments)
- `2`: Moderate (balanced, recommended)
- `3`: Most aggressive (may cut off speech)

### Recording Control
| Parameter | Options | Description | Default |
|-----------|---------|-------------|---------|
| `--auto-stop` | flag | Enable auto-stop | `True` |
| `--no-auto-stop` | flag | Disable auto-stop | `False` |
| `--allow-early-stop` | flag | Enable manual stop | `True` |
| `--stop-key` | any key | Early stop key | `s` |
| `--no-countdown` | flag | Skip 3-2-1 countdown | `False` |

### Recording Scope
| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `--section` | string | Record specific section | `Search`, `Nav`, `Product` |
| `--intent` | string | Record specific intent | `add_to_cart`, `search` |
| `--samples` | integer | Samples per prompt | `3`, `5` |

## 🎙️ Recording Process Workflow

### 1. Session Initialization
```bash
python -m src.utils.prompt_recorder --participant P01
```

**System displays:**
```
👤 Current Participant: P01
📅 Created: 2024-12-25
🎙️ Total Recordings: 0
📊 Sessions: 0

⚙️ Settings:
   Auto-stop: True
   VAD Aggressiveness: 2
   Silence threshold: 500ms
   Early stop key: 's'
```

### 2. Menu Navigation
```
📋 RECORDING MENU
==============================
1. Record by section
2. Record by intent
3. Record specific prompts
4. Record all prompts
5. Show prompts summary
6. Change participant
7. Save session and exit
8. Exit without saving
```

### 3. Recording Process
For each prompt:
```
🎯 Prompt: Hwehwɛ ntadeɛ
📝 Meaning: Search for clothing
🎪 Intent: search
📂 Section: Search
🔢 Sample: 1

⏳ Preparing to record...
   3...
   2...
   1...
🎙️ Recording... Speak now!
   (Auto-stop after 500ms silence)
   (Press 's' to stop early)
✅ Saved: search_hwehwe_ntadee_s01_1698234567.wav (2.3s)
```

### 4. Session Management
- **Progress tracking**: Real-time count of completed recordings
- **Quality validation**: Duration and audio level checks
- **Retry options**: Re-record failed or unsatisfactory attempts
- **Session persistence**: Automatic saving of session metadata

## 📁 File Organization

### Directory Structure
```
data/recordings/
├── participants.json              # Participant metadata
├── P01/                           # Participant 1 recordings
│   ├── session_20241225_143022.json
│   ├── search_hwehwe_ntadee_s01_1698234567.wav
│   ├── add_to_cart_fa_yei_s01_1698234590.wav
│   └── help_boa_me_s02_1698234612.wav
├── P02/                           # Participant 2 recordings
│   ├── session_20241225_150315.json
│   └── make_payment_tua_ka_s01_1698234635.wav
└── P03/                           # Participant 3 recordings
    ├── session_20241225_160420.json
    └── show_cart_hwɛ_cart_s01_1698234658.wav
```

### File Naming Convention
```
{intent}_{safe_text}_s{sample_number}_{timestamp}.wav
```

**Examples:**
- `search_hwehwe_ntadee_s01_1698234567.wav`
- `add_to_cart_fa_yei_to_cart_s03_1698234590.wav`
- `help_boa_me_s02_1698234612.wav`

### Session Metadata Format
```json
{
  "start_time": "2024-12-25T14:30:22",
  "end_time": "2024-12-25T15:45:18",
  "recordings_count": 45,
  "settings": {
    "auto_stop": true,
    "vad_aggressiveness": 2,
    "silence_ms": 500
  },
  "recordings": [
    {
      "filename": "search_hwehwe_s01_1698234567.wav",
      "filepath": "/data/recordings/P01/search_hwehwe_s01_1698234567.wav",
      "text": "Hwehwɛ ntadeɛ",
      "intent": "search",
      "section": "Search",
      "meaning": "Search for clothing",
      "duration": 2.3,
      "sample_number": 1,
      "timestamp": "2024-12-25T14:30:45"
    }
  ]
}
```

## 🎯 Recording Strategies

### Strategy 1: Systematic Section Coverage
**Goal**: Complete coverage of all functionality areas

```bash
# Record each section systematically
sections=("Nav" "Search" "FilterSort" "Product" "Cart" "Orders" "Account" "Deals" "Brand" "Attributes" "Address" "Notify" "Support")

for section in "${sections[@]}"; do
    echo "Recording section: $section"
    python -m src.utils.prompt_recorder --participant P01 --section "$section" --samples 3
done
```

**Benefits:**
- Comprehensive coverage
- Organized workflow
- Balanced representation

### Strategy 2: Intent-Focused Recording
**Goal**: Balance dataset for specific intents

```bash
# Focus on high-frequency intents first
high_freq_intents=("search" "help" "make_payment" "show_items" "show_cart")

for intent in "${high_freq_intents[@]}"; do
    echo "Recording intent: $intent"
    python -m src.utils.prompt_recorder --participant P02 --intent "$intent" --samples 5
done
```

**Benefits:**
- Balanced intent distribution
- Targeted improvement
- Efficient use of recording time

### Strategy 3: Quality-Focused Recording
**Goal**: High-quality recordings with strict controls

```bash
# High-quality recording session
python -m src.utils.prompt_recorder --participant P03 \
    --vad webrtc \
    --vad-aggressiveness 3 \
    --silence-ms 300 \
    --no-countdown \
    --section Search \
    --samples 5
```

**Benefits:**
- Consistent quality
- Minimal background noise
- Professional-grade recordings

### Strategy 4: Multi-Participant Diversity
**Goal**: Diverse speaker representation

```bash
# Multiple participants for diversity
participants=("P01" "P02" "P03" "P04" "P05")
sections=("Search" "Product" "Cart")

for participant in "${participants[@]}"; do
    for section in "${sections[@]}"; do
        echo "Recording $participant - $section"
        python -m src.utils.prompt_recorder --participant "$participant" --section "$section" --samples 2
    done
done
```

**Benefits:**
- Speaker diversity
- Robust model training
- Better generalization

## 🔄 Integration with Training Pipeline

### 1. Convert Recordings to Training Format
```bash
# Convert participant recordings to training format
python src/utils/convert_recordings_to_training.py \
    --recordings-dir data/recordings \
    --output-dir data/enhanced_processed
```

### 2. Extract Features
```bash
# Extract features from converted recordings
python update_model_with_prompts.py --step extract-features
```

### 3. Train Enhanced Model
```bash
# Train model with participant data
python update_model_with_prompts.py --step train-model --epochs 100
```

### 4. Complete Pipeline Integration
```bash
# Run complete pipeline with recording
python update_model_with_prompts.py --complete --collect-audio --participant P01 --epochs 100
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. WebRTC VAD Installation Issues
```bash
# Install webrtcvad
pip install webrtcvad

# If compilation errors on macOS:
brew install portaudio
pip install webrtcvad

# If still failing, use energy VAD
python -m src.utils.prompt_recorder --participant P01 --vad energy
```

#### 2. Keyboard Detection Problems
```bash
# Install keyboard module
pip install keyboard

# If permission issues on macOS:
# Grant terminal accessibility permissions in System Preferences

# Alternative: disable early stopping
python -m src.utils.prompt_recorder --participant P01 --no-allow-early-stop
```

#### 3. Audio Device Issues
```bash
# List available audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Test microphone
python -c "
import sounddevice as sd
import numpy as np
print('Testing microphone...')
audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1)
sd.wait()
print(f'Recorded audio level: {np.max(np.abs(audio)):.4f}')
"
```

#### 4. File Permission Errors
```bash
# Create and set permissions
mkdir -p data/recordings
chmod 755 data/recordings

# Check disk space
df -h data/recordings
```

#### 5. VAD Too Sensitive/Not Sensitive Enough
```bash
# Too sensitive (cutting off speech)
python -m src.utils.prompt_recorder --participant P01 --vad-aggressiveness 1 --silence-ms 600

# Not sensitive enough (not stopping)
python -m src.utils.prompt_recorder --participant P01 --vad-aggressiveness 3 --silence-ms 300
```

### Performance Optimization

#### For Slow Systems
```bash
# Use simpler VAD and longer timeouts
python -m src.utils.prompt_recorder --participant P01 \
    --vad energy \
    --silence-ms 800 \
    --vad-aggressiveness 1
```

#### For Noisy Environments
```bash
# More aggressive VAD and shorter timeouts
python -m src.utils.prompt_recorder --participant P01 \
    --vad webrtc \
    --vad-aggressiveness 3 \
    --silence-ms 200
```

#### For Large Recording Sessions
```bash
# Break into smaller sessions to avoid fatigue
# Record 50-100 prompts per session
# Take breaks every 15-20 minutes
python -m src.utils.prompt_recorder --participant P01 --section Search --samples 3
# ... take break ...
python -m src.utils.prompt_recorder --participant P01 --section Product --samples 3
```

## 📊 Quality Assurance

### Recording Quality Metrics
- **Duration**: 0.5-10 seconds per recording
- **Audio Level**: Clear speech without clipping
- **Background Noise**: Minimal interference
- **Consistency**: Similar volume and clarity across recordings
- **Completeness**: Full phrase capture without cutoffs

### Session Validation
```bash
# Check recording statistics
python -c "
import json
from pathlib import Path

recordings_dir = Path('data/recordings')
if (recordings_dir / 'participants.json').exists():
    with open(recordings_dir / 'participants.json', 'r') as f:
        participants = json.load(f)

    print('📊 Recording Statistics:')
    for pid, info in participants.items():
        print(f'  {pid}: {info[\"total_recordings\"]} recordings, {len(info[\"recording_sessions\"])} sessions')
else:
    print('No participants.json found')
"
```

### Audio Quality Check
```bash
# Check audio file durations
find data/recordings -name "*.wav" -exec soxi -D {} \; | sort -n | tail -10

# Check for silent recordings
find data/recordings -name "*.wav" -exec sh -c 'duration=$(soxi -D "$1"); if [ $(echo "$duration < 0.5" | bc) -eq 1 ]; then echo "Short: $1 ($duration s)"; fi' _ {} \;
```

## 🎉 Best Practices

### Recording Environment
- **Quiet Space**: Minimize background noise
- **Consistent Setup**: Same microphone position and distance
- **Good Equipment**: Use quality microphone for clear audio
- **Stable Connection**: Ensure reliable hardware setup

### Participant Guidelines
- **Natural Speech**: Speak naturally, not robotically
- **Clear Pronunciation**: Enunciate clearly in Twi
- **Consistent Pace**: Maintain steady speaking rate
- **Complete Phrases**: Finish full expressions

### Session Management
- **Regular Breaks**: 15-20 minute sessions to avoid fatigue
- **Progress Tracking**: Monitor completion and quality
- **Session Limits**: 50-100 recordings per session
- **Quality Review**: Listen to sample recordings periodically

### Multi-Participant Coordination
- **Diverse Voices**: Different ages, genders, regional accents
- **Native Speakers**: Authentic Twi pronunciation
- **Consistent Instructions**: Same guidelines for all participants
- **Balanced Contribution**: Similar recording counts per participant

## 🚀 Getting Started Checklist

### Initial Setup
- [ ] Install Python dependencies (`pip install -r requirements.txt`)
- [ ] Install enhanced recording features (`pip install webrtcvad keyboard`)
- [ ] Process prompts (`python update_model_with_prompts.py --step process-prompts`)
- [ ] Test audio setup (`python -c "import sounddevice as sd; print(sd.query_devices())"`)

### First Recording Session
- [ ] Choose participant ID (e.g., P01)
- [ ] Start with small section (e.g., `--section Nav --samples 2`)
- [ ] Test recording quality
- [ ] Adjust VAD settings if needed
- [ ] Complete full section recording

### Production Recording
- [ ] Plan recording strategy (systematic vs. targeted)
- [ ] Set up recording schedule
- [ ] Coordinate multiple participants
- [ ] Monitor progress and quality
- [ ] Convert recordings for training

### Model Training Integration
- [ ] Convert recordings (`python src/utils/convert_recordings_to_training.py`)
- [ ] Extract features (`python update_model_with_prompts.py --step extract-features`)
- [ ] Train model (`python update_model_with_prompts.py --step train-model`)
- [ ] Test updated model (`python test_enhanced_model.py`)

---

## 🎯 Quick Start Example

```bash
# 1. Setup (one-time)
pip install webrtcvad keyboard
python update_model_with_prompts.py --step process-prompts

# 2. Start recording
python -m src.utils.prompt_recorder --participant P01 --auto-stop --vad webrtc --vad-aggressiveness 2 --silence-ms 400

# 3. Follow interactive menu to record by section/intent
# 4. Convert and train
python update_model_with_prompts.py --complete --collect-audio --participant P01
```

Your enhanced recording system is now ready for professional-grade Akan speech data collection! 🎉
