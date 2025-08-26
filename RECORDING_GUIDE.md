# Enhanced Recording Guide for Akan Speech Dataset

This guide covers the new participant-based recording system with Voice Activity Detection (VAD) and advanced features.

## 🚀 Quick Start

### 1. Install Recording Dependencies
```bash
pip install -r requirements.txt
# or specifically:
pip install webrtcvad keyboard
```

### 2. Start Recording Session
```bash
# Basic recording with participant ID
python -m src.utils.prompt_recorder --participant P01

# Advanced recording with auto-stop and VAD
python -m src.utils.prompt_recorder --participant P03 --auto-stop --vad webrtc --vad-aggressiveness 2 --silence-ms 400 --allow-early-stop --stop-key s --no-countdown
```

## 📋 Command Line Options

### Required Parameters
- `--participant ID` - Unique participant identifier (e.g., P01, P02, P03)

### Recording Control
- `--auto-stop` - Enable automatic stopping on silence detection (default: enabled)
- `--no-auto-stop` - Disable automatic stopping (manual control only)
- `--silence-ms 400` - Silence duration for auto-stop in milliseconds (default: 500)
- `--allow-early-stop` - Allow manual early stopping with key press (default: enabled)
- `--stop-key s` - Key for manual early stopping (default: 's')
- `--no-countdown` - Skip 3-second countdown before recording

### Voice Activity Detection (VAD)
- `--vad webrtc` - Use WebRTC VAD (recommended) or 'energy' for simple detection
- `--vad-aggressiveness 2` - VAD sensitivity (0=least, 3=most aggressive)

### Quick Recording Modes
- `--section Nav` - Record specific section only
- `--intent search` - Record specific intent only
- `--samples 5` - Number of samples per prompt (default: 3)

### File Paths
- `--output-dir data/recordings` - Base directory for recordings
- `--prompts-file data/processed_prompts/training_metadata.json` - Prompts source

## 🎯 Recording Modes

### 1. Interactive Mode (Recommended)
```bash
python -m src.utils.prompt_recorder --participant P01
```
**Features:**
- Menu-driven interface
- Record by section, intent, or specific prompts
- Real-time progress tracking
- Session management

### 2. Section-Specific Recording
```bash
python -m src.utils.prompt_recorder --participant P01 --section Search --samples 5
```
**Use Cases:**
- Focus on specific functionality (e.g., Navigation, Shopping)
- Systematic coverage by category
- Quality improvement for specific areas

### 3. Intent-Specific Recording
```bash
python -m src.utils.prompt_recorder --participant P02 --intent add_to_cart --samples 3
```
**Use Cases:**
- Balance dataset for specific intents
- Target low-performing intents
- Create intent-specific datasets

## 👥 Participant Management

### Participant Organization
```
data/recordings/
├── P01/                          # Participant 1
│   ├── session_20241225_143022.json
│   ├── search_hwehwe_s01_1698234567.wav
│   └── add_to_cart_fa_yei_s02_1698234590.wav
├── P02/                          # Participant 2
│   ├── session_20241225_150315.json
│   └── help_boa_me_s01_1698234612.wav
├── P03/                          # Participant 3
└── participants.json             # Participant metadata
```

### Participant Metadata
Each participant gets:
- **Unique ID**: Consistent identifier (P01, P02, etc.)
- **Session tracking**: Multiple recording sessions
- **Statistics**: Total recordings, session count
- **Metadata**: Creation date, session history

### File Naming Convention
```
{intent}_{safe_text}_s{sample_number}_{timestamp}.wav
```
Examples:
- `search_hwehwe_ntadee_s01_1698234567.wav`
- `add_to_cart_fa_yei_to_cart_s03_1698234590.wav`
- `help_boa_me_s02_1698234612.wav`

## 🎙️ Recording Features

### Voice Activity Detection (VAD)
**WebRTC VAD** (Recommended):
- Aggressiveness levels 0-3
- Robust speech detection
- Automatic silence detection
- Real-time processing

**Energy-based VAD** (Fallback):
- Simple energy threshold
- Works without WebRTC
- Less accurate but reliable

### Auto-Stop Functionality
- **Silence Detection**: Stops after specified silence duration
- **Speech Trigger**: Only starts counting silence after speech detected
- **Max Duration**: 30-second safety limit
- **Configurable**: Adjust silence threshold per participant

### Manual Controls
- **Early Stop**: Press configured key (default 's') to stop
- **Session Control**: Pause between recordings
- **Skip Options**: Skip difficult prompts
- **Retry**: Re-record failed attempts

## 📊 Session Management

### Recording Session Structure
```json
{
  "start_time": "2024-12-25T14:30:22",
  "end_time": "2024-12-25T15:45:18",
  "recordings_count": 45,
  "settings": {
    "auto_stop": true,
    "vad_aggressiveness": 2,
    "silence_ms": 400
  },
  "recordings": [
    {
      "filename": "search_hwehwe_s01_1698234567.wav",
      "text": "Hwehwɛ ntadeɛ",
      "intent": "search",
      "section": "Search",
      "meaning": "Search for clothing",
      "duration": 2.3,
      "sample_number": 1
    }
  ]
}
```

### Quality Control
- **Duration Check**: Recordings between 0.5-10 seconds
- **Silence Validation**: Ensure speech was captured
- **File Integrity**: Verify audio file creation
- **Metadata Consistency**: Track all recording details

## 🎯 Recording Strategies

### Strategy 1: Comprehensive Coverage
```bash
# Record all sections systematically
python -m src.utils.prompt_recorder --participant P01 --section Nav --samples 3
python -m src.utils.prompt_recorder --participant P01 --section Search --samples 3
python -m src.utils.prompt_recorder --participant P01 --section Product --samples 3
# ... continue for all sections
```

### Strategy 2: Balanced Intent Dataset
```bash
# Focus on balancing intent distribution
python -m src.utils.prompt_recorder --participant P02 --intent search --samples 2
python -m src.utils.prompt_recorder --participant P02 --intent add_to_cart --samples 5
python -m src.utils.prompt_recorder --participant P02 --intent help --samples 3
```

### Strategy 3: Quality-Focused Recording
```bash
# High-quality recording with strict VAD
python -m src.utils.prompt_recorder --participant P03 --vad-aggressiveness 3 --silence-ms 300 --no-countdown
```

## 📈 Best Practices

### Recording Environment
- **Quiet Space**: Minimize background noise
- **Good Microphone**: Clear audio input
- **Consistent Distance**: Same mic positioning
- **Stable Connection**: Reliable hardware setup

### Participant Guidelines
- **Natural Speech**: Speak naturally, not robotic
- **Clear Pronunciation**: Enunciate clearly
- **Consistent Pace**: Maintain steady speaking rate
- **Complete Phrases**: Finish full expressions

### Session Management
- **Regular Breaks**: Avoid fatigue (15-20 min sessions)
- **Session Limits**: 50-100 recordings per session
- **Progress Tracking**: Monitor completion status
- **Quality Review**: Check sample recordings

### Multiple Participants
- **Diverse Voices**: Different ages, genders, accents
- **Native Speakers**: Authentic Twi pronunciation
- **Regional Variation**: Different Twi dialects if applicable
- **Consistent Instructions**: Same recording guidelines

## 🔧 Troubleshooting

### Common Issues

**WebRTC VAD Not Working:**
```bash
pip install webrtcvad
# If still issues, fallback to energy VAD
python -m src.utils.prompt_recorder --participant P01 --vad energy
```

**Keyboard Detection Issues:**
```bash
pip install keyboard
# Or disable early stopping
python -m src.utils.prompt_recorder --participant P01 --no-allow-early-stop
```

**Audio Device Problems:**
```bash
# List available audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Test audio recording
python -c "import sounddevice as sd; import numpy as np; print('Recording test...'); audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1); sd.wait(); print('Test complete')"
```

**File Permission Errors:**
```bash
# Check/create recording directory
mkdir -p data/recordings
chmod 755 data/recordings
```

### Performance Optimization

**For Slow Systems:**
- Use energy VAD instead of WebRTC
- Increase silence threshold (--silence-ms 800)
- Reduce VAD aggressiveness (--vad-aggressiveness 1)

**For Noisy Environments:**
- Increase VAD aggressiveness (--vad-aggressiveness 3)
- Reduce silence threshold (--silence-ms 300)
- Use manual stopping (--no-auto-stop)

## 📊 Data Analysis

### Session Statistics
```bash
# View participant summary
python -c "
import json
with open('data/recordings/participants.json', 'r') as f:
    participants = json.load(f)
for pid, info in participants.items():
    print(f'{pid}: {info[\"total_recordings\"]} recordings, {len(info[\"recording_sessions\"])} sessions')
"
```

### Recording Quality Check
```bash
# Check recording durations
find data/recordings -name "*.wav" -exec soxi -D {} \; | sort -n
```

## 🎉 Integration with Model Training

### After Recording Sessions
1. **Validate Recordings**: Check audio quality and completeness
2. **Create Metadata**: Generate training metadata from session files
3. **Extract Features**: Process audio files for model training
4. **Train Model**: Use participant-diverse dataset

### Metadata Generation
The recorded sessions automatically generate metadata compatible with the training pipeline:
- File paths and participant information
- Intent and section labels
- Recording quality metrics
- Session context and settings

### Training Integration
```bash
# Convert recordings to training format
python src/utils/convert_recordings_to_training.py --recordings-dir data/recordings --output-dir data/enhanced_processed

# Train with participant data
python update_model_with_prompts.py --step train-model --data-dir data/enhanced_processed
```

## 🎯 Example Recording Session

```bash
# 1. Start recording session for participant P01
python -m src.utils.prompt_recorder --participant P01 --auto-stop --vad webrtc --vad-aggressiveness 2

# The system will show:
# 👤 Current Participant: P01
# 📅 Created: 2024-12-25
# 🎙️ Total Recordings: 0
# 📊 Sessions: 0
#
# 📋 RECORDING MENU
# 1. Record by section
# 2. Record by intent
# ...

# 2. Choose "1. Record by section"
# Select "Search" section
# Choose 3 samples per prompt

# 3. Recording process:
# 🎯 Prompt: Hwehwɛ ntadeɛ
# 📝 Meaning: Search for clothing
# 🎪 Intent: search
# ⏳ Preparing to record...
#    3...
#    2...
#    1...
# 🎙️ Recording... Speak now!
# ✅ Saved: search_hwehwe_ntadee_s01_1698234567.wav (2.3s)

# 4. Continue through all prompts in section
# 5. Save session and exit
```

This enhanced recording system provides professional-grade data collection capabilities with participant management, quality control, and seamless integration with the model training pipeline.
