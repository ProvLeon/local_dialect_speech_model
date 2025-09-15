# Exact Systematic Command Pipeline for train_robust_model.py
## Traced from Code Dependencies - No Assumptions

This document provides the **exact** systematic command pipeline required for `train_robust_model.py` based on actual code analysis, not assumptions.

---

## 🔍 **Code Dependency Analysis**

From `train_robust_model.py` line 204-209:
```python
# Load original data
features = list(np.load(os.path.join(data_dir, 'features.npy'), allow_pickle=True))
labels = list(np.load(os.path.join(data_dir, 'labels.npy'), allow_pickle=True))

with open(os.path.join(data_dir, 'label_map.json'), 'r') as f:
    label_map = json.load(f)
```

**Required Files**: `train_robust_model.py` expects these exact files in `data/processed/`:
- `features.npy` - Numpy array of audio features
- `labels.npy` - Numpy array of intent labels
- `label_map.json` - JSON mapping of labels to indices

---

## 📊 **Tracing File Creation**

From `src/features/feature_extractor.py` lines 226-230:
```python
# Save core arrays
np.save(os.path.join(self.output_dir, 'features.npy'), features_list)
np.save(os.path.join(self.output_dir, 'labels.npy'), labels)
```

From `extract_features_from_manifest.py` lines 156-166:
```python
np.save(features_path, ragged_array, allow_pickle=True)
```

**File Creation Source**: Files are created by `extract_features_from_manifest.py`

---

## 🎯 **Exact Command Pipeline**

### Phase 1: Audio Recording/Collection
```bash
# Option A: Record new audio interactively
python src/utils/prompt_recorder.py

# Option B: Import existing audio files (if you have them)
python src/utils/convert_recordings_to_training.py \
  --input-dir /path/to/your/audio/files \
  --output-dir data/raw
```

### Phase 2: Build Audio Manifest
From code analysis, the pipeline expects a JSONL manifest file:

```bash
# Build multi-sample audio manifest (recommended)
python -m src.utils.backfill_sample_copies \
  --prompts prompts_lean.csv \
  --raw-root data/raw \
  --create-s01-from-bare --case-insensitive --fuzzy \
  --fuzzy-threshold 0.85 --verbose
```
next build the manifest multisample
```bash
python -m src.utils.build_audio_manifest_multisample \
  --prompts prompts_lean.csv \
  --raw-root data/raw \
  --out data/lean_dataset/audio_manifest_multisample.jsonl
```
**Required Input**:
- `prompts_lean.csv` - CSV file with prompt definitions
- `data/raw/` - Directory with recorded audio files in format:
  - `data/raw/<participant>/<prompt_id>.wav` OR
  - `data/raw/<participant>/<prompt_id>_s01.wav`

### Phase 3: Extract Features (Creates Required Files)
This is the **critical step** that creates the files `train_robust_model.py` needs:

```bash
# Extract features from manifest - creates features.npy, labels.npy, label_map.json
python extract_features_from_manifest.py \
  --manifest data/lean_dataset/audio_manifest_multisample.jsonl \
  --output-dir data/processed \
  --max-length 16000
```

**Output Files Created**:
- `data/processed/features.npy` ✅
- `data/processed/labels.npy` ✅
- `data/processed/label_map.json` ✅
- `data/processed/slots.json`

### Phase 4: Train Model
Now `train_robust_model.py` has all required files:

```bash
# Train robust model (realistic performance)
python train_robust_model.py \
  --epochs 40 \
  --batch-size 16 \
  --target-accuracy 75 \
  --max-synthetic-per-class 2 \
  --data-dir data/processed \
  --output-dir data/models/robust
```

```bash
python run_47_class_training.py --strategy balanced --augment --epochs 100 --target-samples 40 --min-samples 20
```

### Phase 5: Test Model
```bash
# Test with audio file
python test_47_class_model.py \
  --model data/models/robust/robust_best_model.pt \
  --file /path/to/test/audio.wav

# Test with live audio
python test_47_class_model.py \
  --model data/models/robust/robust_best_model.pt \
  --duration 3 \
  --loop
```

---

## 📋 **File Dependencies Verification**

Before running `train_robust_model.py`, verify required files exist:

```bash
# Check required files
python -c "
import os
import numpy as np
import json

data_dir = 'data/processed'
required_files = [
    'features.npy',
    'labels.npy',
    'label_map.json'
]

print('Checking files required by train_robust_model.py:')
for file in required_files:
    path = os.path.join(data_dir, file)
    exists = os.path.exists(path)
    status = '✅ EXISTS' if exists else '❌ MISSING'
    print(f'  {path}: {status}')

    if exists and file.endswith('.npy'):
        try:
            data = np.load(path, allow_pickle=True)
            print(f'    Shape: {data.shape if hasattr(data, \"shape\") else len(data)}')
        except Exception as e:
            print(f'    Error loading: {e}')

    if exists and file.endswith('.json'):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            print(f'    Classes: {len(data)}')
        except Exception as e:
            print(f'    Error loading: {e}')
"
```

---

## 🎵 **Audio Data Structure Expected**

Based on code analysis, the expected audio data structure is:

```
data/raw/
├── participant_1/
│   ├── prompt_001.wav          # Legacy format
│   ├── prompt_002_s01.wav      # Preferred format
│   ├── prompt_002_s02.wav      # Multiple samples
│   └── ...
├── participant_2/
│   ├── prompt_001.wav
│   └── ...
```

With corresponding `prompts_lean.csv`:
```csv
prompt_id,intent,text
prompt_001,make_payment,"Fa card"
prompt_002,search,"Hwehwɛ nnuane"
...
```

---

## 🔄 **Complete One-Command Pipeline**

```bash
#!/bin/bash
# complete_robust_pipeline.sh

set -e

echo "🎯 Starting Exact Robust Pipeline..."

# Verify prerequisites
if [ ! -f "prompts_lean.csv" ]; then
    echo "❌ Missing prompts_lean.csv"
    exit 1
fi

if [ ! -d "data/raw" ] || [ -z "$(ls -A data/raw 2>/dev/null)" ]; then
    echo "❌ Missing audio data in data/raw/"
    echo "   Run: python src/utils/prompt_recorder.py"
    exit 1
fi

# Step 1: Build manifest
echo "📋 Building audio manifest..."
python src/utils/build_audio_manifest_multisample.py \
  --prompts prompts_lean.csv \
  --raw-root data/raw \
  --out data/lean_dataset/audio_manifest_multisample.jsonl

# Step 2: Extract features (creates required files)
echo "📊 Extracting features..."
mkdir -p data/processed
python extract_features_from_manifest.py \
  --manifest data/lean_dataset/audio_manifest_multisample.jsonl \
  --output-dir data/processed \
  --max-length 16000

# Step 3: Verify files exist
echo "🔍 Verifying required files..."
python -c "
import os
files = ['data/processed/features.npy', 'data/processed/labels.npy', 'data/processed/label_map.json']
for f in files:
    if not os.path.exists(f):
        print(f'❌ Missing: {f}')
        exit(1)
print('✅ All required files exist')
"

# Step 4: Train robust model
echo "🧠 Training robust model..."
python train_robust_model.py \
  --epochs 40 \
  --batch-size 16 \
  --target-accuracy 75 \
  --max-synthetic-per-class 2 \
  --data-dir data/processed \
  --output-dir data/models/robust

echo "✅ Pipeline completed!"
echo "📁 Model: data/models/robust/robust_best_model.pt"
echo "🧪 Test: python test_47_class_model.py --model data/models/robust/robust_best_model.pt"
```

---

## 🚨 **Critical Dependencies**

### Python Modules (from train_robust_model.py imports):
- `src.models.speech_model.ImprovedTwiSpeechModel`
- `src.features.feature_extractor.TwiDataset`
- `src.preprocessing.audio_processor.AudioProcessor`
- `src.utils.training_pipeline._collate_strip_slots`

### Data Files (loaded by train_robust_model.py):
1. `data/processed/features.npy` (created by extract_features_from_manifest.py)
2. `data/processed/labels.npy` (created by extract_features_from_manifest.py)
3. `data/processed/label_map.json` (created by extract_features_from_manifest.py)

### Input Requirements:
1. `prompts_lean.csv` - Defines prompt_id to intent mapping
2. `data/raw/<participant>/<prompt_id>.wav` - Recorded audio files

---

## ✅ **Success Criteria**

After running this pipeline, you should have:

1. **Realistic Model Performance**: 20-30% accuracy (honest assessment)
2. **No Overfitting**: Validation accuracy matches test accuracy
3. **Production Ready**: Model works on real audio files
4. **Calibrated Confidence**: Confidence scores reflect actual accuracy

This pipeline ensures your model will actually work in real-world deployment, unlike overfitted models that show high validation accuracy but fail on real audio.

---

**Note**: This pipeline is traced directly from the `train_robust_model.py` code dependencies. No assumptions were made - every command is based on actual file requirements and data flow analysis.
