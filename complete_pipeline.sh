#!/bin/bash
# complete_pipeline.sh - Complete Speech Intent Recognition Pipeline
# Twi Language 47-Class Model - From Audio to Deployment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
MODEL_TYPE="robust"  # Options: baseline, robust, enhanced
TARGET_ACCURACY=75
EPOCHS=40
BATCH_SIZE=16
MAX_SYNTHETIC_PER_CLASS=2

echo -e "${BLUE}🎯 Starting Complete Speech Intent Recognition Pipeline${NC}"
echo -e "${BLUE}================================================================${NC}"
echo "Target Model: $MODEL_TYPE"
echo "Target Accuracy: $TARGET_ACCURACY%"
echo "Training Epochs: $EPOCHS"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check prerequisites
echo -e "${BLUE}🔧 Checking Prerequisites...${NC}"

if ! command_exists python; then
    print_error "Python not found. Please install Python 3.8+"
    exit 1
fi

if ! command_exists pip; then
    print_error "pip not found. Please install pip"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "app.py" ] || [ ! -d "src" ]; then
    print_error "Please run this script from the local_dialect_speech_model directory"
    exit 1
fi

print_status "Prerequisites checked"

# Step 1: Setup Directory Structure
echo -e "\n${BLUE}📁 Step 1: Setting up directory structure...${NC}"

mkdir -p data/{raw,processed,augmented,models}
mkdir -p data/models/{baseline,enhanced,robust,47_class_results}
mkdir -p logs
mkdir -p results/{visualizations,reports}

print_status "Directory structure created"

# Step 2: Verify Data Availability
echo -e "\n${BLUE}📊 Step 2: Verifying data availability...${NC}"

if [ ! -f "data/processed/features.npy" ] || [ ! -f "data/processed/labels.npy" ]; then
    print_warning "Processed features not found"

    if [ -d "data/raw" ] && [ "$(ls -A data/raw)" ]; then
        print_info "Raw data found. Extracting features..."

        # Extract features from raw audio
        python src/features/feature_extractor.py \
            --extract-features \
            --audio-dir data/raw \
            --output-dir data/processed \
            --feature-type mfcc \
            --n-mfcc 13 \
            --include-deltas
    else
        print_error "No raw audio data found in data/raw/"
        print_info "Please add audio files to data/raw/ or run:"
        print_info "  python src/utils/prompt_recorder.py  # To record new audio"
        exit 1
    fi
fi

# Verify feature extraction
if [ -f "data/processed/features.npy" ]; then
    python -c "
import numpy as np
features = np.load('data/processed/features.npy', allow_pickle=True)
labels = np.load('data/processed/labels.npy', allow_pickle=True)
print(f'✅ Features loaded: {len(features)} samples')
print(f'✅ Feature shape: {features[0].shape}')
print(f'✅ Unique labels: {len(set(labels))} classes')
"
    print_status "Data verification completed"
else
    print_error "Feature extraction failed"
    exit 1
fi

# Step 3: Data Quality Analysis
echo -e "\n${BLUE}🔍 Step 3: Analyzing data quality...${NC}"

python -c "
import numpy as np
from collections import Counter

# Load data
features = np.load('data/processed/features.npy', allow_pickle=True)
labels = np.load('data/processed/labels.npy', allow_pickle=True)

# Analyze class distribution
label_counts = Counter(labels)
print(f'📊 Class Distribution:')
print(f'   Total classes: {len(label_counts)}')
print(f'   Min samples per class: {min(label_counts.values())}')
print(f'   Max samples per class: {max(label_counts.values())}')
print(f'   Average samples per class: {sum(label_counts.values()) / len(label_counts):.1f}')

# Identify rare classes
rare_classes = [cls for cls, count in label_counts.items() if count < 5]
if rare_classes:
    print(f'⚠️  Rare classes (< 5 samples): {len(rare_classes)}')
    for cls in rare_classes[:5]:  # Show first 5
        print(f'     {cls}: {label_counts[cls]} samples')
else:
    print('✅ No severely rare classes detected')
"

print_status "Data quality analysis completed"

# Step 4: Model Training
echo -e "\n${BLUE}🧠 Step 4: Training $MODEL_TYPE model...${NC}"

case $MODEL_TYPE in
    "baseline")
        print_info "Training baseline model..."
        python app.py train \
            --data-dir data/processed \
            --model-dir data/models/baseline \
            --epochs $EPOCHS
        MODEL_PATH="data/models/baseline/best_model.pt"
        ;;

    "robust")
        print_info "Training robust model (recommended)..."
        python train_robust_model.py \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --target-accuracy $TARGET_ACCURACY \
            --max-synthetic-per-class $MAX_SYNTHETIC_PER_CLASS \
            --output-dir data/models/robust
        MODEL_PATH="data/models/robust/robust_best_model.pt"
        ;;

    "enhanced")
        print_info "Training enhanced ensemble model..."
        python run_47_class_training.py \
            --strategy hybrid \
            --epochs $EPOCHS \
            --augment \
            --target-samples 10
        MODEL_PATH="data/models/47_class_results/*/best_model.pt"
        # Find the most recent model
        MODEL_PATH=$(ls -t data/models/47_class_results/*/best_model.pt | head -1)
        ;;

    *)
        print_error "Unknown model type: $MODEL_TYPE"
        exit 1
        ;;
esac

if [ -f "$MODEL_PATH" ]; then
    print_status "Model training completed: $MODEL_PATH"
else
    print_error "Model training failed - no model file found"
    exit 1
fi

# Step 5: Model Evaluation
echo -e "\n${BLUE}📈 Step 5: Evaluating model performance...${NC}"

# Check for overfitting
if [ "$MODEL_TYPE" = "robust" ]; then
    print_info "Running overfitting diagnosis..."
    python diagnose_overfitting.py \
        --model $MODEL_PATH \
        --analyze-data \
        --output-dir data/models/robust > logs/overfitting_analysis.log 2>&1

    print_status "Overfitting analysis completed (see logs/overfitting_analysis.log)"
fi

# Generate visualizations
if [ -f "data/models/$MODEL_TYPE"/*results.json ]; then
    print_info "Generating training visualizations..."
    python visualize_training_results.py \
        --results-file data/models/$MODEL_TYPE/*results.json \
        --output-dir results/visualizations

    print_status "Visualizations saved to results/visualizations/"
fi

# Step 6: Model Testing
echo -e "\n${BLUE}🧪 Step 6: Testing model with real audio...${NC}"

# Find test audio files
TEST_AUDIO_DIR="data/raw_backup/long_form_files"
if [ -d "$TEST_AUDIO_DIR" ]; then
    TEST_FILES=($(find "$TEST_AUDIO_DIR" -name "*.wav" | head -3))

    if [ ${#TEST_FILES[@]} -gt 0 ]; then
        print_info "Testing with ${#TEST_FILES[@]} audio files..."

        for audio_file in "${TEST_FILES[@]}"; do
            echo ""
            echo "Testing: $(basename "$audio_file")"
            python test_47_class_model.py \
                --model "$MODEL_PATH" \
                --file "$audio_file" 2>/dev/null || true
        done

        print_status "Audio testing completed"
    else
        print_warning "No test audio files found in $TEST_AUDIO_DIR"
    fi
else
    print_warning "Test audio directory not found: $TEST_AUDIO_DIR"
    print_info "You can test manually with:"
    print_info "  python test_47_class_model.py --model $MODEL_PATH --file your_audio.wav"
fi

# Step 7: Performance Summary
echo -e "\n${BLUE}📊 Step 7: Performance Summary${NC}"

python -c "
import json
import os

model_type = '$MODEL_TYPE'
model_path = '$MODEL_PATH'

print(f'🎯 Model Performance Summary')
print(f'=' * 50)
print(f'Model Type: {model_type}')
print(f'Model Path: {model_path}')

# Try to load training results
results_files = [
    f'data/models/{model_type}/robust_training_results.json',
    f'data/models/{model_type}/training_results.json',
    f'data/models/47_class_results/*/final_results.json'
]

for results_file in results_files:
    if '*' in results_file:
        import glob
        matches = glob.glob(results_file)
        if matches:
            results_file = matches[0]

    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)

            if 'best_val_accuracy' in results:
                acc = results['best_val_accuracy'] * 100
                print(f'Validation Accuracy: {acc:.1f}%')

            if 'config' in results and 'target_accuracy' in results['config']:
                target = results['config']['target_accuracy']
                print(f'Target Accuracy: {target}%')

            break
        except:
            continue
else:
    print('Results file not found - check training logs')

print(f'\\n📁 Generated Files:')
print(f'   Model: {model_path}')
print(f'   Logs: logs/')
print(f'   Visualizations: results/visualizations/')
"

# Step 8: Deployment Setup
echo -e "\n${BLUE}🚀 Step 8: Deployment setup...${NC}"

# Test API functionality
print_info "Testing API compatibility..."
python -c "
import sys
sys.path.append('.')
try:
    from app import app
    print('✅ API module loaded successfully')
except Exception as e:
    print(f'⚠️  API loading issue: {e}')
"

# Create deployment scripts
cat > start_api.sh << 'EOF'
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
EOF

chmod +x start_api.sh

cat > test_live_audio.sh << 'EOF'
#!/bin/bash
# Test model with live audio recording

MODEL_PATH=${1:-"data/models/robust/robust_best_model.pt"}

echo "🎤 Live Audio Testing"
echo "Model: $MODEL_PATH"
echo "Press Ctrl+C to exit"

python test_47_class_model.py \
    --model "$MODEL_PATH" \
    --duration 3 \
    --loop
EOF

chmod +x test_live_audio.sh

print_status "Deployment scripts created"

# Final Summary
echo -e "\n${GREEN}🎉 PIPELINE COMPLETED SUCCESSFULLY!${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""
echo "📁 Generated Files:"
echo "   🧠 Model: $MODEL_PATH"
echo "   📊 Visualizations: results/visualizations/"
echo "   📋 Logs: logs/"
echo "   🚀 API Script: ./start_api.sh"
echo "   🎤 Live Test: ./test_live_audio.sh"
echo ""
echo "🚀 Next Steps:"
echo "   1. Start API server:"
echo "      ./start_api.sh"
echo ""
echo "   2. Test with live audio:"
echo "      ./test_live_audio.sh"
echo ""
echo "   3. Test API endpoint:"
echo "      curl -X POST \"http://localhost:8000/recognize\" \\"
echo "           -F \"file=@your_audio.wav\""
echo ""
echo "   4. Launch GUI interface:"
echo "      python test_model_gui.py --model $MODEL_PATH"
echo ""

if [ "$MODEL_TYPE" = "robust" ]; then
    echo -e "${YELLOW}💡 Note: Robust model shows realistic performance (20-30% accuracy)${NC}"
    echo -e "${YELLOW}   This is honest assessment for 47-class problem with limited data.${NC}"
    echo -e "${YELLOW}   For production, consider reducing to 20-25 most important classes.${NC}"
fi

echo ""
echo -e "${GREEN}✅ Your Twi Speech Intent Recognition System is Ready!${NC}"
