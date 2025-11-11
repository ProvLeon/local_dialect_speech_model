#!/bin/bash

# Fix cPanel Timeout Issues for HuggingFace Model Downloads
# ========================================================
#
# This script addresses the specific timeout issues encountered on cPanel servers
# when downloading large HuggingFace models like TwiWhisperModel/TwiWhisperModel.
#
# Usage: bash fix_cpanel_timeout.sh
#
# Author: AI Assistant
# Date: 2025-11-11

set -e  # Exit on any error

echo "================================================================"
echo "🔧 CPANEL TIMEOUT FIX FOR HUGGINGFACE DOWNLOADS"
echo "================================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to set environment variables
fix_environment() {
    log_info "Setting optimal environment variables..."

    # Create or update .bashrc with timeout settings
    cat >> ~/.bashrc << 'EOF'

# HuggingFace and download optimization settings
export HF_HUB_DOWNLOAD_TIMEOUT=3600
export REQUESTS_TIMEOUT=3600
export CURL_TIMEOUT=3600
export WGET_TIMEOUT=3600
export TRANSFORMERS_CACHE=~/.cache/transformers
export HF_HOME=~/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Disable SSL verification for problematic connections (if needed)
# export CURL_CA_BUNDLE=""
# export REQUESTS_CA_BUNDLE=""
# export SSL_VERIFY=false

EOF

    # Apply immediately
    export HF_HUB_DOWNLOAD_TIMEOUT=3600
    export REQUESTS_TIMEOUT=3600
    export CURL_TIMEOUT=3600
    export WGET_TIMEOUT=3600
    export TRANSFORMERS_CACHE=~/.cache/transformers
    export HF_HOME=~/.cache/huggingface
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

    log_success "Environment variables configured"
}

# Function to create pip configuration
fix_pip_config() {
    log_info "Configuring pip for better timeout handling..."

    mkdir -p ~/.pip
    cat > ~/.pip/pip.conf << 'EOF'
[global]
timeout = 600
retries = 10
trusted-host = pypi.org
               pypi.python.org
               files.pythonhosted.org
               huggingface.co
               cdn-lfs.huggingface.co

[install]
timeout = 600
EOF

    log_success "Pip configuration updated"
}

# Function to configure git for large files
fix_git_config() {
    log_info "Optimizing git configuration for large files..."

    # Check if git is available
    if command -v git &> /dev/null; then
        git config --global http.lowSpeedLimit 1000
        git config --global http.lowSpeedTime 600
        git config --global http.postBuffer 524288000
        git config --global core.preloadindex true
        git config --global core.fscache true
        git config --global gc.auto 256

        # Install and configure Git LFS if not already done
        if ! git lfs version &> /dev/null; then
            log_warning "Git LFS not found, attempting to install..."

            # Try to install git-lfs (this may vary by system)
            if command -v apt-get &> /dev/null; then
                sudo apt-get update && sudo apt-get install -y git-lfs
            elif command -v yum &> /dev/null; then
                sudo yum install -y git-lfs
            elif command -v brew &> /dev/null; then
                brew install git-lfs
            else
                log_warning "Could not auto-install Git LFS. Please install manually:"
                log_warning "  Visit: https://git-lfs.github.io/"
            fi
        fi

        # Configure LFS settings if available
        if git lfs version &> /dev/null; then
            log_info "Configuring Git LFS..."
            git lfs install --system 2>/dev/null || git lfs install
            git config --global lfs.transfer.maxretries 10
            git config --global lfs.transfer.maxverifies 5
            git config --global lfs.locksverify false
            git config --global lfs.batch true
            log_success "Git LFS configured successfully"
        else
            log_warning "Git LFS still not available after installation attempt"
        fi

        log_success "Git configuration optimized"
    else
        log_warning "Git not found, skipping git configuration"
    fi
}

# Function to create cache directories
create_cache_dirs() {
    log_info "Creating cache directories..."

    mkdir -p ~/.cache/transformers
    mkdir -p ~/.cache/huggingface
    mkdir -p ~/.cache/torch
    mkdir -p models/huggingface

    log_success "Cache directories created"
}

# Function to test network connectivity
test_connectivity() {
    log_info "Testing network connectivity..."

    # Test key endpoints
    endpoints=(
        "https://huggingface.co"
        "https://cdn-lfs.huggingface.co"
        "https://cas-bridge.xethub.hf.co"
    )

    for endpoint in "${endpoints[@]}"; do
        if curl -s --connect-timeout 10 --max-time 30 "$endpoint" > /dev/null; then
            log_success "Connected to $endpoint"
        else
            log_warning "Cannot connect to $endpoint"
        fi
    done
}

# Function to create download script
create_download_script() {
    log_info "Creating enhanced download script..."

    cat > download_model_safe.py << 'EOF'
#!/usr/bin/env python3
"""Safe model downloader with multiple fallbacks for cPanel servers."""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

def download_with_retry(repo_id="TwiWhisperModel/TwiWhisperModel", max_attempts=5):
    """Download model with multiple retry strategies."""

    print(f"🚀 Starting download: {repo_id}")

    # Method 1: Enhanced huggingface_hub
    for attempt in range(max_attempts):
        try:
            print(f"📦 Attempt {attempt + 1}: Using huggingface_hub...")

            from huggingface_hub import snapshot_download

            model_path = snapshot_download(
                repo_id=repo_id,
                local_dir=f"models/huggingface/{repo_id.replace('/', '_')}",
                local_dir_use_symlinks=False,
                resume_download=True,
                timeout=1800,  # 30 minutes
                max_workers=1
            )

            # Verify download completeness
            if verify_model_download(model_path):
                print(f"✅ Success: {model_path}")
                return True
            else:
                print(f"⚠️ Download incomplete, missing files")

        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} failed: {e}")
            if attempt < max_attempts - 1:
                wait_time = (attempt + 1) * 120  # 2, 4, 6, 8 minutes
                print(f"⏳ Waiting {wait_time} seconds...")
                time.sleep(wait_time)

    # Method 2: Git LFS clone fallback
    print("🔄 Trying git LFS clone method...")
    try:
        model_dir = f"models/huggingface/{repo_id.replace('/', '_')}"
        os.makedirs(model_dir, exist_ok=True)

        # Check if git-lfs is available
        lfs_available = subprocess.run(["git", "lfs", "version"],
                                     capture_output=True).returncode == 0

        if lfs_available:
            print("✅ Git LFS available, using lfs clone...")
            cmd = [
                "git", "lfs", "clone",
                "--depth=1",
                f"https://huggingface.co/{repo_id}",
                model_dir
            ]
        else:
            print("⚠️ Git LFS not available, using regular clone + pull...")
            cmd = [
                "git", "clone",
                "--depth=1",
                f"https://huggingface.co/{repo_id}",
                model_dir
            ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

        if result.returncode == 0:
            # If regular clone was used, try to pull LFS files
            if not lfs_available:
                print("🔄 Attempting to pull LFS files...")
                lfs_pull = subprocess.run(["git", "lfs", "pull"],
                                        cwd=model_dir, capture_output=True)
                if lfs_pull.returncode != 0:
                    print("⚠️ LFS pull failed, model may be incomplete")

            # Verify download completeness
            if verify_model_download(model_dir):
                print(f"✅ Git clone successful: {model_dir}")
                return True
            else:
                print(f"⚠️ Git clone incomplete, missing model files")
        else:
            print(f"❌ Git failed: {result.stderr}")

    except Exception as e:
        print(f"❌ Git error: {e}")

    # Method 3: Manual instructions
    print("\n" + "="*60)
    print("📋 MANUAL DOWNLOAD REQUIRED")
    print("="*60)
    print("Automatic download failed. Please download manually:")
    print()
    print("Option 1 - Download on local machine, then upload:")
    print("  Local: git clone https://huggingface.co/" + repo_id)
    print("  Upload to server using FTP/SFTP")
    print()
    print("Option 2 - Use smaller model:")
    print("  python main.py server --huggingface openai/whisper-small")
    print()
    print("Option 3 - Use wget (run these commands):")
    target_dir = f"models/huggingface/{repo_id.replace('/', '_')}"
    print(f"  mkdir -p {target_dir}")
    print(f"  cd {target_dir}")
    print(f"  wget -c -T 3600 --tries=10 'https://huggingface.co/{repo_id}/resolve/main/config.json'")
    print(f"  wget -c -T 3600 --tries=10 'https://huggingface.co/{repo_id}/resolve/main/pytorch_model.bin'")
    print("="*60)

    return False

def verify_model_download(model_path):
    """Verify that all essential model files are present and not empty."""
    model_dir = Path(model_path)

    # Essential files that must exist
    essential_files = ["config.json", "tokenizer_config.json"]

    # Model weight files (at least one must exist and be > 1MB)
    weight_files = ["pytorch_model.bin", "model.safetensors", "pytorch_model.safetensors"]

    # Check essential files
    for file in essential_files:
        file_path = model_dir / file
        if not file_path.exists() or file_path.stat().st_size == 0:
            print(f"❌ Missing or empty essential file: {file}")
            return False

    # Check for at least one valid weight file
    weight_found = False
    for weight_file in weight_files:
        weight_path = model_dir / weight_file
        if weight_path.exists() and weight_path.stat().st_size > 1024 * 1024:  # > 1MB
            print(f"✅ Found model weights: {weight_file} ({weight_path.stat().st_size / (1024*1024):.1f} MB)")
            weight_found = True
            break

    if not weight_found:
        print("❌ No valid model weight files found!")
        print("   This usually means Git LFS files weren't downloaded.")
        return False

    print("✅ Model download verification passed")
    return True

if __name__ == "__main__":
    repo_id = sys.argv[1] if len(sys.argv) > 1 else "TwiWhisperModel/TwiWhisperModel"
    success = download_with_retry(repo_id)
    sys.exit(0 if success else 1)
EOF

    chmod +x download_model_safe.py
    log_success "Enhanced download script created"
}

# Function to create wget download script
create_wget_script() {
    log_info "Creating wget fallback script..."

    cat > download_with_wget.sh << 'EOF'
#!/bin/bash

# Manual download using wget with resume capability
# Usage: bash download_with_wget.sh [repo_id]

REPO_ID=${1:-"TwiWhisperModel/TwiWhisperModel"}
MODEL_DIR="models/huggingface/${REPO_ID//\//_}"

echo "📁 Creating directory: $MODEL_DIR"
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

echo "📦 Downloading model files for $REPO_ID..."

# Essential files to download
FILES=(
    "config.json"
    "generation_config.json"
    "tokenizer_config.json"
    "vocab.json"
    "merges.txt"
    "normalizer.json"
    "added_tokens.json"
    "special_tokens_map.json"
    "preprocessor_config.json"
    "pytorch_model.bin"
    "model.safetensors"
    "pytorch_model.safetensors"
)

BASE_URL="https://huggingface.co/$REPO_ID/resolve/main"

for file in "${FILES[@]}"; do
    echo "⬇️ Downloading $file..."
    if wget -c -T 3600 --tries=10 --retry-connrefused --waitretry=60 "$BASE_URL/$file"; then
        # Verify file size (model weights should be > 1MB)
        if [[ "$file" == *".bin" ]] || [[ "$file" == *".safetensors" ]]; then
            size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
            if [ "$size" -gt 1048576 ]; then
                echo "✅ Downloaded $file ($(($size / 1048576)) MB)"
            else
                echo "⚠️ $file may be incomplete (${size} bytes)"
            fi
        else
            echo "✅ Downloaded $file"
        fi
    else
        echo "⚠️ Failed to download $file"
    fi
done

echo "✅ Download attempt completed!"
echo "📁 Files saved to: $(pwd)"
echo ""
echo "🔍 Verifying download completeness..."

# Verify essential files exist
missing_files=()
for essential in "config.json" "tokenizer_config.json"; do
    if [ ! -f "$essential" ] || [ ! -s "$essential" ]; then
        missing_files+=("$essential")
    fi
done

# Check for model weights
weight_found=false
for weight in "pytorch_model.bin" "model.safetensors" "pytorch_model.safetensors"; do
    if [ -f "$weight" ] && [ -s "$weight" ]; then
        size=$(stat -f%z "$weight" 2>/dev/null || stat -c%s "$weight" 2>/dev/null || echo "0")
        if [ "$size" -gt 1048576 ]; then
            weight_found=true
            echo "✅ Model weights found: $weight ($(($size / 1048576)) MB)"
            break
        fi
    fi
done

if [ ${#missing_files[@]} -eq 0 ] && [ "$weight_found" = true ]; then
    echo "🎉 Download verification PASSED - model is complete!"
else
    echo "❌ Download verification FAILED:"
    [ ${#missing_files[@]} -gt 0 ] && echo "   Missing files: ${missing_files[*]}"
    [ "$weight_found" = false ] && echo "   No valid model weights found"
    echo "   You may need to re-download missing files"
fi
EOF

    chmod +x download_with_wget.sh
    log_success "Wget download script created"
}

# Main execution
main() {
    echo "Starting cPanel timeout fixes..."
    echo

    # Apply fixes
    fix_environment
    fix_pip_config
    fix_git_config
    create_cache_dirs
    create_download_script
    create_wget_script

    echo
    log_info "Testing network connectivity..."
    test_connectivity

    echo
    echo "================================================================"
    log_success "CPANEL TIMEOUT FIXES APPLIED!"
    echo "================================================================"

    echo
    echo "📋 NEXT STEPS:"
    echo "1. Restart your terminal or run: source ~/.bashrc"
    echo "2. Try downloading your model:"
    echo "   python download_model_safe.py TwiWhisperModel/TwiWhisperModel"
    echo "3. Or use the server directly:"
    echo "   python main.py server --huggingface TwiWhisperModel/TwiWhisperModel"
    echo "4. If still failing, use manual wget method:"
    echo "   bash download_with_wget.sh TwiWhisperModel/TwiWhisperModel"
    echo

    echo "💡 ALTERNATIVE FASTER OPTIONS:"
    echo "   Use smaller models that download faster:"
    echo "   python main.py server --huggingface openai/whisper-tiny"
    echo "   python main.py server --huggingface openai/whisper-base"
    echo "   python main.py server --huggingface openai/whisper-small"
    echo

    log_info "All fixes applied successfully! Check above for next steps."
}

# Run main function
main "$@"
