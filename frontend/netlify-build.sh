#!/bin/bash

# Netlify Production Build Script with Memory Optimization and Fallbacks
# This script tries multiple build approaches to handle memory constraints

set -e  # Exit on any error

echo "🚀 Starting Netlify Production Build"
echo "====================================="

# Check available memory
echo "📊 System Information:"
free -h 2>/dev/null || echo "Memory info not available"
echo "Node.js version: $(node --version 2>/dev/null || echo 'Not available')"
echo "Deno version: $(deno --version 2>/dev/null | head -1 || echo 'Not available')"
echo "Working directory: $(pwd)"

# Set memory limits
export NODE_OPTIONS="--max-old-space-size=1024"
export DENO_V8_FLAGS="--max-old-space-size=1024,--optimize-for-size,--gc-interval=50"

# Function to cleanup on exit
cleanup() {
    echo "🧹 Cleaning up..."
    # Kill any remaining processes
    pkill -f "next build" 2>/dev/null || true
    pkill -f "deno run" 2>/dev/null || true
}
trap cleanup EXIT

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to build with timeout
build_with_timeout() {
    local cmd="$1"
    local timeout_seconds=900  # 15 minutes

    echo "⏱️  Running: $cmd (timeout: ${timeout_seconds}s)"

    timeout $timeout_seconds bash -c "$cmd" || {
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo "❌ Build timed out after ${timeout_seconds} seconds"
        else
            echo "❌ Build failed with exit code: $exit_code"
        fi
        return $exit_code
    }
}

# Function to check build output
check_build_output() {
    if [ -d ".next" ] && [ "$(ls -A .next 2>/dev/null)" ]; then
        echo "✅ Build output verified (.next directory exists and not empty)"
        du -sh .next 2>/dev/null || echo "Build size calculation not available"
        return 0
    else
        echo "❌ Build output missing or empty"
        return 1
    fi
}

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf .next out dist build 2>/dev/null || true

# Build Attempt 1: Ultra-light Deno build
if command_exists deno; then
    echo ""
    echo "🎯 ATTEMPT 1: Ultra-light Deno build"
    echo "-------------------------------------"

    if build_with_timeout "deno task build:ultra-light"; then
        if check_build_output; then
            echo "🎉 Success with ultra-light Deno build!"
            exit 0
        fi
    fi

    echo "⚠️  Ultra-light Deno build failed, trying memory-optimized..."
    rm -rf .next 2>/dev/null || true
fi

# Build Attempt 2: Memory-optimized Deno build
if command_exists deno; then
    echo ""
    echo "🎯 ATTEMPT 2: Memory-optimized Deno build"
    echo "-------------------------------------------"

    if build_with_timeout "deno task build:memory-optimized"; then
        if check_build_output; then
            echo "🎉 Success with memory-optimized Deno build!"
            exit 0
        fi
    fi

    echo "⚠️  Memory-optimized Deno build failed, trying standard..."
    rm -rf .next 2>/dev/null || true
fi

# Build Attempt 3: Standard Deno build
if command_exists deno; then
    echo ""
    echo "🎯 ATTEMPT 3: Standard Deno build"
    echo "---------------------------------"

    if build_with_timeout "deno task build"; then
        if check_build_output; then
            echo "🎉 Success with standard Deno build!"
            exit 0
        fi
    fi

    echo "⚠️  All Deno builds failed, falling back to npm..."
    rm -rf .next 2>/dev/null || true
fi

# Build Attempt 4: npm build with memory optimization
if command_exists npm; then
    echo ""
    echo "🎯 ATTEMPT 4: npm build (fallback)"
    echo "----------------------------------"

    # Install dependencies if needed
    if [ ! -d "node_modules" ] || [ -z "$(ls -A node_modules 2>/dev/null)" ]; then
        echo "📦 Installing dependencies..."
        export NODE_OPTIONS="--max-old-space-size=512"
        npm ci --no-audit --no-fund --silent || npm install --no-audit --no-fund --silent
    fi

    # Build with npm
    export NODE_OPTIONS="--max-old-space-size=1024"
    if build_with_timeout "npm run build"; then
        if check_build_output; then
            echo "🎉 Success with npm build!"
            exit 0
        fi
    fi

    echo "⚠️  npm build failed, trying with even less memory..."
    rm -rf .next 2>/dev/null || true
fi

# Build Attempt 5: Minimal npm build
if command_exists npm; then
    echo ""
    echo "🎯 ATTEMPT 5: Minimal npm build"
    echo "-------------------------------"

    export NODE_OPTIONS="--max-old-space-size=768"
    if build_with_timeout "npm run build"; then
        if check_build_output; then
            echo "🎉 Success with minimal npm build!"
            exit 0
        fi
    fi
fi

# Build Attempt 6: Static generation only
echo ""
echo "🎯 ATTEMPT 6: Static generation only"
echo "------------------------------------"

# Create minimal static build
mkdir -p .next
echo '{"version": 3, "routes": []}' > .next/routes-manifest.json

# Try to generate static files manually
if command_exists npm; then
    npm run build 2>/dev/null || true
fi

# Check if we have any output
if [ -d ".next" ]; then
    echo "⚠️  Partial build available"
    ls -la .next/ 2>/dev/null || true
else
    echo "❌ Creating emergency static build..."
    mkdir -p .next/static
    echo '<!DOCTYPE html><html><head><title>Build Error</title></head><body><h1>Build in progress...</h1></body></html>' > .next/index.html
fi

echo ""
echo "💥 ALL BUILD ATTEMPTS FAILED"
echo "============================="
echo "This appears to be a memory/resource constraint issue on the build environment."
echo ""
echo "💡 Recommended solutions:"
echo "1. Upgrade Netlify build plan for more memory"
echo "2. Simplify the application to reduce memory usage"
echo "3. Use a different deployment platform with more resources"
echo "4. Build locally and deploy static files"
echo ""
echo "📊 Build attempts made:"
echo "   ✗ Ultra-light Deno build"
echo "   ✗ Memory-optimized Deno build"
echo "   ✗ Standard Deno build"
echo "   ✗ npm build"
echo "   ✗ Minimal npm build"
echo "   ✗ Static generation"

exit 1
