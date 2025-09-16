#!/bin/bash

# Universal build script for the frontend
# Works with Node.js, npm, and various deployment environments

set -e  # Exit on any error

echo "🚀 Starting Frontend Build Process..."
echo "====================================="

# Check if we're in the frontend directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: package.json not found. Please run this script from the frontend directory."
    exit 1
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect the available package manager and Node.js runtime
echo "🔍 Detecting environment..."

if command_exists npm; then
    echo "✅ npm found"
    PACKAGE_MANAGER="npm"
elif command_exists yarn; then
    echo "✅ yarn found"
    PACKAGE_MANAGER="yarn"
elif command_exists pnpm; then
    echo "✅ pnpm found"
    PACKAGE_MANAGER="pnpm"
else
    echo "❌ No package manager found. Please install npm, yarn, or pnpm."
    exit 1
fi

# Check Node.js version
if command_exists node; then
    NODE_VERSION=$(node --version)
    echo "✅ Node.js version: $NODE_VERSION"
else
    echo "❌ Node.js not found. Please install Node.js."
    exit 1
fi

# Install dependencies if node_modules doesn't exist or is empty
if [ ! -d "node_modules" ] || [ -z "$(ls -A node_modules 2>/dev/null)" ]; then
    echo "📦 Installing dependencies..."
    case $PACKAGE_MANAGER in
        npm)
            npm install
            ;;
        yarn)
            yarn install
            ;;
        pnpm)
            pnpm install
            ;;
    esac
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed"
fi

# Clean previous build
echo "🧹 Cleaning previous build..."
rm -rf .next
rm -rf out
rm -rf dist
echo "✅ Clean completed"

# Run linting (optional, continue even if it fails)
echo "🔍 Running linter..."
case $PACKAGE_MANAGER in
    npm)
        npm run lint || echo "⚠️  Linting failed, but continuing with build..."
        ;;
    yarn)
        yarn lint || echo "⚠️  Linting failed, but continuing with build..."
        ;;
    pnpm)
        pnpm run lint || echo "⚠️  Linting failed, but continuing with build..."
        ;;
esac

# Run the build
echo "🏗️  Building application..."
case $PACKAGE_MANAGER in
    npm)
        npm run build
        ;;
    yarn)
        yarn build
        ;;
    pnpm)
        pnpm run build
        ;;
esac

# Verify build output
if [ -d ".next" ]; then
    echo "✅ Build completed successfully!"
    echo "📊 Build output:"
    du -sh .next 2>/dev/null || echo "Build size calculation unavailable"

    # List important build files
    if [ -d ".next/static" ]; then
        echo "📁 Static assets generated"
    fi

    if [ -f ".next/BUILD_ID" ]; then
        BUILD_ID=$(cat .next/BUILD_ID)
        echo "🆔 Build ID: $BUILD_ID"
    fi

else
    echo "❌ Build failed - .next directory not found"
    exit 1
fi

# Environment-specific instructions
echo ""
echo "🚀 Deployment Instructions:"
echo "============================="

if [ -n "$NETLIFY" ]; then
    echo "📡 Netlify detected"
    echo "   - Build output: .next"
    echo "   - Publish directory should be set to .next"
elif [ -n "$VERCEL" ]; then
    echo "▲ Vercel detected"
    echo "   - Build output: .next"
    echo "   - Framework preset: Next.js"
elif [ -n "$RENDER" ]; then
    echo "🎨 Render detected"
    echo "   - Build output: .next"
    echo "   - Start command: npm start"
else
    echo "🌐 Generic deployment"
    echo "   - Build output: .next"
    echo "   - Start command: npm start"
    echo "   - Port: 3000 (configurable via PORT env var)"
fi

echo ""
echo "✅ Build process completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Deploy the .next directory"
echo "   2. Set environment variables if needed"
echo "   3. Configure your deployment platform to serve the application"
echo ""
echo "🔗 Useful commands:"
echo "   - Test locally: $PACKAGE_MANAGER start"
echo "   - Run dev server: $PACKAGE_MANAGER run dev"
echo "   - Run linter: $PACKAGE_MANAGER run lint"
