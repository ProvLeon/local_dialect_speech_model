#!/bin/bash
set -e

# Render Deployment Script for Twi Speech Model
# This script helps prepare and deploy the model to Render.com

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_banner() {
    echo "================================================================"
    echo "           Twi Speech Model - Render Deployment Helper"
    echo "================================================================"
    echo ""
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking deployment prerequisites..."

    local missing_deps=()

    # Check git
    if ! command_exists git; then
        missing_deps+=("git")
    fi

    # Check render CLI (optional but helpful)
    if ! command_exists render; then
        print_warning "Render CLI not found. You can install it with: npm install -g @render/cli"
    fi

    # Check if we're in a git repository
    if [ ! -d ".git" ]; then
        print_error "This directory is not a git repository. Please run 'git init' first."
        exit 1
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing required dependencies: ${missing_deps[*]}"
        exit 1
    fi

    print_success "Prerequisites check passed"
}

# Function to validate project structure
validate_project() {
    print_info "Validating project structure..."

    local required_files=(
        "deployable_twi_speech_model/model/model_state_dict.bin"
        "deployable_twi_speech_model/utils/serve.py"
        "deployable_twi_speech_model/requirements.txt"
        "Dockerfile"
        "render.yaml"
    )

    local missing_files=()

    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            missing_files+=("$file")
        else
            print_info "✓ Found: $file"
        fi
    done

    if [ ${#missing_files[@]} -ne 0 ]; then
        print_error "Missing required files: ${missing_files[*]}"
        return 1
    fi

    print_success "Project structure validation passed"
}

# Function to check git status
check_git_status() {
    print_info "Checking git status..."

    # Check if there are uncommitted changes
    if ! git diff-index --quiet HEAD --; then
        print_warning "You have uncommitted changes. Consider committing them before deployment."
        git status --short
        echo ""
        read -p "Do you want to continue anyway? (y/N): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Deployment cancelled by user"
            exit 0
        fi
    fi

    # Get current branch
    local current_branch=$(git branch --show-current)
    print_info "Current branch: $current_branch"

    # Check if main/master branch exists and suggest using it
    if [[ "$current_branch" != "main" && "$current_branch" != "master" ]]; then
        if git show-ref --verify --quiet refs/heads/main; then
            print_warning "You're not on the 'main' branch. Render typically deploys from 'main'."
        elif git show-ref --verify --quiet refs/heads/master; then
            print_warning "You're not on the 'master' branch. Render typically deploys from 'master'."
        fi
    fi
}

# Function to prepare for deployment
prepare_deployment() {
    print_info "Preparing project for Render deployment..."

    # Create .dockerignore if it doesn't exist
    if [ ! -f ".dockerignore" ]; then
        print_info "Creating .dockerignore file..."
        cat > .dockerignore << 'EOF'
# Git
.git
.gitignore

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
*.egg-info/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Temporary files
temp/
tmp/
*.tmp

# Large model files not in deployable package
data/
models/
results/
dist/

# Development files
tests/
test_*.py
*_test.py
EOF
        print_success "Created .dockerignore file"
    fi

    # Ensure render.yaml is properly configured
    if [ -f "render.yaml" ]; then
        print_success "Found render.yaml configuration"
    else
        print_error "render.yaml not found. This file is required for Render deployment."
        return 1
    fi

    print_success "Project prepared for deployment"
}

# Function to test Docker build locally
test_docker_build() {
    print_info "Testing Docker build locally (optional)..."

    read -p "Do you want to test the Docker build locally? This may take a few minutes. (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Building Docker image locally..."

        if docker build -t twi-speech-test . --no-cache; then
            print_success "Docker build succeeded!"

            read -p "Do you want to test the container locally? (y/N): " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                print_info "Starting test container on port 8080..."
                docker run -d --name twi-speech-test-container -p 8080:8000 twi-speech-test

                print_info "Waiting for container to start..."
                sleep 10

                if curl -f "http://localhost:8080/health" >/dev/null 2>&1; then
                    print_success "Test container is running successfully!"
                    print_info "You can test it at: http://localhost:8080"
                    print_info "API docs at: http://localhost:8080/docs"
                else
                    print_error "Test container failed to start properly"
                fi

                read -p "Press Enter to stop and remove the test container..."
                docker stop twi-speech-test-container >/dev/null 2>&1
                docker rm twi-speech-test-container >/dev/null 2>&1
                docker rmi twi-speech-test >/dev/null 2>&1
                print_info "Test container cleaned up"
            fi
        else
            print_error "Docker build failed. Please fix the issues before deploying to Render."
            return 1
        fi
    fi
}

# Function to show deployment instructions
show_deployment_instructions() {
    echo ""
    print_success "🎉 Your project is ready for Render deployment!"
    echo ""
    print_info "Next steps:"
    echo ""
    echo "1. COMMIT AND PUSH your code to GitHub:"
    echo "   git add ."
    echo "   git commit -m 'Prepare for Render deployment'"
    echo "   git push origin main"
    echo ""
    echo "2. GO TO RENDER DASHBOARD:"
    echo "   • Visit: https://dashboard.render.com"
    echo "   • Sign in with your GitHub account"
    echo ""
    echo "3. CREATE NEW WEB SERVICE:"
    echo "   • Click 'New +' → 'Web Service'"
    echo "   • Connect your GitHub repository"
    echo "   • Select this repository"
    echo ""
    echo "4. CONFIGURE DEPLOYMENT:"
    echo "   • Render will auto-detect your render.yaml file"
    echo "   • Review the configuration"
    echo "   • Click 'Create Web Service'"
    echo ""
    echo "5. MONITOR DEPLOYMENT:"
    echo "   • Watch the build logs in Render dashboard"
    echo "   • Deployment usually takes 5-10 minutes"
    echo "   • Your API will be available at: https://YOUR-SERVICE-NAME.onrender.com"
    echo ""
    print_info "Important endpoints after deployment:"
    echo "   • Health Check: https://YOUR-SERVICE-NAME.onrender.com/health"
    echo "   • API Documentation: https://YOUR-SERVICE-NAME.onrender.com/docs"
    echo "   • Model Info: https://YOUR-SERVICE-NAME.onrender.com/model-info"
    echo ""
    print_warning "Note: Free tier services on Render may take 30+ seconds to wake up from sleep."
    echo ""
}

# Function to show render CLI usage
show_render_cli_usage() {
    if command_exists render; then
        echo ""
        print_info "You have Render CLI installed. You can also deploy using:"
        echo ""
        echo "1. Login to Render:"
        echo "   render auth login"
        echo ""
        echo "2. Deploy from CLI:"
        echo "   render deploy"
        echo ""
        print_info "This will use your render.yaml configuration automatically."
    fi
}

# Function to troubleshoot common issues
show_troubleshooting() {
    echo ""
    print_info "Common deployment issues and solutions:"
    echo ""
    echo "🔧 BUILD FAILURES:"
    echo "   • Check that all files in deployable_twi_speech_model/ are committed"
    echo "   • Verify requirements.txt has all necessary dependencies"
    echo "   • Test Docker build locally first"
    echo ""
    echo "🔧 SERVICE WON'T START:"
    echo "   • Check Render logs for Python errors"
    echo "   • Verify model files are present in deployable_twi_speech_model/"
    echo "   • Ensure port 8000 is exposed in Dockerfile"
    echo ""
    echo "🔧 HEALTH CHECK FAILS:"
    echo "   • Make sure /health endpoint exists in serve.py"
    echo "   • Check that the service starts within 10 minutes"
    echo "   • Verify no blocking operations in startup code"
    echo ""
    echo "🔧 OUT OF MEMORY:"
    echo "   • Consider upgrading to a paid plan with more RAM"
    echo "   • Optimize model loading (lazy loading)"
    echo "   • Use smaller model files if possible"
    echo ""
}

# Main execution
main() {
    print_banner

    case "${1:-deploy}" in
        "prepare"|"prep")
            check_prerequisites
            validate_project || exit 1
            prepare_deployment || exit 1
            print_success "Project preparation completed!"
            ;;

        "validate"|"check")
            check_prerequisites
            validate_project || exit 1
            check_git_status
            print_success "Validation completed successfully!"
            ;;

        "test"|"test-docker")
            check_prerequisites
            validate_project || exit 1
            test_docker_build
            ;;

        "deploy"|"")
            check_prerequisites
            validate_project || exit 1
            check_git_status
            prepare_deployment || exit 1
            test_docker_build
            show_deployment_instructions
            show_render_cli_usage
            ;;

        "instructions"|"help")
            show_deployment_instructions
            show_render_cli_usage
            show_troubleshooting
            ;;

        "troubleshoot"|"debug")
            show_troubleshooting
            ;;

        *)
            print_error "Unknown command: $1"
            echo ""
            echo "Usage: $0 [COMMAND]"
            echo ""
            echo "Commands:"
            echo "  deploy       Full deployment preparation (default)"
            echo "  prepare      Prepare project files only"
            echo "  validate     Validate project structure"
            echo "  test         Test Docker build locally"
            echo "  instructions Show deployment instructions"
            echo "  troubleshoot Show troubleshooting guide"
            echo ""
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
