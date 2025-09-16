#!/bin/bash
set -e

# Comprehensive Project Setup Script for Twi Speech Model
# This script sets up the entire development environment from scratch

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="local_dialect_speech_model"
PYTHON_VERSION="3.9"
NODE_VERSION="18"
CONDA_ENV_NAME="twi_speech"

# Function to print colored output
print_header() {
    echo -e "${PURPLE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${PURPLE}║                                                              ║${NC}"
    echo -e "${PURPLE}║           Twi Speech Model - Project Setup                  ║${NC}"
    echo -e "${PURPLE}║                                                              ║${NC}"
    echo -e "${PURPLE}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_section() {
    echo -e "${CYAN}━━━ $1 ━━━${NC}"
}

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

print_step() {
    echo -e "${GREEN}➤${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Function to install system dependencies based on OS
install_system_dependencies() {
    print_section "Installing System Dependencies"

    local os=$(detect_os)
    print_info "Detected OS: $os"

    case $os in
        "linux")
            print_step "Installing Linux dependencies..."
            if command_exists apt-get; then
                sudo apt-get update
                sudo apt-get install -y \
                    python3 python3-pip python3-venv python3-dev \
                    build-essential \
                    libsndfile1 libsndfile1-dev \
                    ffmpeg \
                    portaudio19-dev \
                    git \
                    curl \
                    wget \
                    docker.io \
                    docker-compose

                # Add user to docker group
                sudo usermod -aG docker $USER
                print_warning "You may need to log out and back in for Docker permissions to take effect"

            elif command_exists yum; then
                sudo yum update -y
                sudo yum install -y \
                    python3 python3-pip python3-devel \
                    gcc gcc-c++ make \
                    libsndfile-devel \
                    ffmpeg \
                    portaudio-devel \
                    git \
                    curl \
                    wget \
                    docker \
                    docker-compose

                sudo systemctl enable docker
                sudo systemctl start docker
                sudo usermod -aG docker $USER
            fi
            ;;

        "macos")
            print_step "Installing macOS dependencies..."
            if ! command_exists brew; then
                print_info "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi

            brew update
            brew install \
                python@3.9 \
                libsndfile \
                ffmpeg \
                portaudio \
                git \
                curl \
                wget \
                docker \
                docker-compose \
                node@18

            # Start Docker Desktop if not running
            if ! docker info >/dev/null 2>&1; then
                print_info "Please start Docker Desktop manually"
                open -a Docker
                print_info "Waiting for Docker to start..."
                sleep 10
            fi
            ;;

        "windows")
            print_warning "Windows detected. Please install the following manually:"
            echo "  - Python 3.9+ from python.org"
            echo "  - Git from git-scm.com"
            echo "  - Docker Desktop from docker.com"
            echo "  - Node.js 18+ from nodejs.org"
            echo "  - FFmpeg from ffmpeg.org"
            print_info "Or use Windows Subsystem for Linux (WSL) for better compatibility"
            ;;

        *)
            print_error "Unsupported operating system: $os"
            print_info "Please install dependencies manually:"
            echo "  - Python 3.9+"
            echo "  - Git"
            echo "  - Docker"
            echo "  - FFmpeg"
            echo "  - Node.js 18+"
            ;;
    esac

    print_success "System dependencies installation completed"
}

# Function to setup Python environment
setup_python_environment() {
    print_section "Setting up Python Environment"

    # Check Python version
    if command_exists python3; then
        python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
        print_info "Python version: $python_version"
    else
        print_error "Python 3 not found. Please install Python 3.8 or higher."
        exit 1
    fi

    # Setup conda environment if conda is available
    if command_exists conda; then
        print_step "Setting up Conda environment..."
        conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y || true
        print_info "Conda environment '$CONDA_ENV_NAME' created"
        print_info "To activate: conda activate $CONDA_ENV_NAME"
    else
        print_step "Setting up virtual environment..."
        if [ ! -d "venv" ]; then
            python3 -m venv venv
            print_success "Virtual environment created"
        else
            print_info "Virtual environment already exists"
        fi

        # Activate virtual environment
        source venv/bin/activate
        print_info "Virtual environment activated"
    fi

    # Upgrade pip
    print_step "Upgrading pip..."
    python -m pip install --upgrade pip

    # Install core dependencies
    print_step "Installing core Python dependencies..."
    pip install \
        torch torchvision torchaudio \
        numpy pandas scipy scikit-learn \
        librosa soundfile \
        fastapi uvicorn python-multipart \
        pydantic \
        pytest pytest-cov \
        jupyter notebook \
        matplotlib seaborn \
        tqdm \
        python-dotenv \
        pyyaml

    # Install requirements if file exists
    if [ -f "requirements.txt" ]; then
        print_step "Installing project requirements..."
        pip install -r requirements.txt
    fi

    # Install development dependencies
    if [ -f "requirements_dev.txt" ]; then
        print_step "Installing development requirements..."
        pip install -r requirements_dev.txt
    else
        print_step "Installing development tools..."
        pip install \
            black isort flake8 mypy \
            pre-commit \
            bandit safety \
            sphinx sphinx-rtd-theme
    fi

    print_success "Python environment setup completed"
}

# Function to setup Node.js environment for frontend
setup_frontend_environment() {
    print_section "Setting up Frontend Environment"

    if [ -d "frontend" ]; then
        print_step "Setting up Next.js frontend..."
        cd frontend

        # Check if Node.js is available
        if command_exists node; then
            node_version=$(node --version)
            print_info "Node.js version: $node_version"
        else
            print_warning "Node.js not found. Frontend setup skipped."
            cd ..
            return
        fi

        # Install frontend dependencies
        if [ -f "package.json" ]; then
            print_step "Installing frontend dependencies..."
            npm install
            print_success "Frontend dependencies installed"
        fi

        cd ..
    else
        print_info "No frontend directory found, skipping frontend setup"
    fi
}

# Function to setup Git hooks
setup_git_hooks() {
    print_section "Setting up Git Hooks"

    if [ -d ".git" ]; then
        print_step "Installing pre-commit hooks..."

        # Install pre-commit if available
        if command_exists pre-commit; then
            pre-commit install
            print_success "Pre-commit hooks installed"
        else
            print_info "Pre-commit not available, skipping hooks setup"
        fi

        # Create basic git hooks if pre-commit is not available
        if [ ! -f ".git/hooks/pre-commit" ]; then
            print_step "Creating basic pre-commit hook..."
            cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Basic pre-commit hook for code quality

echo "Running pre-commit checks..."

# Check Python code formatting
if command -v black >/dev/null 2>&1; then
    echo "Checking Python code formatting..."
    black --check . || {
        echo "Code formatting issues found. Run 'black .' to fix."
        exit 1
    }
fi

# Check imports
if command -v isort >/dev/null 2>&1; then
    echo "Checking import sorting..."
    isort --check-only . || {
        echo "Import sorting issues found. Run 'isort .' to fix."
        exit 1
    }
fi

# Run basic linting
if command -v flake8 >/dev/null 2>&1; then
    echo "Running basic linting..."
    flake8 . --select=E9,F63,F7,F82 --show-source
fi

echo "Pre-commit checks passed!"
EOF
            chmod +x .git/hooks/pre-commit
            print_success "Basic pre-commit hook created"
        fi
    else
        print_info "Not a git repository, skipping git hooks setup"
    fi
}

# Function to setup Docker environment
setup_docker_environment() {
    print_section "Setting up Docker Environment"

    if command_exists docker; then
        # Check if Docker is running
        if ! docker info >/dev/null 2>&1; then
            print_warning "Docker daemon is not running. Please start Docker."
            return
        fi

        print_step "Checking Docker installation..."
        docker_version=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
        print_info "Docker version: $docker_version"

        # Create .dockerignore if it doesn't exist
        if [ ! -f ".dockerignore" ]; then
            print_step "Creating .dockerignore file..."
            cat > .dockerignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# Jupyter Notebook
.ipynb_checkpoints

# Data and models (exclude large files)
data/raw/
data/processed/
*.pkl
*.model
*.h5
*.pb

# Logs
logs/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Git
.git/
.gitignore

# Docker
Dockerfile*
docker-compose*

# Documentation
docs/_build/

# Temporary files
tmp/
temp/
*.tmp

# Node.js (if frontend exists)
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Build artifacts
dist/
build/
*.egg-info/
EOF
            print_success ".dockerignore created"
        fi

        print_success "Docker environment setup completed"
    else
        print_warning "Docker not found. Some deployment features will not be available."
    fi
}

# Function to setup project directories
setup_project_structure() {
    print_section "Setting up Project Structure"

    # Create necessary directories
    local dirs=(
        "data/raw"
        "data/processed"
        "data/external"
        "data/interim"
        "logs"
        "notebooks"
        "reports"
        "models"
        "scripts"
        "tests/unit"
        "tests/integration"
        "tests/samples"
        "docs"
        "config"
    )

    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_step "Created directory: $dir"
        fi
    done

    # Create .gitkeep files for empty directories
    for dir in "${dirs[@]}"; do
        if [ ! -f "$dir/.gitkeep" ] && [ -z "$(ls -A $dir 2>/dev/null)" ]; then
            touch "$dir/.gitkeep"
        fi
    done

    print_success "Project structure setup completed"
}

# Function to setup environment configuration
setup_environment_config() {
    print_section "Setting up Environment Configuration"

    # Create .env.example file
    if [ ! -f ".env.example" ]; then
        print_step "Creating .env.example file..."
        cat > .env.example << 'EOF'
# Environment Configuration
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Model Configuration
MODEL_PATH=deployable_twi_speech_model/
ENABLE_CUDA=false
BATCH_SIZE=1
MAX_AUDIO_LENGTH=30

# Database Configuration (if needed)
# DATABASE_URL=sqlite:///./app.db

# Security Configuration
SECRET_KEY=your-secret-key-here
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080

# Monitoring Configuration
ENABLE_METRICS=true
METRICS_PORT=9090

# External Services (if needed)
# AWS_ACCESS_KEY_ID=your-aws-access-key
# AWS_SECRET_ACCESS_KEY=your-aws-secret-key
# AWS_REGION=us-west-2
EOF
        print_success ".env.example created"
    fi

    # Create local .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        print_step "Creating local .env file..."
        cp .env.example .env
        print_info "Please edit .env file with your specific configuration"
    fi

    print_success "Environment configuration setup completed"
}

# Function to validate installation
validate_installation() {
    print_section "Validating Installation"

    local errors=()

    # Check Python
    if command_exists python3; then
        print_step "✓ Python 3 installed"
    else
        errors+=("Python 3 not found")
    fi

    # Check pip
    if command_exists pip; then
        print_step "✓ pip installed"
    else
        errors+=("pip not found")
    fi

    # Check Git
    if command_exists git; then
        print_step "✓ Git installed"
    else
        errors+=("Git not found")
    fi

    # Check Docker
    if command_exists docker; then
        if docker info >/dev/null 2>&1; then
            print_step "✓ Docker installed and running"
        else
            print_warning "⚠ Docker installed but not running"
        fi
    else
        print_warning "⚠ Docker not found (optional)"
    fi

    # Check Node.js (if frontend exists)
    if [ -d "frontend" ]; then
        if command_exists node; then
            print_step "✓ Node.js installed"
        else
            errors+=("Node.js not found (required for frontend)")
        fi
    fi

    # Check Python packages
    if python3 -c "import torch, numpy, librosa, fastapi" >/dev/null 2>&1; then
        print_step "✓ Core Python packages installed"
    else
        errors+=("Some core Python packages missing")
    fi

    # Check model files
    if [ -f "deployable_twi_speech_model/model/model_state_dict.bin" ]; then
        print_step "✓ Model files found"
    else
        print_warning "⚠ Model files not found (you may need to train or download the model)"
    fi

    if [ ${#errors[@]} -eq 0 ]; then
        print_success "All validations passed!"
    else
        print_error "Validation errors found:"
        for error in "${errors[@]}"; do
            echo "  - $error"
        done
        return 1
    fi
}

# Function to show next steps
show_next_steps() {
    print_section "Next Steps"

    echo -e "${GREEN}🎉 Setup completed successfully!${NC}"
    echo ""
    echo -e "${CYAN}Next steps:${NC}"
    echo ""
    echo -e "${YELLOW}1.${NC} Activate your environment:"
    if command_exists conda && conda env list | grep -q $CONDA_ENV_NAME; then
        echo -e "   ${BLUE}conda activate $CONDA_ENV_NAME${NC}"
    else
        echo -e "   ${BLUE}source venv/bin/activate${NC}"
    fi
    echo ""

    echo -e "${YELLOW}2.${NC} Configure your environment:"
    echo -e "   ${BLUE}cp .env.example .env${NC}"
    echo -e "   ${BLUE}nano .env${NC}  # Edit with your settings"
    echo ""

    echo -e "${YELLOW}3.${NC} Test the installation:"
    echo -e "   ${BLUE}./deploy.sh status${NC}"
    echo ""

    echo -e "${YELLOW}4.${NC} Deploy the model:"
    echo -e "   ${BLUE}./deploy.sh --quick${NC}  # Quick deployment"
    echo -e "   ${BLUE}./deploy.sh deploy --environment development${NC}  # Full deployment"
    echo ""

    echo -e "${YELLOW}5.${NC} Access the API:"
    echo -e "   ${BLUE}http://localhost:8000/docs${NC}  # API documentation"
    echo -e "   ${BLUE}http://localhost:8000/health${NC}  # Health check"
    echo ""

    if [ -d "frontend" ]; then
        echo -e "${YELLOW}6.${NC} Start the frontend (optional):"
        echo -e "   ${BLUE}cd frontend && npm run dev${NC}"
        echo ""
    fi

    echo -e "${YELLOW}7.${NC} Development workflow:"
    echo -e "   ${BLUE}jupyter notebook${NC}  # For data exploration"
    echo -e "   ${BLUE}python -m pytest tests/${NC}  # Run tests"
    echo -e "   ${BLUE}black . && isort .${NC}  # Format code"
    echo ""

    echo -e "${CYAN}Useful commands:${NC}"
    echo -e "   ${BLUE}./deploy.sh help${NC}              # Show deployment help"
    echo -e "   ${BLUE}python scripts/package_model.py${NC}  # Package model for distribution"
    echo -e "   ${BLUE}docker-compose up${NC}            # Start with Docker Compose"
    echo ""

    echo -e "${GREEN}Happy coding! 🚀${NC}"
}

# Main execution
main() {
    print_header

    print_info "Starting comprehensive project setup..."
    print_info "This will install all dependencies and configure the development environment."
    echo ""

    # Confirm setup
    read -p "Do you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Setup cancelled."
        exit 0
    fi

    # Run setup steps
    install_system_dependencies
    echo ""

    setup_project_structure
    echo ""

    setup_python_environment
    echo ""

    setup_frontend_environment
    echo ""

    setup_docker_environment
    echo ""

    setup_git_hooks
    echo ""

    setup_environment_config
    echo ""

    # Validate installation
    if validate_installation; then
        echo ""
        show_next_steps
    else
        echo ""
        print_error "Setup completed with some issues. Please check the validation errors above."
        exit 1
    fi
}

# Trap to cleanup on exit
trap 'print_info "Setup interrupted"' INT TERM

# Check if script is run from project root
if [ ! -f "setup.sh" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Run main function
main "$@"
