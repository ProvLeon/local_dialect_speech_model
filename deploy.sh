#!/bin/bash
set -e

# One-Click Deployment Script for Twi Speech Model
# This script provides automated deployment with minimal user interaction

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="twi-speech-model"
DEFAULT_PORT=8000
DEFAULT_ENVIRONMENT="development"
DEFAULT_TARGET="local"

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
    echo "           Twi Speech Model - One-Click Deployment"
    echo "================================================================"
    echo ""
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check system requirements
check_requirements() {
    print_info "Checking system requirements..."

    local missing_deps=()

    # Check Python
    if ! command_exists python3; then
        missing_deps+=("python3")
    else
        python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
        print_info "Python version: $python_version"
    fi

    # Check Docker
    if ! command_exists docker; then
        missing_deps+=("docker")
    else
        docker_version=$(docker --version 2>&1 | cut -d' ' -f3 | cut -d',' -f1)
        print_info "Docker version: $docker_version"

        # Check if Docker daemon is running
        if ! docker info >/dev/null 2>&1; then
            print_error "Docker daemon is not running. Please start Docker and try again."
            exit 1
        fi
    fi

    # Check pip
    if ! command_exists pip3; then
        missing_deps+=("pip3")
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing required dependencies: ${missing_deps[*]}"
        print_info "Please install the missing dependencies and try again."
        exit 1
    fi

    print_success "All system requirements satisfied"
}

# Function to install Python dependencies
install_dependencies() {
    print_info "Installing Python dependencies..."

    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv venv
    fi

    # Activate virtual environment
    source venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip

    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        print_warning "requirements.txt not found, installing basic dependencies..."
        pip install torch numpy librosa soundfile fastapi uvicorn python-multipart
    fi

    print_success "Dependencies installed successfully"
}

# Function to validate model files
validate_model() {
    print_info "Validating model files..."

    local model_base_dir="deployable_twi_speech_model"
    local required_files=("model/model_state_dict.bin" "config/config.json" "tokenizer/label_map.json")

    if [ ! -d "$model_base_dir" ]; then
        print_error "Model directory not found: $model_base_dir"
        return 1
    fi

    for file in "${required_files[@]}"; do
        if [ ! -f "$model_base_dir/$file" ]; then
            print_warning "Model file not found: $model_base_dir/$file"
        else
            print_info "✓ Found: $file"
        fi
    done

    # Run model validation script if available
    if [ -f "scripts/package_model.py" ]; then
        print_info "Running model validation..."
        python scripts/package_model.py --validate-only || {
            print_error "Model validation failed"
            return 1
        }
    fi

    print_success "Model validation completed"
}

# Function to build Docker image
build_docker_image() {
    local image_tag="$1"
    print_info "Building Docker image: $image_tag"

    # Create Dockerfile if it doesn't exist
    if [ ! -f "Dockerfile" ]; then
        print_info "Creating Dockerfile..."
        cat > Dockerfile << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model package
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["python", "-m", "deployable_twi_speech_model.utils.serve"]
EOF
        print_success "Dockerfile created"
    fi

    # Build the image
    docker build -t "$image_tag" . || {
        print_error "Docker build failed"
        return 1
    }

    print_success "Docker image built successfully"
}

# Function to deploy locally
deploy_local() {
    local environment="$1"
    local port="$2"
    local image_tag="$PROJECT_NAME:latest"

    print_info "Deploying locally to $environment environment on port $port"

    # Stop existing container if running
    local container_name="$PROJECT_NAME-$environment"
    if docker ps -q -f name="$container_name" | grep -q .; then
        print_info "Stopping existing container..."
        docker stop "$container_name" >/dev/null 2>&1 || true
        docker rm "$container_name" >/dev/null 2>&1 || true
    fi

    # Run new container
    print_info "Starting new container..."
    docker run -d \
        --name "$container_name" \
        --restart unless-stopped \
        -p "$port:8000" \
        -e ENVIRONMENT="$environment" \
        -e LOG_LEVEL="INFO" \
        "$image_tag" || {
        print_error "Failed to start container"
        return 1
    }

    # Wait for container to be ready
    print_info "Waiting for service to be ready..."
    local max_attempts=30
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if curl -f "http://localhost:$port/health" >/dev/null 2>&1; then
            break
        fi
        sleep 2
        attempt=$((attempt + 1))
        printf "."
    done
    echo ""

    if [ $attempt -eq $max_attempts ]; then
        print_error "Service failed to become ready within timeout"
        print_info "Container logs:"
        docker logs "$container_name" --tail 20
        return 1
    fi

    print_success "Service is ready and healthy"
}

# Function to run quick deployment
quick_deploy() {
    local environment="${1:-$DEFAULT_ENVIRONMENT}"
    local port="${2:-$DEFAULT_PORT}"

    print_info "Starting quick deployment..."
    print_info "Environment: $environment"
    print_info "Port: $port"

    # Check if we can run without Docker (development mode)
    if [ "$environment" = "development" ] && command_exists python3; then
        print_info "Running in development mode (without Docker)..."

        # Install dependencies if needed
        if [ ! -d "venv" ]; then
            install_dependencies
        fi

        # Activate virtual environment
        source venv/bin/activate

        # Validate model
        validate_model || return 1

        # Start the server directly
        print_info "Starting development server on port $port..."
        cd deployable_twi_speech_model/utils
        python serve.py &
        local server_pid=$!
        cd ../..

        # Wait for server to be ready
        print_info "Waiting for server to start..."
        sleep 5

        if curl -f "http://localhost:$port/health" >/dev/null 2>&1; then
            print_success "Development server is ready!"
            print_info "Server PID: $server_pid"
            print_info "To stop the server: kill $server_pid"
        else
            print_error "Server failed to start"
            kill $server_pid 2>/dev/null || true
            return 1
        fi
    else
        # Docker deployment
        build_docker_image "$PROJECT_NAME:latest" || return 1
        deploy_local "$environment" "$port" || return 1
    fi
}

# Function to show deployment status
show_status() {
    local port="${1:-$DEFAULT_PORT}"

    print_info "Checking deployment status..."

    # Check if service is running
    if curl -f "http://localhost:$port/health" >/dev/null 2>&1; then
        print_success "Service is running and healthy"

        # Get model info
        model_info=$(curl -s "http://localhost:$port/model-info" 2>/dev/null || echo "{}")
        if [ "$model_info" != "{}" ]; then
            print_info "Model Information:"
            echo "$model_info" | python3 -m json.tool 2>/dev/null || echo "$model_info"
        fi

        # Show endpoints
        echo ""
        print_info "Available endpoints:"
        echo "  • Health Check: http://localhost:$port/health"
        echo "  • Model Info:   http://localhost:$port/model-info"
        echo "  • Prediction:   http://localhost:$port/test-intent"
        echo "  • API Docs:     http://localhost:$port/docs"

    else
        print_error "Service is not running or not healthy"

        # Check Docker containers
        local containers=$(docker ps -f name="$PROJECT_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}")
        if [ -n "$containers" ]; then
            print_info "Docker containers:"
            echo "$containers"
        fi

        return 1
    fi
}

# Function to stop deployment
stop_deployment() {
    print_info "Stopping deployment..."

    # Stop Docker containers
    local containers=$(docker ps -q -f name="$PROJECT_NAME")
    if [ -n "$containers" ]; then
        docker stop $containers >/dev/null 2>&1
        docker rm $containers >/dev/null 2>&1
        print_success "Docker containers stopped"
    fi

    # Kill Python processes (development mode)
    local python_pids=$(pgrep -f "serve.py" 2>/dev/null || true)
    if [ -n "$python_pids" ]; then
        kill $python_pids 2>/dev/null || true
        print_success "Python server processes stopped"
    fi

    print_success "Deployment stopped"
}

# Function to run tests
run_tests() {
    local port="${1:-$DEFAULT_PORT}"

    print_info "Running deployment tests..."

    # Test health endpoint
    if ! curl -f "http://localhost:$port/health" >/dev/null 2>&1; then
        print_error "Health check failed"
        return 1
    fi
    print_success "✓ Health check passed"

    # Test model info endpoint
    if ! curl -f "http://localhost:$port/model-info" >/dev/null 2>&1; then
        print_error "Model info endpoint failed"
        return 1
    fi
    print_success "✓ Model info endpoint passed"

    # Test prediction endpoint (if test audio is available)
    local test_audio="tests/samples/test_audio.wav"
    if [ -f "$test_audio" ]; then
        if curl -X POST -F "file=@$test_audio" "http://localhost:$port/test-intent" >/dev/null 2>&1; then
            print_success "✓ Prediction endpoint passed"
        else
            print_warning "⚠ Prediction endpoint test failed (might be normal if no test audio)"
        fi
    else
        print_info "ℹ Skipping prediction test (no test audio file)"
    fi

    print_success "All tests passed"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  deploy       Deploy the model (default)"
    echo "  status       Show deployment status"
    echo "  stop         Stop the deployment"
    echo "  test         Run deployment tests"
    echo "  package      Package the model"
    echo "  help         Show this help message"
    echo ""
    echo "Options:"
    echo "  --environment ENV    Deployment environment (default: development)"
    echo "  --port PORT          Service port (default: 8000)"
    echo "  --target TARGET      Deployment target (local, docker, kubernetes)"
    echo "  --quick              Quick deployment with minimal checks"
    echo "  --no-docker          Force non-Docker deployment"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Quick deploy to development"
    echo "  $0 deploy --environment staging      # Deploy to staging"
    echo "  $0 deploy --port 8080                # Deploy on port 8080"
    echo "  $0 status                            # Check deployment status"
    echo "  $0 stop                              # Stop deployment"
    echo ""
}

# Parse command line arguments
COMMAND="deploy"
ENVIRONMENT="$DEFAULT_ENVIRONMENT"
PORT="$DEFAULT_PORT"
TARGET="$DEFAULT_TARGET"
QUICK_MODE=false
NO_DOCKER=false

while [[ $# -gt 0 ]]; do
    case $1 in
        deploy|status|stop|test|package|help)
            COMMAND="$1"
            shift
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --no-docker)
            NO_DOCKER=true
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_banner

    case $COMMAND in
        deploy)
            if [ "$QUICK_MODE" = true ]; then
                quick_deploy "$ENVIRONMENT" "$PORT"
            else
                check_requirements
                validate_model || exit 1

                if [ "$NO_DOCKER" = true ] || [ "$TARGET" = "local" ] && [ "$ENVIRONMENT" = "development" ]; then
                    install_dependencies
                    quick_deploy "$ENVIRONMENT" "$PORT"
                else
                    build_docker_image "$PROJECT_NAME:latest" || exit 1
                    deploy_local "$ENVIRONMENT" "$PORT" || exit 1
                fi
            fi

            echo ""
            show_status "$PORT"

            echo ""
            print_success "🎉 Deployment completed successfully!"
            print_info "Your Twi Speech Model is now running at: http://localhost:$PORT"
            print_info "API Documentation: http://localhost:$PORT/docs"
            echo ""
            ;;

        status)
            show_status "$PORT"
            ;;

        stop)
            stop_deployment
            ;;

        test)
            run_tests "$PORT"
            ;;

        package)
            check_requirements
            if [ -f "scripts/package_model.py" ]; then
                python scripts/package_model.py --verbose
            else
                print_error "Package script not found"
                exit 1
            fi
            ;;

        help)
            show_usage
            ;;

        *)
            print_error "Unknown command: $COMMAND"
            show_usage
            exit 1
            ;;
    esac
}

# Trap to cleanup on exit
trap 'print_info "Script interrupted"' INT TERM

# Run main function
main "$@"
