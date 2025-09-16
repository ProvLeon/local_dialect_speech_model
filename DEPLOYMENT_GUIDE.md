# Twi Speech Model - Deployment Guide

🚀 **Professional-grade packaging and deployment system for the Twi Speech Intent Recognition Model**

This guide provides comprehensive instructions for packaging, deploying, and managing the Twi Speech Model in production environments, similar to how ChatGPT and other commercial AI models are deployed.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [System Overview](#system-overview)
- [Installation & Setup](#installation--setup)
- [Packaging the Model](#packaging-the-model)
- [Deployment Options](#deployment-options)
- [CI/CD Integration](#cicd-integration)
- [Monitoring & Management](#monitoring--management)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

## 🚀 Quick Start

### One-Click Deployment

```bash
# Clone the repository
git clone <repository-url>
cd local_dialect_speech_model

# Setup everything (dependencies, environment, etc.)
./setup.sh

# Deploy with one command
./deploy.sh --quick
```

Your API will be available at `http://localhost:8000`

### Quick Test

```bash
# Check status
./deploy.sh status

# Run tests
python scripts/test_model.py --quick

# Stop deployment
./deploy.sh stop
```

## 🏗️ System Overview

### Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   API Gateway   │    │   Model Service │
│   (Next.js)     │───▶│   (FastAPI)     │───▶│   (PyTorch)     │
│   Port 3000     │    │   Port 8000     │    │   In-Memory     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │      Monitoring        │
                    │   (Health, Metrics)    │
                    └─────────────────────────┘
```

### Components

- **Model Package**: Self-contained PyTorch model with preprocessing
- **API Server**: FastAPI-based REST API with automatic documentation
- **Frontend**: Next.js web interface for testing and interaction
- **Packaging System**: Automated model packaging and versioning
- **Deployment Scripts**: One-click deployment with multiple targets
- **CI/CD Pipeline**: Automated testing, building, and deployment
- **Monitoring**: Health checks, metrics, and performance tracking

## 🛠️ Installation & Setup

### Prerequisites

- **Python 3.8+** (3.9 recommended)
- **Docker** (optional, for containerized deployment)
- **Node.js 18+** (for frontend)
- **Git** (for version control)

### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv \
  libsndfile1 ffmpeg portaudio19-dev build-essential \
  docker.io docker-compose git curl
```

**macOS:**
```bash
brew install python@3.9 libsndfile ffmpeg portaudio \
  docker docker-compose git curl node@18
```

**Windows:**
- Install Python 3.9+ from python.org
- Install Docker Desktop
- Install Node.js 18+
- Install Git
- Consider using WSL for better compatibility

### Automated Setup

```bash
# Run the comprehensive setup script
./setup.sh

# This will:
# - Install system dependencies
# - Create Python virtual environment
# - Install Python packages
# - Setup Node.js environment (if frontend exists)
# - Configure Git hooks
# - Setup Docker environment
# - Create project structure
# - Configure environment files
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install black isort flake8 pytest pre-commit

# Setup pre-commit hooks
pre-commit install

# Copy environment configuration
cp .env.example .env
# Edit .env with your settings
```

## 📦 Packaging the Model

### Automated Packaging

```bash
# Package the model with full validation
python scripts/package_model.py

# Quick validation only
python scripts/package_model.py --validate-only

# Custom output directory
python scripts/package_model.py --output-dir ./releases

# Verbose output
python scripts/package_model.py --verbose
```

### Package Contents

The packaging system creates a complete distribution with:

```
TwiSpeechIntentClassifier_1.0.0/
├── deployable_twi_speech_model/
│   ├── model/                 # Model files
│   ├── config/               # Configuration
│   ├── utils/                # Inference utilities
│   ├── preprocessor/         # Audio preprocessing
│   └── tokenizer/           # Label mappings
├── src/                     # Source code
├── docs/                    # Documentation
├── scripts/                 # Deployment scripts
├── tests/                   # Test suite
├── docker/                  # Docker configurations
├── Dockerfile              # Container definition
├── docker-compose.yml      # Multi-service setup
├── requirements.txt        # Dependencies
├── setup.py               # Python package setup
└── README.md              # Quick start guide
```

### Package Validation

The packaging system performs comprehensive validation:

- ✅ Model file integrity
- ✅ Configuration validity
- ✅ Dependency compatibility
- ✅ Performance benchmarks
- ✅ Security scanning
- ✅ Documentation completeness

### Distribution

```bash
# The packager creates:
TwiSpeechIntentClassifier_1.0.0.tar.gz    # Main package
TwiSpeechIntentClassifier_1.0.0.sha256    # Integrity hash
TwiSpeechIntentClassifier_1.0.0_MANIFEST.json  # Package metadata
```

## 🚀 Deployment Options

### 1. Local Development

**Quick Development Server:**
```bash
./deploy.sh --quick
# or
cd deployable_twi_speech_model/utils && python serve.py
```

**With Docker:**
```bash
./deploy.sh deploy --environment development --target docker
```

### 2. Docker Deployment

**Build and Run:**
```bash
# Build image
docker build -t twi-speech-model .

# Run container
docker run -d -p 8000:8000 --name twi-speech twi-speech-model

# Or use deployment script
./deploy.sh deploy --environment staging --target docker
```

**Docker Compose:**
```bash
docker-compose up -d
```

### 3. Kubernetes Deployment

**Automated Deployment:**
```bash
python scripts/deploy.py deploy \
  --environment production \
  --target kubernetes \
  --image-tag twi-speech-model:latest
```

**Manual Deployment:**
```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get deployments,services,ingress -n twi-speech
```

### 4. Cloud Deployment

**AWS ECS:**
```bash
python scripts/deploy.py deploy \
  --environment production \
  --target aws-ecs \
  --image-tag twi-speech-model:latest
```

**Google Cloud Run:**
```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT-ID/twi-speech-model

# Deploy to Cloud Run
gcloud run deploy twi-speech-model \
  --image gcr.io/PROJECT-ID/twi-speech-model \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## 🔄 CI/CD Integration

### GitHub Actions

The project includes a comprehensive CI/CD pipeline:

**Pipeline Stages:**
1. **Testing** - Unit tests, integration tests, linting
2. **Security** - Vulnerability scanning, code security analysis
3. **Validation** - Model validation, performance testing
4. **Building** - Docker image building, package creation
5. **Deployment** - Automated deployment to environments

**Workflow Triggers:**
- Push to `main` → Deploy to staging
- Push to `develop` → Deploy to development
- Tag `v*` → Deploy to production
- Manual trigger → Custom deployment

**Configuration:**
```yaml
# .github/workflows/deploy.yml
name: Deploy Twi Speech Model
on:
  push:
    branches: [main, develop]
  workflow_dispatch:
    inputs:
      environment:
        description: 'Deployment environment'
        required: true
        type: choice
        options: [development, staging, production]
```

### Environment Configuration

**Development Environment:**
- Automatic deployment on `develop` branch
- Basic testing and validation
- Local or development cluster deployment

**Staging Environment:**
- Deployment on `main` branch
- Full test suite including integration tests
- Performance benchmarking
- Kubernetes deployment

**Production Environment:**
- Manual approval required
- Comprehensive testing and validation
- Blue-green deployment strategy
- Monitoring and alerting setup
- Rollback capability

## 📊 Monitoring & Management

### Health Monitoring

**Health Check Endpoint:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "message": "Speech Model API is running",
  "uptime_seconds": 3600,
  "model_info": {
    "name": "TwiSpeechIntentClassifier",
    "version": "1.0.0",
    "device": "cpu"
  }
}
```

### Performance Monitoring

**Metrics Endpoint:**
```bash
curl http://localhost:8000/metrics
```

**Key Metrics:**
- Request latency (p50, p95, p99)
- Throughput (requests per second)
- Error rates
- Memory usage
- Model inference time

### Logging

**Structured Logging:**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "message": "Prediction completed",
  "request_id": "req-123",
  "intent": "add_to_cart",
  "confidence": 0.95,
  "processing_time_ms": 245
}
```

**Log Aggregation:**
- Centralized logging with structured format
- Request tracing and correlation
- Performance metrics extraction
- Error tracking and alerting

### Status Dashboard

```bash
# Check deployment status
./deploy.sh status

# Generate deployment report
python scripts/deploy.py status --environment production
```

## 🔧 Troubleshooting

### Common Issues

**1. Model Loading Errors**
```bash
# Check model files
ls -la deployable_twi_speech_model/model/

# Validate model package
python scripts/package_model.py --validate-only

# Check dependencies
pip list | grep -E "(torch|librosa|soundfile)"
```

**2. API Connection Issues**
```bash
# Check if service is running
curl http://localhost:8000/health

# Check Docker container
docker ps | grep twi-speech
docker logs twi-speech-container

# Check port availability
lsof -i :8000
```

**3. Performance Issues**
```bash
# Run performance tests
python scripts/test_model.py --performance-only

# Check resource usage
docker stats twi-speech-container

# Monitor system resources
htop
```

**4. Frontend Connection Issues**
```bash
# Check CORS configuration
curl -H "Origin: http://localhost:3000" \
     -H "Access-Control-Request-Method: POST" \
     -X OPTIONS http://localhost:8000/test-intent

# Test API endpoints directly
curl -X POST -F "file=@test_audio.wav" \
     http://localhost:8000/test-intent?top_k=5
```

### Debugging Commands

```bash
# Verbose deployment
./deploy.sh deploy --environment development --verbose

# Debug mode API server
DEBUG=true python deployable_twi_speech_model/utils/serve.py

# Run comprehensive tests
python scripts/test_model.py --verbose

# Check Docker build logs
docker build --no-cache -t twi-speech-model . 2>&1 | tee build.log
```

### Log Analysis

```bash
# View recent logs
docker logs --tail 100 twi-speech-container

# Follow logs in real-time
docker logs -f twi-speech-container

# Search for errors
docker logs twi-speech-container 2>&1 | grep -i error

# Export logs for analysis
docker logs twi-speech-container > app.log 2>&1
```

## ⚙️ Advanced Configuration

### Environment Variables

```bash
# Core Configuration
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
MAX_REQUESTS=1000

# Model Configuration
MODEL_PATH=deployable_twi_speech_model/
ENABLE_CUDA=true
BATCH_SIZE=1
MAX_AUDIO_LENGTH=30

# Security Configuration
SECRET_KEY=your-secure-secret-key
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# Monitoring Configuration
ENABLE_METRICS=true
METRICS_PORT=9090
LOG_FORMAT=json
```

### Custom Deployment Configuration

**deployment_config.yaml:**
```yaml
app:
  name: twi-speech-model
  version: "1.0.0"
  port: 8000
  workers: 4

environments:
  production:
    domain: api.yourdomain.com
    ssl: true
    replicas: 3
    auto_scaling:
      enabled: true
      min_replicas: 2
      max_replicas: 10
      target_cpu_utilization: 70

kubernetes:
  namespace: production
  deployment_name: twi-speech-deployment
  service_name: twi-speech-service
  ingress_name: twi-speech-ingress

monitoring:
  health_endpoint: /health
  metrics_endpoint: /metrics
  log_retention_days: 30
  alert_email: alerts@yourdomain.com
```

### Performance Tuning

**High-Throughput Configuration:**
```bash
# Increase workers
API_WORKERS=8

# Enable batch processing
BATCH_SIZE=4

# Optimize memory usage
TORCH_NUM_THREADS=4
OMP_NUM_THREADS=4

# Use GPU if available
ENABLE_CUDA=true
```

**Low-Latency Configuration:**
```bash
# Single worker for consistency
API_WORKERS=1

# Disable batch processing
BATCH_SIZE=1

# Optimize for speed
TORCH_THREADS=1
MODEL_OPTIMIZATION=speed
```

### Security Hardening

```bash
# Enable security features
ENABLE_CORS=true
ALLOWED_ORIGINS=https://yourdomain.com
MAX_REQUEST_SIZE_MB=10

# API Authentication
ENABLE_API_AUTH=true
API_KEY_REQUIRED=true

# Rate limiting
ENABLE_RATE_LIMITING=true
REQUESTS_PER_MINUTE=100

# Security headers
ENABLE_SECURITY_HEADERS=true
```

### Scaling Configuration

**Horizontal Scaling (Kubernetes):**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: twi-speech-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: twi-speech-deployment
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

**Load Balancing:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: twi-speech-service
spec:
  selector:
    app: twi-speech-model
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## 📚 API Documentation

Once deployed, comprehensive API documentation is available at:
- **Interactive Docs**: `http://localhost:8000/docs`
- **OpenAPI Spec**: `http://localhost:8000/openapi.json`
- **ReDoc**: `http://localhost:8000/redoc`

### Key Endpoints

**Health Check:**
```http
GET /health
```

**Model Information:**
```http
GET /model-info
```

**Intent Prediction:**
```http
POST /test-intent?top_k=5
Content-Type: multipart/form-data

file: audio_file.wav
```

## 🤝 Contributing

### Development Workflow

1. **Setup Development Environment:**
   ```bash
   ./setup.sh
   source venv/bin/activate
   ```

2. **Create Feature Branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes and Test:**
   ```bash
   # Format code
   black . && isort .

   # Run tests
   python scripts/test_model.py --quick

   # Test deployment
   ./deploy.sh --quick
   ```

4. **Commit and Push:**
   ```bash
   git add .
   git commit -m "Add your feature"
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**

### Code Quality

- **Formatting**: Black, isort
- **Linting**: flake8, mypy
- **Testing**: pytest with coverage
- **Security**: bandit, safety
- **Pre-commit hooks**: Automated quality checks

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🚀 Ready to Deploy?

```bash
# Quick start
./setup.sh && ./deploy.sh --quick

# Production deployment
./deploy.sh deploy --environment production --target kubernetes

# Monitor deployment
./deploy.sh status
```

For support and questions, please open an issue in the repository or contact the development team.

**Happy deploying! 🎉**
