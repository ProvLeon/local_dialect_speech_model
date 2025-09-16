# Render Deployment Guide for Twi Speech Model

This guide explains how to deploy your Twi Speech Model to Render.com, a cloud platform that makes deployment simple and reliable.

## 🚀 Quick Start

**TL;DR:** You **cannot** use `deploy.sh` on Render. Instead, follow these steps:

1. Run the deployment helper: `./render-deploy.sh`
2. Commit and push your code to GitHub
3. Connect your GitHub repo to Render
4. Deploy using the included `render.yaml` configuration

## 📋 Prerequisites

### Required Tools
- **Git** - for version control
- **GitHub account** - to host your code
- **Render account** - sign up at [render.com](https://render.com)
- **Docker** (optional) - for local testing

### Required Files (✅ Already Included)
- `Dockerfile` - Container configuration
- `render.yaml` - Render service configuration
- `deployable_twi_speech_model/` - Your packaged model
- `render-deploy.sh` - Deployment helper script

## 🔧 Deployment Process

### Step 1: Prepare Your Project

Run the deployment helper script:

```bash
./render-deploy.sh
```

This script will:
- ✅ Validate your project structure
- ✅ Check for required files
- ✅ Create necessary configuration files
- ✅ Test Docker build (optional)
- ✅ Show you next steps

### Step 2: Commit and Push to GitHub

```bash
# Add all changes
git add .

# Commit with a descriptive message
git commit -m "Prepare Twi Speech Model for Render deployment"

# Push to your main branch
git push origin main
```

### Step 3: Connect to Render

1. **Go to Render Dashboard**: Visit [dashboard.render.com](https://dashboard.render.com)
2. **Sign in** with your GitHub account
3. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select this repository

### Step 4: Configure Deployment

Render will automatically detect your `render.yaml` file with these settings:

```yaml
services:
  - type: web
    name: akan-twi-speech-api
    env: docker
    plan: starter
    dockerfilePath: ./Dockerfile
    healthCheckPath: /health
```

**Review the configuration and click "Create Web Service"**

### Step 5: Monitor Deployment

- Watch build logs in the Render dashboard
- Deployment typically takes 5-10 minutes
- Your API will be available at: `https://YOUR-SERVICE-NAME.onrender.com`

## 🔗 API Endpoints

After successful deployment, your API will have these endpoints:

| Endpoint | Description | URL |
|----------|-------------|-----|
| Health Check | Service status | `GET /health` |
| API Documentation | Interactive API docs | `GET /docs` |
| Model Info | Model metadata | `GET /model-info` |
| Predict Intent | Audio classification | `POST /predict` |
| Test Intent | Frontend-compatible endpoint | `POST /test-intent` |

### Example Usage

```bash
# Check if service is running
curl https://your-app.onrender.com/health

# Get model information
curl https://your-app.onrender.com/model-info

# Test with audio file
curl -X POST -F "file=@test_audio.wav" https://your-app.onrender.com/test-intent
```

## 🐳 Docker Configuration

Your `Dockerfile` is configured for Render with:

- **Base Image**: Python 3.11 slim
- **Dependencies**: Audio processing libraries (libsndfile1, ffmpeg)
- **Model Location**: `deployable_twi_speech_model/`
- **Port**: 8000 (automatically mapped by Render)
- **Security**: Non-root user execution
- **Health Check**: Built-in health monitoring

## 💰 Pricing and Plans

### Free Tier (Starter Plan)
- ✅ **Perfect for testing and demos**
- 📦 512 MB RAM, 0.1 CPU
- 😴 **Sleeps after 15 minutes of inactivity**
- ⏰ **30+ second cold start time**
- 🆓 **Completely free**

### Paid Plans (Standard+)
- 🚀 **Always-on service (no sleeping)**
- 💪 More RAM and CPU
- ⚡ Instant response times
- 💳 Starting at $7/month

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 🚨 Build Failures
**Problem**: Docker build fails on Render

**Solutions**:
```bash
# Test Docker build locally first
./render-deploy.sh test

# Check for missing files
./render-deploy.sh validate

# Ensure all model files are committed
git add deployable_twi_speech_model/
git commit -m "Add model files"
```

#### 🚨 Service Won't Start
**Problem**: Service builds but fails to start

**Check**:
- Render logs for Python errors
- Model files are present in `deployable_twi_speech_model/`
- Port 8000 is properly exposed
- No blocking operations in startup code

#### 🚨 Health Check Fails
**Problem**: Health check endpoint returns errors

**Solutions**:
- Verify `/health` endpoint exists in `serve.py`
- Check service starts within 10 minutes
- Ensure no infinite loops in model loading

#### 🚨 Out of Memory
**Problem**: Service crashes due to memory limits

**Solutions**:
- Upgrade to paid plan with more RAM
- Optimize model loading (lazy loading)
- Use smaller model files if possible
- Monitor memory usage in Render dashboard

#### 🚨 Slow Cold Starts (Free Tier)
**Problem**: Service takes 30+ seconds to respond after sleeping

**This is normal for free tier**:
- Service sleeps after 15 minutes of inactivity
- Consider upgrading to paid plan for always-on service
- Use a uptime monitoring service to prevent sleeping

## 🔧 Advanced Configuration

### Environment Variables

You can add environment variables in Render dashboard or `render.yaml`:

```yaml
envVars:
  - key: LOG_LEVEL
    value: "INFO"
  - key: MODEL_CACHE_SIZE
    value: "100"
  - key: MAX_FILE_SIZE
    value: "10485760"  # 10MB
```

### Custom Domain

1. Go to your service in Render dashboard
2. Click "Settings" → "Custom Domains"
3. Add your domain and configure DNS

### Scaling

Render automatically handles:
- **Load balancing** for multiple instances
- **Auto-scaling** based on traffic
- **Health monitoring** and automatic restarts

## 📊 Monitoring and Logs

### Viewing Logs
```bash
# Using Render CLI (if installed)
render logs --service your-service-name

# Or view in dashboard
# Go to your service → "Logs" tab
```

### Metrics Available
- **Response times**
- **Error rates**
- **Memory usage**
- **CPU usage**
- **Request volume**

## 🛡️ Security Best Practices

Your deployment includes:

- ✅ **Non-root user** execution
- ✅ **CORS** configured for web access
- ✅ **Health checks** for monitoring
- ✅ **HTTPS** by default on Render
- ✅ **Environment variable** protection

### Additional Security (Optional)

```python
# Add API key authentication
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != os.environ.get("API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key
```

## 🎯 Performance Optimization

### Model Loading
```python
# Lazy loading for faster startup
class ModelInference:
    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model
```

### Caching
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def preprocess_audio(audio_path: str):
    # Cache preprocessing results
    pass
```

## 📚 Alternative Deployment Methods

### Using Render CLI

```bash
# Install Render CLI
npm install -g @render/cli

# Login
render auth login

# Deploy
render deploy
```

### Manual Deployment

1. **Create service manually** in Render dashboard
2. **Configure environment** variables
3. **Set build command**: `docker build -t app .`
4. **Set start command**: `python deployable_twi_speech_model/utils/serve.py`

## 🔄 Updates and Redeployment

### Automatic Deployment
- **Push to main branch** → Automatic redeployment
- **Enable auto-deploy** in render.yaml (already configured)

### Manual Deployment
- Click "Manual Deploy" in Render dashboard
- Select branch/commit to deploy

### Rolling Back
- Go to "Deploys" tab in dashboard
- Click "Redeploy" on previous successful deployment

## 📞 Support and Resources

### Render Documentation
- [Render Docs](https://render.com/docs)
- [Docker Deployment Guide](https://render.com/docs/docker)
- [FastAPI Deployment](https://render.com/docs/deploy-fastapi)

### Project Support
- Check the project's GitHub issues
- Review `DEPLOYMENT_GUIDE.md` for general deployment info
- Use `./render-deploy.sh troubleshoot` for common problems

## ✅ Success Checklist

After deployment, verify:

- [ ] Health check returns 200: `curl https://your-app.onrender.com/health`
- [ ] Model info loads: `curl https://your-app.onrender.com/model-info`
- [ ] API docs accessible: `https://your-app.onrender.com/docs`
- [ ] Can upload and process audio files
- [ ] Logs show no critical errors
- [ ] Response times are acceptable

## 🎉 You're All Set!

Your Twi Speech Model is now running on Render! The service will:

- 🌐 **Auto-scale** based on traffic
- 🔄 **Auto-restart** if it crashes
- 📊 **Monitor** performance metrics
- 🔒 **Secure** with HTTPS by default
- 💰 **Start free** with option to upgrade

**Your API is live at**: `https://your-service-name.onrender.com`

---

*Need help? Run `./render-deploy.sh troubleshoot` for common solutions or check the Render documentation.*
