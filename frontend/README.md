# Twi Speech Recognition Frontend

A modern Next.js frontend for the Twi Speech Recognition system with support for both local API and Gradio HuggingFace Space integration.

## Features

- 🎤 **Real-time Audio Recording** with live processing
- 📁 **File Upload Support** for various audio formats (WAV, MP3, WebM)
- 🌐 **Dual API Support** - Local API server + Gradio HuggingFace Space
- 🔄 **Automatic Fallback** - Tries Gradio first, falls back to local API
- 📊 **Live Intent Streaming** with confidence visualization
- ⚙️ **Configuration Panel** to monitor API status
- 🎯 **Intent Classification** with top-K predictions
- 📱 **Responsive Design** with modern UI components

## Quick Start

### 1. Installation

```bash
npm install
# or
yarn install
```

### 2. Configuration

Copy the environment template:
```bash
cp .env.example .env.local
```

Edit `.env.local` with your configuration:

#### For Gradio HuggingFace Space (Recommended):
```bash
NEXT_PUBLIC_USE_GRADIO=true
NEXT_PUBLIC_GRADIO_SPACE_ID=TwiWhisperModel/TwiSpeechModel
NEXT_PUBLIC_HF_TOKEN=hf_your_token_here  # Optional for public spaces
NEXT_PUBLIC_DEBUG_MODE=false
```

#### For Local API Server:
```bash
NEXT_PUBLIC_USE_GRADIO=false
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
NEXT_PUBLIC_DEBUG_MODE=true
```

### 3. Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the application.

## Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `NEXT_PUBLIC_USE_GRADIO` | Yes | Enable Gradio API | `true` or `false` |
| `NEXT_PUBLIC_GRADIO_SPACE_ID` | If using Gradio | HuggingFace Space ID | `TwiWhisperModel/TwiSpeechModel` |
| `NEXT_PUBLIC_HF_TOKEN` | For private spaces | HuggingFace API token | `hf_xxxxxxxxxxxx` |
| `NEXT_PUBLIC_API_BASE_URL` | If using local API | Local server URL | `http://localhost:8000` |
| `NEXT_PUBLIC_DEBUG_MODE` | No | Enable debug logging | `true` or `false` |

## API Integration

### Gradio Integration

The frontend integrates with Gradio HuggingFace Spaces using the `@gradio/client` package:

- **Automatic Connection**: Connects to your deployed Gradio space
- **Retry Logic**: Handles connection failures with exponential backoff
- **Fallback Support**: Falls back to local API if Gradio fails
- **Real-time Status**: Shows connection status in settings panel

### Local API Integration

Supports the local FastAPI server with:
- **File Upload**: Direct audio file processing
- **Streaming**: Chunked audio processing for real-time feedback
- **Health Monitoring**: Automatic health checks and status display

## Usage

### 1. Audio Recording
- Click the microphone button to start recording
- Speak in Twi language
- Click stop to process the audio
- View transcription and intent results

### 2. File Upload
- Click "Choose File" to select an audio file
- Supported formats: WAV, MP3, WebM, M4A
- Maximum file size: 50MB
- Get instant results with confidence scores

### 3. Live Streaming (Local API Only)
- Enable "Live Recording" mode
- Speak continuously for real-time processing
- See streaming results as you speak

### 4. Configuration Panel
- Click the settings icon (⚙️) in the top-right
- View current API configuration
- Check connection status for both APIs
- See environment variable examples

## Development

### Project Structure

```
frontend/
├── app/
│   ├── page.tsx          # Main application component
│   ├── layout.tsx        # App layout
│   └── globals.css       # Global styles
├── components/
│   └── api-settings.tsx  # Settings panel component
├── lib/
│   ├── api.ts           # Local API client
│   └── gradio-client.ts # Gradio integration
└── public/              # Static assets
```

### Key Components

- **`app/page.tsx`**: Main chat-style interface with audio recording
- **`components/api-settings.tsx`**: Configuration and status panel
- **`lib/api.ts`**: Local API integration with fallback logic
- **`lib/gradio-client.ts`**: HuggingFace Gradio Space client

### API Flow

1. **Primary**: Try Gradio API (if enabled)
2. **Fallback**: Use Local API if Gradio fails
3. **Error Handling**: User-friendly error messages
4. **Status Monitoring**: Real-time connection status

## Deployment

### Vercel (Recommended)

```bash
# Deploy to Vercel
npm run build
vercel deploy

# Set environment variables in Vercel dashboard
```

### Environment Variables for Production:
```bash
NEXT_PUBLIC_USE_GRADIO=true
NEXT_PUBLIC_GRADIO_SPACE_ID=your-username/your-space
NEXT_PUBLIC_HF_TOKEN=hf_your_production_token
NEXT_PUBLIC_DEBUG_MODE=false
```

### Docker

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## Troubleshooting

### Common Issues

1. **Gradio Connection Failed**
   - Check if `NEXT_PUBLIC_GRADIO_SPACE_ID` is correct
   - Verify the HuggingFace Space is running
   - Check if you need a HuggingFace token

2. **Local API Not Found**
   - Ensure local server is running on port 8000
   - Check `NEXT_PUBLIC_API_BASE_URL` setting
   - Verify CORS is properly configured

3. **Audio Not Recording**
   - Grant microphone permissions in browser
   - Check if using HTTPS (required for getUserMedia)
   - Try different audio formats

### Debug Mode

Enable debug mode for detailed logging:
```bash
NEXT_PUBLIC_DEBUG_MODE=true
```

This will show:
- API request/response details
- Connection status logs
- Audio processing information
- Error stack traces

## Learn More

- [Next.js Documentation](https://nextjs.org/docs)
- [Gradio Client Documentation](https://www.gradio.app/guides/gradio-client-in-javascript)
- [HuggingFace Spaces](https://huggingface.co/docs/hub/spaces)
- [Twi Speech Recognition Model](https://huggingface.co/TwiWhisperModel/TwiWhisperModel)
