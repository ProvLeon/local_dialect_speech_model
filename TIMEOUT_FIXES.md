# Timeout Issues and Fixes for Twi Speech Model API

## Problem Summary

The speech model API was experiencing timeout issues during audio processing, specifically in the `/test-intent` endpoint. Users were reporting that requests would hang and never complete, causing frontend applications to timeout while waiting for responses.

## Root Cause Analysis

Through detailed investigation, we identified that the timeout was occurring in the **deployable version** of the API (`deployable_twi_speech_model/utils/serve.py`), not the main development API. The specific bottlenecks were:

1. **Audio Loading**: `librosa.load()` operations without timeout limits
2. **Feature Extraction**: MFCC computation hanging on certain audio formats
3. **No Timeout Handling**: No timeout mechanisms in the audio processing pipeline
4. **Subprocess Calls**: FFmpeg conversion calls without timeout limits
5. **Slow Processing**: Complex audio processing taking 2+ seconds per request

## Files Modified

### 1. `deployable_twi_speech_model/utils/inference.py`
- Added timeout handling for all audio processing steps
- Implemented context manager `_timeout_handler()` using signal alarms
- Added comprehensive logging throughout the pipeline
- Modified methods:
  - `load_audio()`: Added 30s timeout
  - `extract_features()`: Added 15s timeout
  - `process_audio_file()`: Added 45s timeout
  - `predict()`: Added 60s timeout
  - `predict_topk()`: Added 60s timeout

### 2. `deployable_twi_speech_model/utils/serve.py`
- Added async timeout handling for API endpoints
- Implemented proper cleanup of temporary files
- Added timeout-specific error responses (HTTP 408)
- Modified endpoints:
  - `/predict`: 90s total timeout
  - `/test-intent`: 90s total timeout

### 3. `src/utils/audio_converter.py`
- Added timeout handling for audio conversion operations
- Implemented signal-based timeout mechanism
- Added timeouts for:
  - FFmpeg conversion: 30s
  - Pydub conversion: configurable timeout
  - Librosa conversion: configurable timeout
  - Audio validation: 10s

### 4. `src/preprocessing/audio_processor.py`
- Added timeout handling for the main audio preprocessing pipeline
- Enhanced logging for debugging
- Added timeouts for each processing step

### 5. `src/api/speech_api.py`
- Added async timeout handling for the main API
- Implemented proper error handling for timeout scenarios
- Added comprehensive logging

### 6. `config/api_config.py`
- Added timeout configuration section
- Defined timeout values for different operations

## Timeout Configuration

```python
"timeouts": {
    "audio_conversion": 30,      # seconds
    "audio_validation": 15,      # seconds
    "audio_preprocessing": 60,   # seconds
    "model_inference": 30,       # seconds
    "total_request": 120,        # seconds
    "ffmpeg_check": 10,          # seconds
    "subprocess": 30,            # seconds
}
```

## New Error Handling

### HTTP Status Codes
- **408 Request Timeout**: Processing timed out, user should try shorter audio
- **400 Bad Request**: Invalid audio format or corrupted file
- **500 Internal Server Error**: Unexpected processing errors

### Error Messages
- "Processing timed out. Please try a shorter audio file."
- "Audio conversion timed out. Please try a different format."
- "Audio preprocessing timed out. Please try a simpler audio file."

## Performance Improvements

1. **Faster Processing**: Added logging to identify bottlenecks
2. **Better Resource Management**: Proper cleanup of temporary files
3. **Graceful Degradation**: Timeout handling prevents hanging requests
4. **Client Feedback**: Clear error messages for timeout scenarios

## Testing

Created comprehensive test scripts:

### 1. `debug_timeout.py`
- Tests individual components of the audio processing pipeline
- Identifies which step is causing delays
- Creates synthetic test audio for validation

### 2. `check_status.py`
- Validates system health and dependencies
- Checks for running processes and available resources
- Verifies model files and configurations

### 3. `test_timeout_fixes.py`
- End-to-end testing of timeout fixes
- Tests both normal and long audio files
- Validates API response times and error handling

## Deployment Verification

To verify the fixes are working:

1. **Check API Health**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Test with Normal Audio**:
   ```bash
   curl -X POST "http://localhost:8000/test-intent?top_k=5" \
     -F "file=@test_audio.wav" \
     --max-time 60
   ```

3. **Run Diagnostic Script**:
   ```bash
   python check_status.py
   ```

4. **Run Timeout Tests**:
   ```bash
   python test_timeout_fixes.py
   ```

## Expected Performance

After fixes:
- **Normal audio files (2-3s)**: Process in <5 seconds
- **Longer audio files (5-10s)**: Process in <15 seconds or timeout gracefully
- **Invalid files**: Reject with clear error message in <5 seconds
- **API response time**: Include processing time in response

## Monitoring

Key metrics to monitor:
- `processing_time_ms` in API responses
- HTTP 408 error rates
- Server logs for timeout warnings
- Memory usage during audio processing

## Troubleshooting

### If timeouts still occur:

1. **Check audio file format**: Ensure WAV/MP3 format
2. **Verify file size**: Large files (>10MB) may still timeout
3. **Check server resources**: CPU/memory availability
4. **Review logs**: Look for specific timeout errors
5. **Test locally**: Use debug scripts to isolate issues

### Common issues:
- **FFmpeg not available**: Install FFmpeg on the system
- **Memory limits**: Large audio files consuming too much RAM
- **Slow disk I/O**: Temporary file operations taking too long

## Future Improvements

1. **Streaming Processing**: Process audio in chunks to reduce memory usage
2. **Async Audio Loading**: Non-blocking audio file operations
3. **Caching**: Cache processed features for repeated requests
4. **Load Balancing**: Distribute processing across multiple workers
5. **Progressive Timeout**: Different timeout values based on file size

## Configuration for Production

For production deployment, consider adjusting timeouts based on:
- Server hardware capabilities
- Expected audio file sizes
- User experience requirements
- Network latency considerations

Recommended production timeouts:
```python
"timeouts": {
    "audio_conversion": 20,      # Faster for production
    "audio_preprocessing": 45,   # Reduced for better UX
    "total_request": 90,         # Maximum user wait time
}
```
