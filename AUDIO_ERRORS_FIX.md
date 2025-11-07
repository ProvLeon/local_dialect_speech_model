# Audio Processing Errors - Fixed

## Issues Identified and Resolved

### 1. Backend Threading Error (HTTP 500)
**Error**: `signal only works in main thread of the main interpreter`

**Root Cause**: The backend `inference.py` was using `signal.signal()` and `signal.alarm()` for timeout handling, which only work in the main thread. When FastAPI runs with async/await and multiple workers, audio processing happens in background threads where signals are not allowed.

**Fix Applied**:
- Replaced signal-based timeout handling with `concurrent.futures.ThreadPoolExecutor`
- Updated `AudioProcessor._timeout_handler()` method
- Modified `load_audio()` and `extract_features()` methods to use thread-safe timeout handling

### 2. Frontend Audio Conversion Failures
**Error**: `DOMException: The buffer passed to decodeAudioData contains an unknown content type`

**Root Cause**:
- MediaRecorder produces WebM/Opus chunks (since WAV isn't directly supported)
- The frontend was trying to convert these incomplete streaming chunks to WAV
- `decodeAudioData()` can't handle WebM/Opus chunks, especially incomplete ones from live streams

**Fix Applied**:
- Disabled automatic conversion for WebM/Opus chunks
- Backend can handle WebM format directly
- Added better error recovery and logging
- Improved MediaRecorder format selection with fallback chain

### 3. Improved Error Handling
**Changes Made**:
- Added comprehensive logging for debugging both frontend and backend
- Better timeout handling without signals
- Improved error messages and recovery
- Added debug mode with detailed request/response logging

## Files Modified

### Backend Changes
- `deployable_twi_speech_model/utils/inference.py`: Fixed threading issues
- `start_render.py`: Updated to use single worker for better compatibility

### Frontend Changes
- `frontend/app/page.tsx`: Improved audio handling and error recovery
- `frontend/lib/api.ts`: Enhanced logging and error reporting
- `frontend/.env.local`: Added debug configuration

## Technical Details

### Backend Threading Fix
```python
# OLD (signal-based - doesn't work in threads)
with self._timeout_handler(timeout_seconds):
    audio, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration)

# NEW (thread-safe)
def _load_audio_task():
    return librosa.load(audio_path, sr=self.sr, duration=self.duration)

with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(_load_audio_task)
    audio, sr = future.result(timeout=timeout_seconds)
```

### Frontend Audio Processing
```javascript
// OLD (forced conversion causing errors)
if (!ev.data.type.includes('wav')) {
  audioBlob = await convertToWAV(ev.data);
}

// NEW (send original format, backend handles it)
const audioBlob = ev.data; // Backend can process WebM directly
```

## Testing Status
- ✅ Backend no longer throws signal threading errors
- ✅ Frontend sends audio chunks without conversion failures
- ✅ Comprehensive logging added for debugging
- ✅ Error recovery mechanisms in place

## Deployment Notes
- Backend requires restart to apply threading fixes
- Frontend changes are backwards compatible
- Debug logging can be disabled by setting `NEXT_PUBLIC_DEBUG_MODE=false`
- Single worker configuration recommended for optimal stability

## Error Prevention
- Thread-safe timeout handling prevents signal errors
- Direct audio format sending avoids conversion failures
- Better error boundaries prevent cascading failures
- Comprehensive logging helps with future debugging

## Update: WebM Processing Timeout Fix

### Additional Issue Identified
**Error**: Request timeouts when processing WebM/Opus audio chunks
**Root Cause**: Librosa has difficulty processing WebM format efficiently, causing long processing times that exceed client timeouts.

### Additional Fixes Applied
1. **Backend WebM Handling**:
   - Added FFmpeg conversion from WebM to WAV when available
   - Implemented fallback audio loading methods (librosa → soundfile → error handling)
   - Increased timeouts for WebM processing (2 minutes vs 30 seconds)
   - Better WebM content detection using magic bytes

2. **Frontend Optimizations**:
   - Increased chunk recording interval from 2s to 5s
   - Raised minimum chunk size from 5KB to 15KB
   - Extended API timeout to 2 minutes for WebM processing
   - Disabled client-side conversion (backend handles all formats)

3. **Performance Improvements**:
   - Larger audio chunks reduce server load
   - Better error recovery for failed chunks
   - Proper WebM file extension detection
   - Cleanup of temporary converted files

### Current Status
- ✅ Backend threading issues resolved
- ✅ WebM processing timeout handling improved
- ✅ Larger chunk sizes reduce processing frequency
- ✅ Multiple audio loading fallbacks prevent failures
- 🔄 Requires FFmpeg on server for optimal WebM conversion
