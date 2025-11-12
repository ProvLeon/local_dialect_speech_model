/**
 * API Wrapper for Gradio-First Architecture
 * ========================================
 *
 * This wrapper ensures that when Gradio is enabled, we don't make unnecessary
 * calls to the local API that will fail due to CORS or unavailability.
 *
 * It provides the same interface as the original API but with intelligent
 * routing based on configuration.
 */

import {
  getHealth as originalGetHealth,
  getModelInfo as originalGetModelInfo,
  testIntent as originalTestIntent,
  type HealthStatus,
  type ModelInfo,
  type IntentResult,
  type FetchOptions
} from './api';
import {
  isGradioEnabled,
  gradioClient,
  processAudioWithGradio
} from './gradio-client';

const DEBUG_MODE = process.env.NEXT_PUBLIC_DEBUG_MODE === 'true';

/**
 * Smart health check that respects Gradio-first configuration
 */
export async function getHealth(opts?: FetchOptions): Promise<HealthStatus> {
  if (isGradioEnabled()) {
    try {
      if (DEBUG_MODE) {
        console.log('🎭 [Wrapper] Using Gradio health check (primary)');
      }

      const gradioHealth = await gradioClient.getHealth();

      if (DEBUG_MODE) {
        console.log('✅ [Wrapper] Gradio health check successful');
      }

      return gradioHealth;
    } catch (error) {
      if (DEBUG_MODE) {
        console.warn('⚠️ [Wrapper] Gradio health failed, skipping local API to avoid CORS errors');
      }

      // Return a synthetic "unhealthy" status instead of trying local API
      // This prevents CORS errors when local server isn't running
      return {
        status: 'unhealthy',
        service: 'gradio-primary',
        detail: 'Gradio space unavailable, local API not attempted to avoid CORS',
        error: (error as Error).message
      };
    }
  }

  // Only use local API if Gradio is explicitly disabled
  if (DEBUG_MODE) {
    console.log('🏥 [Wrapper] Using local API health check');
  }

  return originalGetHealth(opts);
}

/**
 * Smart model info that respects Gradio-first configuration
 */
export async function getModelInfo(opts?: FetchOptions): Promise<ModelInfo | null> {
  if (isGradioEnabled()) {
    try {
      if (DEBUG_MODE) {
        console.log('🎭 [Wrapper] Using Gradio model info (primary)');
      }

      const gradioModelInfo = await gradioClient.getModelInfo();

      if (DEBUG_MODE) {
        console.log('✅ [Wrapper] Gradio model info successful');
      }

      return gradioModelInfo;
    } catch (error) {
      if (DEBUG_MODE) {
        console.warn('⚠️ [Wrapper] Gradio model info failed, returning synthetic info');
      }

      // Return synthetic model info instead of trying local API
      return {
        model_type: 'gradio-unavailable',
        model_name: 'Gradio Space (Unavailable)',
        version: '1.0.0',
        num_classes: 49,
        service: 'gradio',
        status: 'error',
        error: (error as Error).message
      };
    }
  }

  // Only use local API if Gradio is explicitly disabled
  if (DEBUG_MODE) {
    console.log('📋 [Wrapper] Using local API model info');
  }

  return originalGetModelInfo(opts);
}

/**
 * Smart intent testing with proper Gradio-only logic
 */
export async function testIntent(
  input: File | Blob | FormData,
  topK = 5,
  opts?: FetchOptions
): Promise<IntentResult> {
  if (DEBUG_MODE) {
    console.log('🎯 [Wrapper] Processing intent with smart routing');
  }

  if (isGradioEnabled()) {
    try {
      if (DEBUG_MODE) {
        console.log('🎭 [Wrapper] Using Gradio intent processing (no local fallback)');
      }

      // Convert FormData to Blob if needed
      let audioBlob: Blob | File;
      if (input instanceof FormData) {
        const fileEntry = input.get('file');
        if (fileEntry instanceof File || fileEntry instanceof Blob) {
          audioBlob = fileEntry;
        } else {
          throw new Error('Invalid FormData: no file found');
        }
      } else {
        audioBlob = input;
      }

      const gradioResult = await processAudioWithGradio(audioBlob, {
        timeoutMs: opts?.timeoutMs || 120000,
        retries: 2
      });

      if (DEBUG_MODE) {
        console.log('✅ [Wrapper] Gradio intent processing successful');
      }

      return gradioResult;
    } catch (error) {
      if (DEBUG_MODE) {
        console.error('❌ [Wrapper] Gradio intent processing failed (no local fallback in Gradio-only mode):', error);
      }

      throw new Error(`Gradio processing failed: ${(error as Error).message}`);
    }
  }

  // Only use local API if Gradio is explicitly disabled
  if (DEBUG_MODE) {
    console.log('🏠 [Wrapper] Using local API intent processing');
  }

  return originalTestIntent(input, topK, opts);
}

/**
 * Smart live chunk uploader that disables streaming when Gradio is enabled
 */
export function createLiveChunkUploader(options: any) {
  if (isGradioEnabled()) {
    // Gradio doesn't support streaming, so we return a mock uploader
    if (DEBUG_MODE) {
      console.warn('⚠️ [Wrapper] Live streaming disabled in Gradio mode');
    }

    return {
      uploadChunk: () => {
        if (options.onError) {
          options.onError({
            error: 'Live streaming is not supported in Gradio mode. Use file upload instead.',
            status: 'disabled',
            reason: 'gradio_no_streaming'
          });
        }
      },
      abort: () => {
        if (DEBUG_MODE) {
          console.log('🛑 [Wrapper] Mock uploader abort called');
        }
      }
    };
  }

  // Use original live uploader for local API
  if (DEBUG_MODE) {
    console.log('🔴 [Wrapper] Using local API live streaming');
  }

  // Import the original function dynamically to avoid circular imports
  const { createLiveChunkUploader: originalCreateLiveChunkUploader } = require('./api');
  return originalCreateLiveChunkUploader(options);
}

/**
 * Configuration helper to check what mode we're in
 */
export function getApiMode(): 'gradio-primary' | 'local-primary' | 'gradio-only' {
  const gradioEnabled = isGradioEnabled();

  if (gradioEnabled) {
    // In Gradio mode, we don't fall back to local to avoid CORS
    return 'gradio-only';
  }

  return 'local-primary';
}

/**
 * Check if we should skip local API calls entirely
 */
export function shouldSkipLocalAPI(): boolean {
  return getApiMode() === 'gradio-only';
}

/**
 * Get current configuration summary
 */
export function getConfigSummary() {
  const mode = getApiMode();
  const gradioEnabled = isGradioEnabled();

  return {
    mode,
    gradioEnabled,
    willUseCORS: !gradioEnabled, // Only use CORS when using local API
    primaryService: gradioEnabled ? 'gradio' : 'local',
    fallbackService: gradioEnabled ? 'none' : 'none',
    debugMode: DEBUG_MODE
  };
}

// Re-export everything else from the original API
export * from './api';

// Override the specific functions that need smart routing
export {
  getHealth,
  getModelInfo,
  testIntent,
  createLiveChunkUploader,
  getApiMode,
  shouldSkipLocalAPI,
  getConfigSummary
};
