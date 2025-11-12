/**
 * Gradio Client Service for Twi Speech Recognition
 * ===============================================
 *
 * This service provides integration with the Gradio-hosted speech recognition model.
 * It can be toggled via environment variables to switch between local API and Gradio.
 *
 * Usage:
 * - Set NEXT_PUBLIC_USE_GRADIO=true to enable Gradio client
 * - Set NEXT_PUBLIC_GRADIO_SPACE_ID to specify the HuggingFace Space
 */

import { client } from "@gradio/client";
import type { IntentResult, IntentPrediction } from './api';

// Configuration
const USE_GRADIO = process.env.NEXT_PUBLIC_USE_GRADIO === 'true';
const GRADIO_SPACE_ID = process.env.NEXT_PUBLIC_GRADIO_SPACE_ID || 'TwiWhisperModel/TwiSpeechModel';
const GRADIO_HF_TOKEN = process.env.NEXT_PUBLIC_HF_TOKEN; // Optional for private spaces
const DEBUG_MODE = process.env.NEXT_PUBLIC_DEBUG_MODE === 'true';

// Log configuration in debug mode
if (DEBUG_MODE) {
  console.log('🎭 Gradio Configuration:', {
    USE_GRADIO,
    GRADIO_SPACE_ID,
    HAS_TOKEN: !!GRADIO_HF_TOKEN
  });
}

export interface GradioClientOptions {
  timeoutMs?: number;
  retries?: number;
}

export interface GradioResponse {
  data: [string, string, string]; // [transcription, intent_info, status]
}

export class GradioClient {
  private static instance: GradioClient | null = null;
  private clientPromise: Promise<any> | null = null;

  private constructor() { }

  static getInstance(): GradioClient {
    if (!GradioClient.instance) {
      GradioClient.instance = new GradioClient();
    }
    return GradioClient.instance;
  }

  /**
   * Initialize the Gradio client with connection pooling
   */
  private async getClient() {
    if (!this.clientPromise) {
      this.clientPromise = this.createClient();
    }
    return this.clientPromise;
  }

  private async createClient() {
    try {
      const clientOptions: any = {};

      // Add HuggingFace token if available
      if (GRADIO_HF_TOKEN) {
        clientOptions.hf_token = GRADIO_HF_TOKEN;
      }

      if (DEBUG_MODE) {
        console.log(`🔌 Connecting to Gradio space: ${GRADIO_SPACE_ID}`);
      }

      const app = await client(GRADIO_SPACE_ID, clientOptions);

      if (DEBUG_MODE) {
        console.log('✅ Gradio client connected successfully');
      }

      return app;
    } catch (error) {
      console.error('❌ Failed to connect to Gradio space:', error);
      // Reset promise so we can retry
      this.clientPromise = null;
      throw new Error(`Failed to connect to Gradio space ${GRADIO_SPACE_ID}: ${error}`);
    }
  }

  /**
   * Test if Gradio client is available and working
   */
  async testConnection(): Promise<boolean> {
    try {
      const app = await this.getClient();
      // Try to get the API info to verify connection
      return !!app;
    } catch (error) {
      console.error('🔍 Gradio connection test failed:', error);
      return false;
    }
  }

  /**
   * Get health status for Gradio (simulated)
   */
  async getHealth(): Promise<any> {
    if (!USE_GRADIO) {
      throw new Error('Gradio client is disabled');
    }

    try {
      const isConnected = await this.testConnection();
      return {
        status: isConnected ? 'healthy' : 'unhealthy',
        service: 'gradio',
        space_id: GRADIO_SPACE_ID,
        uptime_seconds: 0, // Not available from Gradio
        detail: isConnected ? 'Gradio space is responsive' : 'Cannot connect to Gradio space'
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        service: 'gradio',
        space_id: GRADIO_SPACE_ID,
        error: (error as Error).message
      };
    }
  }

  /**
   * Get model info for Gradio (simulated)
   */
  async getModelInfo(): Promise<any> {
    if (!USE_GRADIO) {
      throw new Error('Gradio client is disabled');
    }

    try {
      const spaceInfo = await this.getSpaceInfo();
      return {
        model_type: 'whisper-gradio',
        model_name: GRADIO_SPACE_ID,
        version: '1.0.0',
        num_classes: 49, // Based on your intent labels
        service: 'gradio',
        space_id: GRADIO_SPACE_ID,
        status: spaceInfo.status,
        capabilities: ['transcription', 'intent_classification'],
        supported_formats: ['wav', 'mp3', 'webm', 'm4a'],
        max_duration_seconds: 30
      };
    } catch (error) {
      throw new Error(`Failed to get Gradio model info: ${(error as Error).message}`);
    }
  }

  /**
   * Process audio file through Gradio speech recognition
   */
  async processAudio(
    audioBlob: Blob | File,
    options: GradioClientOptions = {}
  ): Promise<IntentResult> {
    const { timeoutMs = 60000, retries = 3 } = options;

    if (!USE_GRADIO) {
      throw new Error('Gradio client is disabled. Set NEXT_PUBLIC_USE_GRADIO=true to enable.');
    }

    let lastError: Error | null = null;

    for (let attempt = 1; attempt <= retries; attempt++) {
      try {
        if (DEBUG_MODE) {
          console.log(`🎤 Processing audio with Gradio (attempt ${attempt}/${retries})...`);
          console.log(`📁 File size: ${audioBlob.size} bytes`);
          console.log(`📁 File type: ${audioBlob.type || 'unknown'}`);
        }

        const app = await this.getClient();

        // Create a timeout promise
        const timeoutPromise = new Promise((_, reject) => {
          setTimeout(() => reject(new Error(`Request timeout after ${timeoutMs}ms`)), timeoutMs);
        });

        // Make the prediction
        const predictionPromise = app.predict("/predict", [audioBlob]);

        // Race between prediction and timeout
        const result: GradioResponse = await Promise.race([predictionPromise, timeoutPromise]);

        if (DEBUG_MODE) {
          console.log('🎭 Raw Gradio response:', result);
        }

        // Parse the response
        const [transcription, intentInfo, status] = result.data;

        if (status === 'Error') {
          throw new Error(`Gradio processing failed: ${intentInfo}`);
        }

        // Parse intent information
        let intent = 'unknown';
        let confidence = 0;
        let predictions: IntentPrediction[] = [];

        if (intentInfo && intentInfo.includes('Intent:')) {
          // Parse "Intent: <intent> (Confidence: <score>)" format
          const match = intentInfo.match(/Intent:\s*(\w+)\s*\(Confidence:\s*([\d.]+)\)/);
          if (match) {
            intent = match[1];
            confidence = parseFloat(match[2]);

            predictions = [{
              intent,
              confidence
            }];
          }
        }

        const result_data: IntentResult = {
          transcription: transcription || '',
          intent,
          confidence,
          predictions,
          processing_time_ms: 0, // Gradio doesn't provide this
          model_info: {
            transcription_model: GRADIO_SPACE_ID,
            intent_model: GRADIO_SPACE_ID,
            version: 'gradio'
          },
          audio_info: {
            duration_seconds: 0, // Would need to calculate this
            sample_rate: 16000, // Assumed
            channels: 1 // Assumed
          }
        };

        if (DEBUG_MODE) {
          console.log('✅ Gradio processing successful:', result_data);
        }

        return result_data;

      } catch (error) {
        lastError = error as Error;
        console.error(`❌ Gradio attempt ${attempt}/${retries} failed:`, error);

        if (attempt < retries) {
          // Exponential backoff: 1s, 2s, 4s
          const delay = Math.pow(2, attempt - 1) * 1000;
          console.log(`⏳ Retrying in ${delay}ms...`);
          await new Promise(resolve => setTimeout(resolve, delay));

          // Reset client connection on retry
          this.clientPromise = null;
        }
      }
    }

    throw new Error(`Gradio processing failed after ${retries} attempts: ${lastError?.message}`);
  }

  /**
   * Get space information
   */
  async getSpaceInfo(): Promise<any> {
    try {
      const app = await this.getClient();
      return {
        space_id: GRADIO_SPACE_ID,
        status: 'connected',
        // Additional info could be extracted from the app object
      };
    } catch (error) {
      return {
        space_id: GRADIO_SPACE_ID,
        status: 'error',
        error: (error as Error).message
      };
    }
  }

  /**
   * Reset client connection (useful for error recovery)
   */
  resetConnection(): void {
    this.clientPromise = null;
    if (DEBUG_MODE) {
      console.log('🔄 Gradio client connection reset');
    }
  }
}

// Export singleton instance
export const gradioClient = GradioClient.getInstance();

// Export utility functions
export function isGradioEnabled(): boolean {
  return USE_GRADIO;
}

export function getGradioConfig() {
  return {
    enabled: USE_GRADIO,
    spaceId: GRADIO_SPACE_ID,
    hasToken: !!GRADIO_HF_TOKEN
  };
}

/**
 * Convenience function to process audio with automatic fallback
 * This will use Gradio if enabled, otherwise throw an error to fall back to local API
 */
export async function processAudioWithGradio(
  audioBlob: Blob | File,
  options?: GradioClientOptions
): Promise<IntentResult> {
  if (!USE_GRADIO) {
    throw new Error('Gradio is disabled');
  }

  return gradioClient.processAudio(audioBlob, options);
}

// Export for backward compatibility
export default gradioClient;
