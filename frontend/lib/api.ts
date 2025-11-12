/* eslint-disable @typescript-eslint/no-explicit-any */

/**
 * Centralized backend API client helpers for the Akan (Twi) Speech Intent system.
 *
 * This module is written to work in both:
 *  - Client components (browser)
 *  - Server actions / RSC / edge (no direct DOM assumptions)
 *
 * It provides:
 *  - Typed wrappers for core endpoints: /health, /model-info, /test-intent
 *  - Safe fetch with:
 *      * Abort + timeout
 *      * Automatic JSON parsing
 *      * Structured error objects
 *      * Optional retry for idempotent GETs
 *  - File / Blob handling convenience (auto-FormData)
 *  - "Pseudo-streaming" helpers for chunked uploads to simulate ChatGPT‑like incremental UI
 *  - Event-driven and async-iterator streaming abstractions
 *  - Gradio client integration with automatic fallback
 *
 * IMPORTANT:
 *  If you introduce a proper streaming backend endpoint later (e.g. Server-Sent Events or WebSocket),
 *  you can extend this client by adding a specialized transport while keeping the same high-level interface.
 */

import { isGradioEnabled, getGradioConfig, processAudioWithGradio, gradioClient } from './gradio-client';

/* -------------------------------------------------------------------------- */
/* Configuration & Types                                                      */
/* -------------------------------------------------------------------------- */

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/+$/, '') || 'http://localhost:8000';

// Debug mode for development
const DEBUG_MODE = process.env.NEXT_PUBLIC_DEBUG_MODE === 'true';

// Log API configuration
if (DEBUG_MODE) {
  console.log('🔧 API Configuration:', {
    API_BASE_URL,
    NODE_ENV: process.env.NODE_ENV,
    DEBUG_MODE,
    USE_GRADIO: isGradioEnabled(),
    GRADIO_CONFIG: getGradioConfig()
  });
}

export interface HealthStatus {
  status: string;
  detail?: string;
  uptime_seconds?: number;
  [k: string]: any;
}

export interface ModelInfo {
  model_type: string;
  num_classes: number;
  input_dim: number;
  version?: string;
  build?: string;
  [k: string]: any;
}

export interface IntentPrediction {
  intent: string;
  confidence: number; // 0–1
}

export interface IntentResult {
  intent: string;
  confidence: number;
  top_predictions: IntentPrediction[];
  filename: string;
  model_type?: string;
  processing_time_ms?: number;
  [k: string]: any;
}

export interface ApiErrorShape {
  name: string;
  message: string;
  status?: number;
  cause?: unknown;
  data?: any;
  url?: string;
  meta?: Record<string, any>;
}

export class ApiError extends Error implements ApiErrorShape {
  status?: number;
  cause?: unknown;
  data?: any;
  url?: string;
  meta?: Record<string, any>;

  constructor(init: ApiErrorShape) {
    super(init.message);
    this.name = init.name || 'ApiError';
    this.status = init.status;
    this.cause = init.cause;
    this.data = init.data;
    this.url = init.url;
    this.meta = init.meta;
  }
}

export interface FetchOptions extends Omit<RequestInit, 'body'> {
  /**
   * If provided, convert (File | Blob | FormData) automatically to a suitable body.
   */
  body?: BodyInit | FormData | Blob | File | null;
  /**
   * Abort the request after this many milliseconds (default 15_000).
   */
  timeoutMs?: number;
  /**
   * Number of retry attempts (only for GET + idempotent).
   */
  retry?: number;
  /**
   * Exponential backoff base (ms).
   */
  retryBaseDelayMs?: number;
  /**
   * Accept text instead of JSON (auto-detected if response is not application/json).
   */
  rawTextOk?: boolean;
  /**
   * Custom error handler transform (e.g., shape backend error).
   */
  mapError?: (e: ApiError) => ApiError;
  /**
   * If true, skip throwing on non-2xx and return raw Response.
   */
  noThrow?: boolean;
}

export interface SafeFetchResponse<T = unknown> {
  ok: boolean;
  status: number;
  url: string;
  headers: Headers;
  data: T;
  raw: Response;
  durationMs: number;
}

/* -------------------------------------------------------------------------- */
/* Low-level Safe Fetch Wrapper                                              */
/* -------------------------------------------------------------------------- */

export async function safeFetch<T = any>(
  path: string,
  {
    timeoutMs = 15_000,
    retry = 0,
    retryBaseDelayMs = 300,
    rawTextOk = true,
    mapError,
    noThrow = false,
    headers,
    body,
    method,
    ...init
  }: FetchOptions = {}
): Promise<SafeFetchResponse<T>> {
  const url = path.startsWith('http') ? path : `${API_BASE_URL}${path.startsWith('/') ? '' : '/'}${path}`;

  if (DEBUG_MODE) {
    console.log('🌐 API Request:', { url, method: method || (body ? 'POST' : 'GET') });
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  const finalMethod = (method || (body ? 'POST' : 'GET')).toUpperCase();
  const canRetry = finalMethod === 'GET' || finalMethod === 'HEAD';

  let attempt = 0;
  let lastError: unknown;

  const start = performance.now();

  while (true) {
    attempt++;

    try {
      // FormData auto-handling (if passing File / Blob directly)
      let finalBody: BodyInit | undefined;
      let finalHeaders: HeadersInit | undefined = headers;

      if (body instanceof FormData) {
        finalBody = body;
      } else if (body instanceof File || body instanceof Blob) {
        const fd = new FormData();
        fd.append('file', body, (body as File).name || 'audio.wav');
        finalBody = fd;
      } else if (
        body != null &&
        typeof body === 'object' &&
        !(body instanceof ArrayBuffer) &&
        !(body instanceof URLSearchParams)
      ) {
        // JSON
        finalHeaders = { 'Content-Type': 'application/json', ...(headers || {}) };
        finalBody = JSON.stringify(body);
      } else {
        finalBody = body as BodyInit | undefined;
      }

      const response = await fetch(url, {
        method: finalMethod,
        body: finalBody,
        signal: controller.signal,
        headers: {
          'Accept': 'application/json',
          'Access-Control-Allow-Origin': '*',
          ...finalHeaders
        },
        mode: 'cors',
        credentials: 'omit',
        ...init
      });

      if (DEBUG_MODE) {
        console.log('📡 API Response:', {
          url,
          status: response.status,
          statusText: response.statusText,
          headers: Object.fromEntries(response.headers.entries())
        });
      }

      const contentType = response.headers.get('content-type') || '';

      let data: any;
      if (/application\/json/i.test(contentType)) {
        data = await response.json().catch(() => null);
      } else if (rawTextOk) {
        data = await response.text().catch(() => null);
      } else {
        data = await response.arrayBuffer().catch(() => null);
      }

      clearTimeout(timer);
      const durationMs = performance.now() - start;

      if (!response.ok && !noThrow) {
        const err = new ApiError({
          name: 'HttpError',
          message: `Request failed with status ${response.status}`,
          status: response.status,
          data,
          url
        });
        throw err;
      }

      return {
        ok: response.ok,
        status: response.status,
        url,
        headers: response.headers,
        data,
        raw: response,
        durationMs
      };
    } catch (err: any) {
      lastError = err?.name === 'AbortError'
        ? new ApiError({ name: 'TimeoutError', message: 'Request timed out', url, cause: err })
        : err;

      if (lastError instanceof ApiError === false) {
        lastError = new ApiError({
          name: 'NetworkError',
          message: err?.message || 'Network request failed',
          cause: err,
          url
        });
      }

      if (mapError) {
        lastError = mapError(lastError as ApiError);
      }

      const ae = lastError as ApiError;

      if (ae.name === 'TimeoutError') {
        if (!noThrow) throw ae;
        return {
          ok: false,
          status: ae.status ?? 0,
          url,
          headers: new Headers(),
          data: null as any,
          raw: new Response(null, { status: ae.status ?? 0 }),
          durationMs: performance.now() - start
        };
      }

      if (attempt > retry || !canRetry) {
        if (!noThrow) throw ae;
        return {
          ok: false,
          status: ae.status ?? 0,
          url,
          headers: new Headers(),
          data: ae.data as any,
          raw: new Response(null, { status: ae.status ?? 0 }),
          durationMs: performance.now() - start
        };
      }

      // Exponential backoff
      const delay = retryBaseDelayMs * Math.pow(2, attempt - 1);
      await new Promise(r => setTimeout(r, delay));
    }
  }
}

/* -------------------------------------------------------------------------- */
/* High-level API Methods                                                    */
/* -------------------------------------------------------------------------- */

export async function getHealth(opts?: FetchOptions): Promise<HealthStatus> {
  // If Gradio is enabled, try Gradio first
  if (isGradioEnabled()) {
    try {
      if (DEBUG_MODE) {
        console.log('🎭 Getting health status from Gradio...');
      }

      const gradioHealth = await gradioClient.getHealth();

      if (DEBUG_MODE) {
        console.log('✅ Gradio health check successful');
      }

      return gradioHealth;
    } catch (gradioError) {
      console.warn('⚠️ Gradio health check failed, falling back to local API:', gradioError);
      // Continue to local API fallback
    }
  }

  // Local API health check (original implementation)
  if (DEBUG_MODE) {
    console.log('🏥 Getting health status from local API...');
  }

  const { data } = await safeFetch<HealthStatus>('/health', { retry: 2, ...opts });
  return data;
}

export async function getModelInfo(opts?: FetchOptions): Promise<ModelInfo | null> {
  // If Gradio is enabled, try Gradio first
  if (isGradioEnabled()) {
    try {
      if (DEBUG_MODE) {
        console.log('🎭 Getting model info from Gradio...');
      }

      const gradioModelInfo = await gradioClient.getModelInfo();

      if (DEBUG_MODE) {
        console.log('✅ Gradio model info successful');
      }

      return gradioModelInfo;
    } catch (gradioError) {
      console.warn('⚠️ Gradio model info failed, falling back to local API:', gradioError);
      // Continue to local API fallback
    }
  }

  // Local API model info (original implementation)
  try {
    if (DEBUG_MODE) {
      console.log('📋 Getting model info from local API...');
    }

    const { data } = await safeFetch<ModelInfo>('/model-info', { retry: 2, ...opts });
    return data;
  } catch {
    // Model info may not exist—fail gracefully.
    return null;
  }
}

/**
 * Test intent recognition with a file / blob / FormData.
 * Supports both Gradio and local API with automatic fallback.
 *
 * @param input - File | Blob | FormData
 * @param topK   - Top predictions to request
 */
export async function testIntent(
  input: File | Blob | FormData,
  topK = 5,
  opts?: FetchOptions
): Promise<IntentResult> {
  // If Gradio is enabled, try Gradio first
  if (isGradioEnabled()) {
    try {
      if (DEBUG_MODE) {
        console.log('🎭 Attempting Gradio processing first...');
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
        console.log('✅ Gradio processing successful');
      }

      return gradioResult;
    } catch (gradioError) {
      console.warn('⚠️ Gradio processing failed, falling back to local API:', gradioError);

      // Continue to local API fallback
    }
  }

  // Local API processing (original implementation)
  let fd: FormData;

  if (input instanceof FormData) {
    fd = input;
  } else {
    fd = new FormData();
    const filename = (input as File).name || 'audio.wav';
    fd.append('file', input, filename);

    if (DEBUG_MODE) {
      console.log('🎤 Preparing local API intent test:', {
        fileSize: input.size,
        fileType: input.type,
        filename,
        topK,
        usingGradio: false
      });
    }
  }

  const query = new URLSearchParams({ top_k: String(topK) });
  const url = `/test-intent?${query.toString()}`;

  try {
    const { data } = await safeFetch<IntentResult>(url, {
      method: 'POST',
      body: fd,
      timeoutMs: 120000, // 2 minute timeout for WebM audio processing
      ...opts
    });

    if (DEBUG_MODE) {
      console.log('🎯 Local API result:', {
        intent: data.intent,
        confidence: data.confidence,
        predictions: data.top_predictions?.length || 0
      });
    }

    return data;
  } catch (error) {
    console.error('💥 Intent test failed (local API):', {
      url: `${API_BASE_URL}${url}`,
      error: error instanceof ApiError ? {
        status: error.status,
        message: error.message,
        data: error.data
      } : String(error)
    });
    throw error;
  }
}

/* -------------------------------------------------------------------------- */
/* Chat-like / Streaming Simulation Helpers                                  */
/* -------------------------------------------------------------------------- */

/**
 * Represents an incremental update for streaming-style recognition.
 * This mimics ChatGPT token events but uses chunk-level results.
 */
export interface StreamingIntentEvent {
  chunkIndex: number;
  timestamp: number;
  intent: string;
  confidence: number;
  top_predictions?: IntentPrediction[];
  raw?: any;
  done?: boolean;
  error?: string;
}

export interface StreamOptions {
  /**
   * Milliseconds per pseudo-chunk slice when breaking up a single file.
   * Only used if you provide a full audio file and want incremental UI.
   */
  msPerSlice?: number;
  /**
   * Top K predictions per request.
   */
  topK?: number;
  /**
   * Abort controller to allow stopping mid-process.
   */
  signal?: AbortSignal;
  /**
   * Called after each chunk upload resolves.
   */
  onEvent?: (evt: StreamingIntentEvent) => void;
  /**
   * Provide a custom file naming strategy.
   */
  filenameBuilder?: (index: number) => string;
  /**
   * Transform each server response before emitting.
   */
  transformResult?: (r: IntentResult, index: number) => Partial<StreamingIntentEvent>;
}

/**
 * Slice a Blob into time-based chunks using approximate duration metadata.
 * NOTE: This uses size heuristics and an assumed bitrate when duration isn't available.
 *
 * If you are recording live via MediaRecorder, prefer sending each chunk directly
 * instead of using this helper.
 */
export async function sliceBlobByApproxDuration(
  blob: Blob,
  msPerSlice: number
): Promise<Blob[]> {
  // Without decoding we can't know exact duration. We approximate using a nominal bitrate.
  // For WAV PCM16 mono 16kHz: 16000 samples * 2 bytes ≈ 32 KB per second ≈ 32 bytes/ms.
  // We'll pick a conservative bytes/ms fallback.
  const DEFAULT_BYTES_PER_MS = 32;
  const targetBytes = msPerSlice * DEFAULT_BYTES_PER_MS;

  if (targetBytes <= 0 || blob.size <= targetBytes) {
    return [blob];
  }

  const parts: Blob[] = [];
  let offset = 0;
  while (offset < blob.size) {
    const end = Math.min(blob.size, offset + targetBytes);
    parts.push(blob.slice(offset, end));
    offset = end;
  }
  return parts;
}

/**
 * Pseudo-stream a single Blob/File by chunking and sending sequentially to the same endpoint,
 * emitting updates for a ChatGPT-like incremental experience.
 *
 * Returns an async iterator you can consume OR you can use onEvent callback.
 */
export async function* streamIntentFromBlob(
  blob: Blob,
  {
    msPerSlice = 1_000,
    topK = 5,
    signal,
    onEvent,
    filenameBuilder = (i) => `chunk_${i}.wav`,
    transformResult
  }: StreamOptions = {}
): AsyncGenerator<StreamingIntentEvent, void, void> {
  const chunks = await sliceBlobByApproxDuration(blob, msPerSlice);
  let index = 0;

  for (const c of chunks) {
    if (signal?.aborted) {
      const abortedEvt: StreamingIntentEvent = {
        chunkIndex: index,
        timestamp: Date.now(),
        intent: '',
        confidence: 0,
        error: 'aborted',
        done: true
      };
      onEvent?.(abortedEvt);
      yield abortedEvt;
      return;
    }

    const fd = new FormData();
    fd.append('file', c, filenameBuilder(index));

    let event: StreamingIntentEvent;

    try {
      const result = await testIntent(fd, topK, { timeoutMs: 30_000, retry: 1 });
      event = {
        chunkIndex: index,
        timestamp: Date.now(),
        intent: result.intent,
        confidence: result.confidence,
        top_predictions: result.top_predictions,
        raw: result,
        ...(transformResult?.(result, index) || {})
      };
    } catch (e: any) {
      event = {
        chunkIndex: index,
        timestamp: Date.now(),
        intent: '',
        confidence: 0,
        error: e?.message || 'Upload failed'
      };
    }

    onEvent?.(event);
    yield event;
    index++;
  }

  const finalEvt: StreamingIntentEvent = {
    chunkIndex: index,
    timestamp: Date.now(),
    intent: '',
    confidence: 0,
    done: true
  };
  onEvent?.(finalEvt);
  yield finalEvt;
}

/**
 * Helper to consume the async generator and provide a simple callback interface.
 */
export function streamIntentFromBlobCallback(
  blob: Blob,
  opts: StreamOptions
): { abort: () => void; done: Promise<void> } {
  const controller = new AbortController();
  const finalSignal =
    opts.signal
      ? mergeAbortSignals(opts.signal, controller.signal)
      : controller.signal;

  const done = (async () => {
    for await (const event of streamIntentFromBlob(blob, { ...opts, signal: finalSignal })) {
      // iteration side-effects handled via onEvent
      void event; // Mark as intentionally unused
    }
  })();

  return {
    abort: () => controller.abort(),
    done
  };
}

/**
 * Merge two abort signals into a derived signal.
 */
function mergeAbortSignals(a: AbortSignal, b: AbortSignal): AbortSignal {
  if (a.aborted) return a;
  if (b.aborted) return b;
  const controller = new AbortController();
  const onAbortA = () => controller.abort();
  const onAbortB = () => controller.abort();
  a.addEventListener('abort', onAbortA);
  b.addEventListener('abort', onAbortB);
  controller.signal.addEventListener('abort', () => {
    a.removeEventListener('abort', onAbortA);
    b.removeEventListener('abort', onAbortB);
  });
  return controller.signal;
}

/* -------------------------------------------------------------------------- */
/* Live Chunk Uploader (for MediaRecorder ondataavailable flows)             */
/* -------------------------------------------------------------------------- */

export interface LiveUploaderOptions {
  topK?: number;
  onResult?: (e: StreamingIntentEvent) => void;
  onError?: (e: ApiError) => void;
  transformResult?: (r: IntentResult, i: number) => Partial<StreamingIntentEvent>;
  filenameBuilder?: (i: number) => string;
}

export function createLiveChunkUploader({
  topK = 5,
  onResult,
  onError,
  transformResult,
  filenameBuilder = (i) => `live_${i}.wav`
}: LiveUploaderOptions = {}) {
  let index = 0;
  let aborted = false;

  return {
    async push(blob: Blob | File) {
      if (aborted) return;
      const fd = new FormData();
      const filename = (blob as File).name || filenameBuilder(index);
      fd.append('file', blob, filename);

      // Debug logging for audio upload
      if (DEBUG_MODE) {
        console.log(`🎵 Uploading audio chunk ${index}:`, {
          size: blob.size,
          type: blob.type,
          filename,
          topK
        });
      }

      try {
        const result = await testIntent(fd, topK);
        const evt: StreamingIntentEvent = {
          chunkIndex: index,
          timestamp: Date.now(),
          intent: result.intent,
          confidence: result.confidence,
          top_predictions: result.top_predictions,
          raw: result,
          ...(transformResult?.(result, index) || {})
        };

        if (DEBUG_MODE) {
          console.log(`✅ Audio chunk ${index} processed:`, {
            intent: result.intent,
            confidence: result.confidence,
            predictions: result.top_predictions?.length || 0
          });
        }

        onResult?.(evt);
      } catch (e: any) {
        console.error(`❌ Audio chunk ${index} failed:`, {
          error: e?.message,
          status: e?.status,
          blobSize: blob.size,
          blobType: blob.type,
          apiError: e instanceof ApiError ? e.data : null
        });

        const apiErr = e instanceof ApiError ? e : new ApiError({
          name: 'UploadError',
          message: e?.message || 'Unknown error',
          cause: e
        });
        onError?.(apiErr);
      } finally {
        index++;
      }
    },
    abort() {
      aborted = true;
    },
    get index() {
      return index;
    },
    get aborted() {
      return aborted;
    }
  };
}

/* -------------------------------------------------------------------------- */
/* Utility Formatters                                                        */
/* -------------------------------------------------------------------------- */

export function formatConfidence(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function classifyConfidenceTier(confidence: number): 'high' | 'medium' | 'low' {
  if (confidence >= 0.8) return 'high';
  if (confidence >= 0.5) return 'medium';
  return 'low';
}

export function humanFileSize(bytes: number): string {
  if (!bytes && bytes !== 0) return '';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
}

/* -------------------------------------------------------------------------- */
/* Aggregation / Summary Helpers                                             */
/* -------------------------------------------------------------------------- */

export interface StreamingSummary {
  totalChunks: number;
  mostFrequentIntent: string | null;
  averageConfidence: number;
  distribution: Record<string, number>;
}

export function summarizeStreamingEvents(events: StreamingIntentEvent[]): StreamingSummary {
  const distribution: Record<string, number> = {};
  let sum = 0;
  let count = 0;
  for (const e of events) {
    if (!e.intent || e.error) continue;
    distribution[e.intent] = (distribution[e.intent] || 0) + 1;
    sum += e.confidence;
    count++;
  }
  const mostFrequentIntent =
    Object.keys(distribution).reduce(
      (acc, cur) => (distribution[cur] > (distribution[acc] || 0) ? cur : acc),
      '' as string
    ) || null;

  return {
    totalChunks: events.filter(e => !e.error && e.intent).length,
    mostFrequentIntent,
    averageConfidence: count ? sum / count : 0,
    distribution
  };
}

/* -------------------------------------------------------------------------- */
/* Export Aggregate                                                          */
/* -------------------------------------------------------------------------- */

const api = {
  API_BASE_URL,
  safeFetch,
  getHealth,
  getModelInfo,
  testIntent,
  streamIntentFromBlob,
  streamIntentFromBlobCallback,
  createLiveChunkUploader,
  summarizeStreamingEvents,
  formatConfidence,
  classifyConfidenceTier,
  humanFileSize
};

export default api;
