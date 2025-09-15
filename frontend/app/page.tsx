"use client";

/**
 * Refactored professional Chat-style interface with:
 *  - Lucide icons replacing emojis
 *  - Improved semantic layout & typography
 *  - Dark/Light adaptive tokens (see globals.css)
 *  - Accessible controls and clearer visual hierarchy
 *  - Stream & live recording modes retained
 *
 * NOTE: Relies on design tokens defined in globals.css (HSL vars).
 */

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type KeyboardEvent,
} from "react";
import {
  API_BASE_URL,
  getHealth,
  getModelInfo,
  testIntent,
  createLiveChunkUploader,
  streamIntentFromBlobCallback,
  formatConfidence,
  summarizeStreamingEvents,
  classifyConfidenceTier,
  type IntentResult,
  type StreamingIntentEvent,
} from "../lib/api";
import {
  Mic,
  Square,
  Headphones,
  Radio,
  Upload,
  RefreshCcw,
  Send,
  FileAudio2,
  Brain,
  Activity,
  Sparkles,
  X,
  CheckCircle,
  AlertCircle,
  XCircle,
} from "lucide-react";

/* -------------------------------------------------------------------------- */
/* Types                                                                      */
/* -------------------------------------------------------------------------- */
type ChatMessage =
  | {
    id: string;
    role: "user";
    type: "prompt";
    content: string;
    createdAt: number;
  }
  | {
    id: string;
    role: "assistant";
    type: "intent-result";
    content: string;
    raw: IntentResult;
    createdAt: number;
  }
  | {
    id: string;
    role: "assistant";
    type: "stream-chunk";
    content: string;
    confidence: number;
    chunkIndex: number;
    createdAt: number;
    done?: boolean;
    error?: string;
  };

type Toast = {
  id: string;
  type: "success" | "error" | "info" | "warning";
  title: string;
  message?: string;
  duration?: number;
};

const SUPPORTED_EXTENSIONS = [".wav", ".mp3"];
const MAX_HISTORY = 500;

/* -------------------------------------------------------------------------- */
/* Component                                                                  */
/* -------------------------------------------------------------------------- */
export default function Home() {
  const [health, setHealth] = useState<string>("checking");
  const [modelInfo, setModelInfo] = useState<any>(null);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const liveUploaderRef = useRef<ReturnType<typeof createLiveChunkUploader> | null>(null);
  const recordingStartRef = useRef<number | null>(null);
  const [recordingElapsed, setRecordingElapsed] = useState<number>(0);
  const recordingTimerRef = useRef<number | null>(null);

  // Streaming session
  const [streamEvents, setStreamEvents] = useState<StreamingIntentEvent[]>([]);
  const [streamingMode, setStreamingMode] = useState<"live" | "chunk-file" | null>(null);
  const [topK, setTopK] = useState(5);

  const [error, setError] = useState<string | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);

  /* -------------------------------- Effects -------------------------------- */
  useEffect(() => {
    (async () => {
      try {
        const h = await getHealth();
        setHealth(h.status || "ok");
      } catch {
        setHealth("unreachable");
      }
      try {
        const mi = await getModelInfo();
        setModelInfo(mi);
      } catch {
        /* silent */
      }
    })();
  }, []);

  /* ------------------------------- Utilities -------------------------------- */
  const addToast = useCallback((toast: Omit<Toast, "id">) => {
    const id = `toast-${Date.now()}-${Math.random()}`;
    const newToast = { ...toast, id };
    setToasts((prev) => [...prev, newToast]);

    // Auto remove after duration
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, toast.duration || 4000);
  }, []);

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const pushMessage = useCallback((msg: ChatMessage) => {
    setMessages((prev) => {
      const next = [...prev, msg];
      if (next.length > MAX_HISTORY) return next.slice(-MAX_HISTORY);
      return next;
    });
  }, []);

  const updateStreamMessage = useCallback((evt: StreamingIntentEvent) => {
    // Handle streaming events with toasts instead of cluttering chat
    if (evt.done) {
      addToast({
        type: "success",
        title: "Streaming Complete",
        message: `Processed ${evt.chunkIndex} chunks`,
      });
      return;
    }

    if (evt.error) {
      addToast({
        type: "error",
        title: `Chunk ${evt.chunkIndex} Failed`,
        message: evt.error,
      });
      return;
    }

    // Only add meaningful intent results to chat
    if (evt.intent && evt.confidence > 0.3) {
      setMessages((prev) => {
        const id = `stream-${evt.chunkIndex}`;
        const idx = prev.findIndex((m) => m.id === id);
        const newMsg: ChatMessage = {
          id,
          role: "assistant",
          type: "stream-chunk",
          content: evt.intent,
          confidence: evt.confidence || 0,
          chunkIndex: evt.chunkIndex,
          createdAt: evt.timestamp,
          done: evt.done,
          error: evt.error,
        };
        if (idx >= 0) {
          const clone = [...prev];
          clone[idx] = newMsg;
          return clone;
        }
        return [...prev, newMsg];
      });
    }
  }, [addToast]);

  /* --------------------------- File Handling -------------------------------- */
  function handleFileSelect(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const lower = file.name.toLowerCase();
    const valid = SUPPORTED_EXTENSIONS.some((ext) => lower.endsWith(ext));
    if (!valid) {
      setError("Unsupported format. Please use WAV or MP3.");
      e.target.value = "";
      return;
    }
    setError(null);
    setSelectedFile(file);
  }

  async function submitAudioFile() {
    if (!selectedFile) {
      setError("Select an audio file first.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      pushMessage({
        id: `user-file-${Date.now()}`,
        role: "user",
        type: "prompt",
        content: `Uploaded audio: ${selectedFile.name}`,
        createdAt: Date.now(),
      });
      const result = await testIntent(selectedFile, topK, { timeoutMs: 60_000 });
      pushMessage({
        id: `intent-${Date.now()}`,
        role: "assistant",
        type: "intent-result",
        content: `Primary intent: ${result.intent}`,
        raw: result,
        createdAt: Date.now(),
      });
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (e: any) {
      setError(e.message || "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  /* ----------------------------- Recording Live ----------------------------- */
  function pickRecorderMime(): string | undefined {
    const candidates = [
      "audio/wav",
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/mp4",
      "audio/mpeg",
    ];
    for (const c of candidates) {
      if (MediaRecorder.isTypeSupported(c)) return c;
    }
    return undefined;
  }

  async function startLiveRecording() {
    setError(null);
    setStreamingMode("live");
    setStreamEvents([]);
    liveUploaderRef.current = createLiveChunkUploader({
      topK,
      onResult: (evt) => {
        setStreamEvents((prev) => [...prev, evt]);
        updateStreamMessage(evt);
      },
      onError: (err) => {
        addToast({
          type: "error",
          title: "Live Upload Failed",
          message: err.message,
        });
      },
    });
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 1,
          sampleRate: 16000
        },
      });

      // Force WAV format for better backend compatibility
      const recorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported('audio/wav') ? 'audio/wav' : 'audio/webm;codecs=opus'
      });

      recordedChunksRef.current = [];
      mediaRecorderRef.current = recorder;
      recorder.ondataavailable = async (ev) => {
        if (ev.data && ev.data.size > 5000 && liveUploaderRef.current) { // Min 5KB chunks
          try {
            // Convert to proper WAV if needed
            let audioBlob = ev.data;
            if (!ev.data.type.includes('wav')) {
              audioBlob = await convertToWAV(ev.data);
            }
            // Only send if blob is valid size
            if (audioBlob.size > 1000) {
              await liveUploaderRef.current.push(audioBlob);
            }
          } catch (conversionError) {
            console.warn('Failed to convert chunk:', conversionError);
          }
        }
      };
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        updateStreamMessage({
          chunkIndex: liveUploaderRef.current?.index || 0,
          timestamp: Date.now(),
          intent: "",
          confidence: 0,
          done: true,
        });
        liveUploaderRef.current?.abort();
      };
      recorder.start(2000); // 2 second chunks for better quality
      setIsRecording(true);
      recordingStartRef.current = performance.now();
      recordingTimerRef.current = window.setInterval(() => {
        if (recordingStartRef.current) {
          setRecordingElapsed(performance.now() - recordingStartRef.current);
        }
      }, 250);
      addToast({
        type: "info",
        title: "Live Recording Started",
        message: "Speaking into microphone...",
      });
    } catch {
      addToast({
        type: "error",
        title: "Microphone Access Failed",
        message: "Please check permissions and try again.",
      });
      setStreamingMode(null);
    }
  }

  // Helper function to convert audio to WAV
  async function convertToWAV(audioBlob: Blob): Promise<Blob> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = async (e) => {
        try {
          const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
          const arrayBuffer = e.target?.result as ArrayBuffer;
          const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

          // Convert to WAV format
          const wavBlob = audioBufferToWav(audioBuffer);
          resolve(wavBlob);
        } catch (error) {
          reject(error);
        }
      };
      reader.onerror = reject;
      reader.readAsArrayBuffer(audioBlob);
    });
  }

  // Convert AudioBuffer to WAV Blob
  function audioBufferToWav(buffer: AudioBuffer): Blob {
    const length = buffer.length;
    const sampleRate = buffer.sampleRate;
    const arrayBuffer = new ArrayBuffer(44 + length * 2);
    const view = new DataView(arrayBuffer);

    // WAV header
    const writeString = (offset: number, string: string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, length * 2, true);

    // Convert audio data
    const channelData = buffer.getChannelData(0);
    let offset = 44;
    for (let i = 0; i < length; i++) {
      const sample = Math.max(-1, Math.min(1, channelData[i]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
      offset += 2;
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
  }

  function stopLiveRecording() {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
    setIsRecording(false);
    setStreamingMode(null);
    addToast({
      type: "success",
      title: "Live Recording Stopped",
    });
  }

  /* ----------------------- Record then Chunk Pseudo-Stream ------------------ */
  async function startRecordThenChunk() {
    setStreamingMode("chunk-file");
    setStreamEvents([]);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 1,
          sampleRate: 16000
        },
      });

      // Force WAV format
      const recorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported('audio/wav') ? 'audio/wav' : 'audio/webm;codecs=opus'
      });

      recordedChunksRef.current = [];
      mediaRecorderRef.current = recorder;
      recorder.ondataavailable = (ev) => {
        if (ev.data && ev.data.size > 0) recordedChunksRef.current.push(ev.data);
      };
      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        setIsRecording(false);

        try {
          let fullBlob = new Blob(recordedChunksRef.current, { type: recorder.mimeType });

          // Convert to WAV if necessary
          if (!fullBlob.type.includes('wav')) {
            fullBlob = await convertToWAV(fullBlob);
          }

          // Validate blob before processing
          if (fullBlob.size < 1000) {
            addToast({
              type: "warning",
              title: "Recording Too Short",
              message: "Please record for at least 1 second",
            });
            return;
          }

          // Send as single file instead of chunks to avoid corruption
          const result = await testIntent(fullBlob, topK, { timeoutMs: 60_000 });
          pushMessage({
            id: `intent-${Date.now()}`,
            role: "assistant",
            type: "intent-result",
            content: `Primary intent: ${result.intent}`,
            raw: result,
            createdAt: Date.now(),
          });

          addToast({
            type: "success",
            title: "Recording Processed",
            message: `Intent: ${result.intent}`,
          });

        } catch (error: any) {
          addToast({
            type: "error",
            title: "Processing Failed",
            message: error.message || "Failed to process recording",
          });
        }
      };
      recorder.start();
      setIsRecording(true);
      recordingStartRef.current = performance.now();
      recordingTimerRef.current = window.setInterval(() => {
        if (recordingStartRef.current) {
          setRecordingElapsed(performance.now() - recordingStartRef.current);
        }
      }, 250);
      addToast({
        type: "info",
        title: "Recording Started",
        message: "Recording audio for analysis...",
      });
    } catch {
      addToast({
        type: "error",
        title: "Microphone Access Failed",
        message: "Please check permissions and try again.",
      });
      setStreamingMode(null);
    }
  }

  function stopChunkRecording() {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
    setIsRecording(false);
  }

  /* ------------------------- Text Prompt (Placeholder) ---------------------- */
  function handleSendPrompt() {
    if (!input.trim()) return;
    pushMessage({
      id: `user-text-${Date.now()}`,
      role: "user",
      type: "prompt",
      content: input.trim(),
      createdAt: Date.now(),
    });
    pushMessage({
      id: `assistant-ack-${Date.now()}`,
      role: "assistant",
      type: "intent-result",
      content: "Text prompt received (no semantic mapping yet).",
      raw: {
        intent: "N/A",
        confidence: 0,
        top_predictions: [],
        filename: "N/A",
      },
      createdAt: Date.now(),
    });
    setInput("");
  }

  function handleTextareaKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendPrompt();
    }
  }

  /* ------------------------- Derived Streaming Summary ---------------------- */
  const streamingSummary =
    streamEvents.length > 0
      ? summarizeStreamingEvents(streamEvents.filter((e) => !e.done && !e.error))
      : null;

  /* --------------------------------- Render --------------------------------- */
  return (
    <div className="flex flex-col h-dvh bg-[var(--color-bg)] text-[var(--color-fg)] font-sans selection:bg-[var(--color-accent)/35]">
      <Header
        health={health}
        modelInfo={modelInfo}
        topK={topK}
        onTopKChange={(k) => setTopK(k)}
      />

      <div className="flex flex-1 overflow-hidden">
        <ChatPane messages={messages} />

        <SidePanel
          streamingMode={streamingMode}
          isRecording={isRecording}
          recordingElapsed={recordingElapsed}
          streamingSummary={streamingSummary}
        />
      </div>

      <InputBar
        input={input}
        onInputChange={setInput}
        onSend={handleSendPrompt}
        onKeyDown={handleTextareaKey}
        selectedFile={selectedFile}
        onFileChange={handleFileSelect}
        fileInputRef={fileInputRef}
        busy={busy}
        onAnalyze={submitAudioFile}
        isRecording={isRecording}
        streamingMode={streamingMode}
        onStartLive={startLiveRecording}
        onStopLive={stopLiveRecording}
        onStartChunk={startRecordThenChunk}
        onStopChunk={stopChunkRecording}
        error={error}
        onDismissError={() => setError(null)}
        topK={topK}
        streamingSummary={streamingSummary}
      />

      <Footer />
      <ToastContainer toasts={toasts} onRemove={removeToast} />
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Header Component                                                           */
/* -------------------------------------------------------------------------- */
function Header({
  health,
  modelInfo,
  topK,
  onTopKChange,
}: {
  health: string;
  modelInfo: any;
  topK: number;
  onTopKChange: (k: number) => void;
}) {
  return (
    <header className="flex items-center justify-between px-6 py-3 border-b border-[var(--color-border)] bg-[var(--color-surface)]/80 backdrop-blur supports-[backdrop-filter]:bg-[var(--color-surface)]/60">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 text-sm font-semibold tracking-wide">
          <Brain className="h-5 w-5 text-[var(--color-accent)]" />
          <span>Akan Intent Studio</span>
        </div>
        <StatusPill status={health} />
        {modelInfo && (
          <div className="hidden md:flex items-center gap-2 text-xs px-2 py-1 rounded-md border border-[var(--color-border)] bg-[var(--color-surface-alt)]">
            <Activity className="h-3.5 w-3.5 opacity-70" />
            <span className="font-medium">
              {modelInfo.model_type} · {modelInfo.num_classes} intents
            </span>
          </div>
        )}
      </div>
      <div className="flex items-center gap-3">
        <select
          value={topK}
          onChange={(e) => onTopKChange(parseInt(e.target.value))}
          className="text-xs rounded-md border border-[var(--color-border)] bg-[var(--color-surface-alt)] px-2 py-1 focus:outline-none focus:ring-2 focus:ring-[var(--color-accent)]/40"
        >
          {[3, 5, 10, 15].map((k) => (
            <option key={k} value={k}>
              Top {k}
            </option>
          ))}
        </select>
        <button
          onClick={() => window.location.reload()}
          className="inline-flex items-center gap-1 text-xs font-medium rounded-md border border-[var(--color-border)] bg-[var(--color-surface-alt)] px-3 py-1.5 hover:bg-[var(--color-surface-hover)] transition focus:outline-none focus:ring-2 focus:ring-[var(--color-accent)]/40"
        >
          <RefreshCcw className="h-3.5 w-3.5" />
          Refresh
        </button>
      </div>
    </header>
  );
}

/* -------------------------------------------------------------------------- */
/* Status Pill                                                                */
/* -------------------------------------------------------------------------- */
function StatusPill({ status }: { status: string }) {
  const map: Record<string, string> = {
    ok: "bg-[var(--color-ok)]",
    connected: "bg-[var(--color-ok)]",
    checking: "bg-[var(--color-warn)] animate-pulse",
    unreachable: "bg-[var(--color-danger)]",
  };
  const cls = map[status] || "bg-neutral-400";
  return (
    <span className="inline-flex items-center gap-1 text-[11px] font-medium px-2 py-1 rounded-full bg-[var(--color-surface-alt)] border border-[var(--color-border)]">
      <span className={`w-2.5 h-2.5 rounded-full ${cls}`} />
      <span className="uppercase tracking-wider">{status}</span>
    </span>
  );
}

/* -------------------------------------------------------------------------- */
/* Chat Pane                                                                  */
/* -------------------------------------------------------------------------- */
function ChatPane({ messages }: { messages: ChatMessage[] }) {
  const listRef = useRef<HTMLDivElement | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  // Always scroll newest message into view (anchored at bottom like ChatGPT)
  useEffect(() => {
    if (bottomRef.current) {
      requestAnimationFrame(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
      });
    }
  }, [messages]);

  return (
    <section className="flex flex-col flex-1 overflow-hidden">
      <div
        ref={listRef}
        className="flex-1 overflow-y-auto px-6 py-6 space-y-5 scrollbar-thin scrollbar-thumb-[var(--color-border)]/60 scrollbar-track-transparent flex flex-col justify-end"
      >
        {messages.length === 0 && (
          <div className="mt-auto mb-24 text-center text-sm text-[var(--muted-foreground)]">
            Upload or record audio to begin intent recognition.
          </div>
        )}
        {messages.map((m) => (
          <ChatBubble key={m.id} message={m} />
        ))}
        <div ref={bottomRef} />
      </div>
    </section>
  );
}

/* -------------------------------------------------------------------------- */
/* Side Panel                                                                 */
/* -------------------------------------------------------------------------- */
function SidePanel({
  streamingMode,
  isRecording,
  recordingElapsed,
  streamingSummary,
}: {
  streamingMode: string | null;
  isRecording: boolean;
  recordingElapsed: number;
  streamingSummary: ReturnType<typeof summarizeStreamingEvents> | null;
}) {
  return (
    <aside className="hidden xl:flex flex-col w-80 border-l border-[var(--color-border)] bg-[var(--color-surface)]/70 backdrop-blur px-5 py-6 gap-8 overflow-y-auto">
      <Panel title="Session">
        <ul className="space-y-2 text-xs text-[var(--muted-foreground)]">
          <li>
            Mode:{" "}
            <span className="text-[var(--color-fg)] font-medium">
              {streamingMode || "idle"}
            </span>
          </li>
          <li>
            Recording:{" "}
            <span className="text-[var(--color-fg)] font-medium">
              {isRecording ? "yes" : "no"}
            </span>
          </li>
          {isRecording && (
            <li>
              Elapsed:{" "}
              <span className="text-[var(--color-fg)] font-medium">
                {(recordingElapsed / 1000).toFixed(1)}s
              </span>
            </li>
          )}
        </ul>
      </Panel>

      <Panel title="Confidence Legend">
        <div className="space-y-2 text-xs">
          <LegendRow color="bg-[var(--color-ok)]" label="High (≥ 80%)" />
          <LegendRow color="bg-[var(--color-warn)]" label="Medium (50–79%)" />
          <LegendRow color="bg-[var(--color-danger)]" label="Low (&lt; 50%)" />
        </div>
      </Panel>

      <Panel title="Streaming Summary">
        {streamingSummary ? (
          <ul className="text-xs space-y-2 text-[var(--muted-foreground)]">
            <li>
              Chunks:{" "}
              <span className="text-[var(--color-fg)] font-medium">
                {streamingSummary.totalChunks}
              </span>
            </li>
            <li>
              Frequent:{" "}
              <span className="text-[var(--color-fg)] font-medium">
                {streamingSummary.mostFrequentIntent || "—"}
              </span>
            </li>
            <li>
              Avg Conf:{" "}
              <span className="text-[var(--color-fg)] font-medium">
                {(streamingSummary.averageConfidence * 100).toFixed(1)}%
              </span>
            </li>
          </ul>
        ) : (
          <div className="text-[11px] text-[var(--muted-foreground)]">
            No live data yet.
          </div>
        )}
      </Panel>

      <Panel title="Tips">
        <ul className="text-xs space-y-2 text-[var(--muted-foreground)] list-disc list-inside">
          <li>Use clear, isolated utterances.</li>
          <li>Lower background noise improves accuracy.</li>
          <li>Prefer WAV for fidelity.</li>
        </ul>
      </Panel>
    </aside>
  );
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-3">
      <h2 className="text-[11px] font-semibold uppercase tracking-wider text-[var(--muted-foreground)] flex items-center gap-2">
        <Sparkles className="h-3.5 w-3.5 opacity-70" />
        {title}
      </h2>
      <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-alt)]/60 p-4 shadow-sm">
        {children}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Input Bar                                                                  */
/* -------------------------------------------------------------------------- */
function InputBar({
  input,
  onInputChange,
  onSend,
  onKeyDown,
  selectedFile,
  onFileChange,
  fileInputRef,
  busy,
  onAnalyze,
  isRecording,
  streamingMode,
  onStartLive,
  onStopLive,
  onStartChunk,
  onStopChunk,
  error,
  onDismissError,
  streamingSummary,
}: {
  input: string;
  onInputChange: (v: string) => void;
  onSend: () => void;
  onKeyDown: (e: KeyboardEvent<HTMLTextAreaElement>) => void;
  selectedFile: File | null;
  onFileChange: (e: ChangeEvent<HTMLInputElement>) => void;
  fileInputRef: React.RefObject<HTMLInputElement>;
  busy: boolean;
  onAnalyze: () => void;
  isRecording: boolean;
  streamingMode: string | null;
  onStartLive: () => void;
  onStopLive: () => void;
  onStartChunk: () => void;
  onStopChunk: () => void;
  error: string | null;
  onDismissError: () => void;
  streamingSummary: ReturnType<typeof summarizeStreamingEvents> | null;
}) {
  return (
    <div className="border-t border-[var(--color-border)] bg-[var(--color-surface)]/85 backdrop-blur px-5 py-4">
      <div className="max-w-5xl mx-auto flex flex-col gap-3">
        {error && (
          <div className="text-xs bg-[var(--color-danger)]/15 border border-[var(--color-danger)]/40 text-[var(--color-danger)] px-3 py-2 rounded flex justify-between items-center">
            <span>{error}</span>
            <button
              onClick={onDismissError}
              className="rounded p-1 hover:bg-[var(--color-danger)]/20"
            >
              ×
            </button>
          </div>
        )}
        <div className="flex gap-3 items-end flex-wrap">
          <div className="flex-1 flex flex-col gap-2">
            <textarea
              rows={1}
              placeholder="Add an optional note or instruction..."
              value={input}
              onChange={(e) => onInputChange(e.target.value)}
              onKeyDown={onKeyDown}
              className="resize-none w-full rounded-md bg-[var(--color-surface-alt)] border border-[var(--color-border)] px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-[var(--color-accent)]/40 placeholder:text-[var(--muted-foreground)]/70"
            />
            <div className="flex flex-wrap gap-2 items-center">
              {/* File Upload */}
              <div className="relative">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".wav,.mp3,audio/wav,audio/mpeg"
                  onChange={onFileChange}
                  className="absolute inset-0 opacity-0 cursor-pointer"
                />
                <button
                  type="button"
                  className="inline-flex items-center gap-1 text-xs font-medium px-3 py-1.5 rounded-md bg-[var(--color-surface-alt)] hover:bg-[var(--color-surface-hover)] border border-[var(--color-border)] transition"
                >
                  <Upload className="h-3.5 w-3.5" />
                  {selectedFile ? "Change File" : "Upload Audio"}
                </button>
              </div>
              {selectedFile && (
                <div className="text-[11px] px-2 py-1 rounded bg-[var(--color-ok)]/15 border border-[var(--color-ok)]/40 inline-flex items-center gap-1">
                  <FileAudio2 className="h-3.5 w-3.5" />
                  {selectedFile.name}
                </div>
              )}
              {/* Live Streaming */}
              {!isRecording && (
                <button
                  type="button"
                  onClick={onStartLive}
                  className="inline-flex items-center gap-1 text-xs font-medium px-3 py-1.5 rounded-md bg-[var(--color-accent)] text-white hover:brightness-110 shadow-sm transition"
                >
                  <Radio className="h-3.5 w-3.5" />
                  Live Stream
                </button>
              )}
              {isRecording && streamingMode === "live" && (
                <button
                  onClick={onStopLive}
                  className="inline-flex items-center gap-1 text-xs font-medium px-3 py-1.5 rounded-md bg-[var(--color-danger)] text-white hover:brightness-110 shadow-sm transition"
                >
                  <Square className="h-3.5 w-3.5" />
                  Stop Live
                </button>
              )}
              {/* Record then chunk */}
              {!isRecording && (
                <button
                  onClick={onStartChunk}
                  className="inline-flex items-center gap-1 text-xs font-medium px-3 py-1.5 rounded-md bg-[var(--color-info)] text-white hover:brightness-110 shadow-sm transition"
                >
                  <Headphones className="h-3.5 w-3.5" />
                  Record & Chunk
                </button>
              )}
              {isRecording && streamingMode === "chunk-file" && (
                <button
                  onClick={onStopChunk}
                  className="inline-flex items-center gap-1 text-xs font-medium px-3 py-1.5 rounded-md bg-[var(--color-danger)] text-white hover:brightness-110 shadow-sm transition"
                >
                  <Square className="h-3.5 w-3.5" />
                  Finish Recording
                </button>
              )}
              {/* Analyze file */}
              {selectedFile && (
                <button
                  onClick={onAnalyze}
                  disabled={busy}
                  className="inline-flex items-center gap-1 text-xs font-medium px-3 py-1.5 rounded-md bg-[var(--color-ok)] text-white hover:brightness-110 shadow-sm transition disabled:opacity-50"
                  type="button"
                >
                  <Mic className="h-3.5 w-3.5" />
                  {busy ? "Analyzing..." : "Analyze"}
                </button>
              )}
              <button
                onClick={onSend}
                disabled={!input.trim()}
                className="inline-flex items-center gap-1 text-xs font-medium px-3 py-1.5 rounded-md bg-[var(--color-surface-alt)] hover:bg-[var(--color-surface-hover)] border border-[var(--color-border)] transition disabled:opacity-40"
                type="button"
              >
                <Send className="h-3.5 w-3.5" />
                Send
              </button>
            </div>
            {streamingSummary && (
              <div className="flex gap-4 flex-wrap text-[11px] text-[var(--muted-foreground)]">
                <span>
                  Chunks:{" "}
                  <strong className="text-[var(--color-fg)]">
                    {streamingSummary.totalChunks}
                  </strong>
                </span>
                <span>
                  Frequent:{" "}
                  <strong className="text-[var(--color-fg)]">
                    {streamingSummary.mostFrequentIntent || "—"}
                  </strong>
                </span>
                <span>
                  Avg Conf:{" "}
                  <strong className="text-[var(--color-fg)]">
                    {(streamingSummary.averageConfidence * 100).toFixed(1)}%
                  </strong>
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Chat Bubble                                                                */
/* -------------------------------------------------------------------------- */
function ChatBubble({ message }: { message: ChatMessage }) {
  // Filter out noisy messages that are now handled by toasts
  if (
    message.type === "prompt" &&
    (/^Recording for later chunked streaming/i.test(message.content) ||
      /^Started live recognition/i.test(message.content))
  ) {
    return null;
  }

  // Filter out streaming completion and low-confidence chunks
  if (
    message.type === "stream-chunk" &&
    (message.done || message.error || message.confidence < 0.3)
  ) {
    return null;
  }
  const isUser = message.role === "user";
  const baseUser = "max-w-[80%] md:max-w-[65%] px-4 py-3 rounded-xl text-sm relative shadow-sm";
  const baseAssistant = "w-full px-4 py-3 rounded-xl text-sm relative shadow-sm";
  const base = isUser ? baseUser : baseAssistant;
  const palette = isUser
    ? "bg-[var(--color-accent)] text-white border border-[var(--color-accent)]/60"
    : message.type === "stream-chunk"
      ? "bg-[var(--color-surface-alt)] border border-[var(--color-border)]"
      : "bg-[var(--color-surface-alt)] border border-[var(--color-border)]";
  const alignment = "self-center";

  return (
    <div className={`flex flex-col ${alignment} gap-1`}>
      <div className={`${base} ${palette}`}>
        {message.type === "intent-result" && "raw" in message ? (
          <IntentResultView result={message.raw as IntentResult} />
        ) : message.type === "stream-chunk" ? (
          <StreamChunkView msg={message} />
        ) : (
          <span className="whitespace-pre-wrap leading-relaxed font-medium">
            {message.content}
          </span>
        )}
      </div>
      <div className="text-[10px] uppercase tracking-wider text-[var(--muted-foreground)] px-1">
        {new Date(message.createdAt).toLocaleTimeString()}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Intent Result View                                                         */
/* -------------------------------------------------------------------------- */
function IntentResultView({ result }: { result: IntentResult }) {
  return (
    <div className="space-y-4">
      {/* Primary intent spotlight */}
      <div className="rounded-xl border border-[var(--color-accent)]/50 bg-[var(--color-accent)]/10 p-4 flex flex-col gap-3 shadow-sm">
        <div className="flex items-start justify-between gap-4">
          <div className="flex flex-col gap-1">
            <span className="text-xl md:text-2xl font-extrabold tracking-wide text-[var(--color-accent)] leading-tight">
              {result.intent}
            </span>
            <span className="text-[11px] uppercase tracking-wider text-[var(--muted-foreground)] font-medium">
              Primary Intent
            </span>
          </div>
          <span className="text-sm font-semibold px-3 py-1 rounded-md bg-[var(--color-accent)]/20 text-[var(--color-accent)]">
            {formatConfidence(result.confidence, 1)}
          </span>
        </div>
        <div className="h-2 rounded-full bg-[var(--color-accent)]/15 overflow-hidden">
          <div
            className="h-full bg-[var(--color-accent)] transition-all duration-500"
            style={{ width: `${(result.confidence * 100).toFixed(1)}%` }}
          />
        </div>
      </div>
      {result.top_predictions.length > 0 && (
        <div className="space-y-1">
          <p className="text-[11px] uppercase tracking-wider text-[var(--muted-foreground)] font-semibold">
            Top {result.top_predictions.length} Predictions
          </p>
          <ul className="space-y-1">
            {result.top_predictions.map((p, i) => {
              const tier = classifyConfidenceTier(p.confidence);
              const color =
                tier === "high"
                  ? "bg-[var(--color-ok)]/20 border-[var(--color-ok)]/50"
                  : tier === "medium"
                    ? "bg-[var(--color-warn)]/20 border-[var(--color-warn)]/50"
                    : "bg-[var(--color-danger)]/20 border-[var(--color-danger)]/50";
              return (
                <li
                  key={i}
                  className={`text-xs flex items-center justify-between px-2 py-1 rounded border ${color}`}
                >
                  <span className="font-medium">{p.intent}</span>
                  <span>{formatConfidence(p.confidence, 1)}</span>
                </li>
              );
            })}
          </ul>
        </div>
      )}
      <div className="text-[10px] text-[var(--muted-foreground)] flex flex-wrap gap-2">
        <span>File: {result.filename}</span>
        {typeof result.processing_time_ms === "number" && (
          <span>{result.processing_time_ms.toFixed(2)} ms</span>
        )}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Stream Chunk View                                                          */
/* -------------------------------------------------------------------------- */
function StreamChunkView({
  msg,
}: {
  msg: Extract<ChatMessage, { type: "stream-chunk" }>;
}) {
  const tier = classifyConfidenceTier(msg.confidence);
  const badgeClasses =
    tier === "high"
      ? "bg-[var(--color-ok)]/25 border-[var(--color-ok)]/50"
      : tier === "medium"
        ? "bg-[var(--color-warn)]/25 border-[var(--color-warn)]/50"
        : "bg-[var(--color-danger)]/25 border-[var(--color-danger)]/50";

  if (msg.done) {
    return (
      <div className="text-xs text-[var(--muted-foreground)] italic">
        Streaming session completed.
      </div>
    );
  }
  if (msg.error) {
    return (
      <div className="text-xs text-[var(--color-danger)] font-medium">
        Chunk {msg.chunkIndex}: {msg.error}
      </div>
    );
  }
  if (!msg.content) {
    return (
      <div className="text-xs text-[var(--muted-foreground)] animate-pulse">
        Awaiting recognition...
      </div>
    );
  }
  return (
    <div className="flex items-center gap-2 text-xs">
      <span
        className={`px-2 py-1 rounded border font-medium ${badgeClasses} whitespace-nowrap`}
      >
        {formatConfidence(msg.confidence, 1)}
      </span>
      <span className="truncate font-medium">{msg.content}</span>
      <span className="text-[var(--muted-foreground)]/70">#{msg.chunkIndex}</span>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Legend Row                                                                 */
/* -------------------------------------------------------------------------- */
function LegendRow({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className={`w-3 h-3 rounded ${color}`} />
      <span>{label}</span>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Footer                                                                     */
/* -------------------------------------------------------------------------- */
function Footer() {
  return (
    <footer className="px-6 py-3 text-[11px] text-[var(--muted-foreground)] border-t border-[var(--color-border)] bg-[var(--color-surface)]/70 backdrop-blur flex items-center justify-between">
      <span>
        Akan Speech Intent · {new Date().getFullYear()}
      </span>
      <span className="hidden sm:inline">
        API: {API_BASE_URL.replace(/^https?:\/\//, "")}
      </span>
    </footer>
  );
}

/* -------------------------------------------------------------------------- */
/* Toast System                                                               */
/* -------------------------------------------------------------------------- */
function ToastContainer({ toasts, onRemove }: { toasts: Toast[]; onRemove: (id: string) => void }) {
  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 max-w-sm">
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onRemove={onRemove} />
      ))}
    </div>
  );
}

function ToastItem({ toast, onRemove }: { toast: Toast; onRemove: (id: string) => void }) {
  const iconMap = {
    success: CheckCircle,
    error: XCircle,
    warning: AlertCircle,
    info: AlertCircle,
  };

  const colorMap = {
    success: "bg-[var(--color-ok)]/15 border-[var(--color-ok)]/40 text-[var(--color-ok)]",
    error: "bg-[var(--color-danger)]/15 border-[var(--color-danger)]/40 text-[var(--color-danger)]",
    warning: "bg-[var(--color-warn)]/15 border-[var(--color-warn)]/40 text-[var(--color-warn)]",
    info: "bg-[var(--color-info)]/15 border-[var(--color-info)]/40 text-[var(--color-info)]",
  };

  const Icon = iconMap[toast.type];

  return (
    <div className={`p-4 rounded-lg border backdrop-blur animate-in slide-in-from-right ${colorMap[toast.type]}`}>
      <div className="flex items-start gap-3">
        <Icon className="h-5 w-5 flex-shrink-0 mt-0.5" />
        <div className="flex-1">
          <div className="font-semibold text-sm">{toast.title}</div>
          {toast.message && (
            <div className="text-xs mt-1 opacity-90">{toast.message}</div>
          )}
        </div>
        <button
          onClick={() => onRemove(toast.id)}
          className="flex-shrink-0 rounded-md p-1 hover:bg-black/10 transition"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}
