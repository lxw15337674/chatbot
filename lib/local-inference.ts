import { getTextFromMessage } from "@/lib/utils";
import {
  getLocalModelDownloadSummary,
  saveLocalModelCacheMeta,
  upsertLocalModelDownloadRecord,
} from "./local-chat-store";
import type { ChatMessage } from "./types";

type LocalDtype = "q4f16" | "q4" | "q8" | "fp16" | "fp32";

type LocalLoadMeta = {
  bytesTotal: number;
  fileCount: number;
  fromCacheCount: number;
};

export type LocalReasoningMode = "normal" | "thinking";

export type LocalModelLoadPhase =
  | "downloading"
  | "initializing"
  | "ready"
  | "failed"
  | "cancelled";

export type LocalModelLoadProgressEvent = {
  phase: LocalModelLoadPhase;
  message: string;
  progress: number | null;
  loaded: number;
  total: number;
};

type HubProgressInfo = {
  status:
    | "initiate"
    | "download"
    | "progress"
    | "progress_total"
    | "done"
    | "ready";
  file?: string;
  progress?: number;
  loaded?: number;
  total?: number;
};

type RuntimeLoadOptions = {
  onLoadProgress?: (event: LocalModelLoadProgressEvent) => void;
  isLoadCancelled?: () => boolean;
};

type ModelRuntime = {
  generator: any;
  device: "webgpu" | "wasm";
  dtype: LocalDtype;
  loadMeta: LocalLoadMeta;
};

const runtimes = new Map<string, Promise<ModelRuntime>>();
const loadedModelIds = new Set<string>();
const MODEL_CACHE_KEY = "chatbot-transformers-cache-v1";
const CHINESE_PRIORITY_MODEL_IDS = new Set([
  "onnx-community/Qwen2.5-0.5B-Instruct",
  "onnx-community/Qwen3.5-0.8B-ONNX",
  "onnx-community/Qwen3.5-2B-ONNX",
]);
let transformersEnvInitialized = false;

function emitLoadProgress(
  options: RuntimeLoadOptions | undefined,
  event: LocalModelLoadProgressEvent
) {
  if (options?.isLoadCancelled?.()) {
    return;
  }

  options?.onLoadProgress?.(event);
}

function toLoadProgressEvent(info: HubProgressInfo): LocalModelLoadProgressEvent {
  switch (info.status) {
    case "initiate":
    case "download":
      return {
        phase: "downloading",
        message: "正在下载模型文件...",
        progress: null,
        loaded: 0,
        total: 0,
      };
    case "progress":
    case "progress_total": {
      const loaded = typeof info.loaded === "number" ? info.loaded : 0;
      const total = typeof info.total === "number" ? info.total : 0;
      const progressValue =
        typeof info.progress === "number"
          ? Math.max(0, Math.min(100, info.progress))
          : total > 0
            ? Math.max(0, Math.min(100, (loaded / total) * 100))
          : null;

      return {
        phase: "downloading",
        message:
          progressValue === null
            ? "正在下载模型文件..."
            : `正在下载模型文件 (${Math.round(progressValue)}%)${info.file ? ` · ${info.file}` : ""}`,
        progress: progressValue,
        loaded,
        total,
      };
    }
    case "done":
      return {
        phase: "initializing",
        message: "正在初始化推理引擎...",
        progress: null,
        loaded: typeof info.loaded === "number" ? info.loaded : 0,
        total: typeof info.total === "number" ? info.total : 0,
      };
    case "ready":
      return {
        phase: "ready",
        message: "模型已就绪",
        progress: 100,
        loaded: 0,
        total: 0,
      };
    default:
      return {
        phase: "initializing",
        message: "正在初始化推理引擎...",
        progress: null,
        loaded: 0,
        total: 0,
      };
  }
}

function hasWebGpuSupport() {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}

function initializeTransformersEnv(env: {
  useBrowserCache?: boolean;
  useWasmCache?: boolean;
  useCustomCache?: boolean;
  cacheKey?: string;
}) {
  if (transformersEnvInitialized) {
    return;
  }

  env.useBrowserCache = true;
  env.useWasmCache = true;
  env.useCustomCache = false;
  env.cacheKey = MODEL_CACHE_KEY;

  transformersEnvInitialized = true;
}

async function resolveDtype(modelId: string): Promise<LocalDtype> {
  try {
    const { ModelRegistry } = await import("@huggingface/transformers");
    const available = await ModelRegistry.get_available_dtypes(modelId);
    const preferred: LocalDtype[] = ["q4f16", "q4", "q8", "fp16", "fp32"];

    return preferred.find((dtype) => available.includes(dtype)) ?? "q4";
  } catch {
    return "q4";
  }
}

async function getRuntime(
  modelId: string,
  options?: RuntimeLoadOptions
): Promise<ModelRuntime> {
  const preferredDevice = hasWebGpuSupport() ? "webgpu" : "wasm";
  const dtype = await resolveDtype(modelId);
  const modelKey = `${modelId}:${dtype}`;
  const key = `${modelId}:${preferredDevice}:${dtype}`;

  emitLoadProgress(options, {
    phase: "initializing",
    message: "正在准备模型加载...",
    progress: null,
    loaded: 0,
    total: 0,
  });

  const existing = runtimes.get(key);
  if (existing) {
    const runtime = await existing;
    if (options?.isLoadCancelled?.()) {
      throw new Error("Model load cancelled");
    }
    return runtime;
  }

  const loading = (async () => {
    const { env, pipeline } = await import("@huggingface/transformers");
    initializeTransformersEnv(env);

    const persistedAtByFile = new Map<string, number>();
    const persistDownloadRecord = (params: {
      file: string;
      status: "downloading" | "done" | "failed";
      loadedBytes: number;
      totalBytes: number;
      force?: boolean;
    }) => {
      const file = params.file || "unknown";
      const now = Date.now();
      const lastPersistedAt = persistedAtByFile.get(file) ?? 0;

      if (!params.force && now - lastPersistedAt < 800) {
        return;
      }

      persistedAtByFile.set(file, now);

      void upsertLocalModelDownloadRecord({
        modelKey,
        modelId,
        dtype,
        file,
        status: params.status,
        loadedBytes: params.loadedBytes,
        totalBytes: params.totalBytes,
        updatedAt: now,
      });
    };

    const downloadSummary = await getLocalModelDownloadSummary(modelKey);
    if (downloadSummary.hasPartial) {
      const resumedProgress =
        downloadSummary.totalFiles > 0
          ? (downloadSummary.completedFiles / downloadSummary.totalFiles) * 100
          : null;

      emitLoadProgress(options, {
        phase: "downloading",
        message: `检测到未完成下载，正在续传 (${downloadSummary.completedFiles}/${downloadSummary.totalFiles})...`,
        progress: resumedProgress,
        loaded: 0,
        total: 0,
      });
    }

    const progressCallback = (progressInfo: HubProgressInfo) => {
      emitLoadProgress(options, toLoadProgressEvent(progressInfo));

      const file = progressInfo.file;
      if (!file) {
        return;
      }

      if (progressInfo.status === "done") {
        persistDownloadRecord({
          file,
          status: "done",
          loadedBytes:
            typeof progressInfo.loaded === "number" ? progressInfo.loaded : 0,
          totalBytes:
            typeof progressInfo.total === "number" ? progressInfo.total : 0,
          force: true,
        });
        return;
      }

      if (
        progressInfo.status === "initiate" ||
        progressInfo.status === "download" ||
        progressInfo.status === "progress"
      ) {
        persistDownloadRecord({
          file,
          status: "downloading",
          loadedBytes:
            typeof progressInfo.loaded === "number" ? progressInfo.loaded : 0,
          totalBytes:
            typeof progressInfo.total === "number" ? progressInfo.total : 0,
        });
      }
    };

    try {
      const generator = await pipeline("text-generation", modelId, {
        device: preferredDevice,
        dtype,
        progress_callback: progressCallback,
      });

      const loadMeta = extractLoadMeta(generator);
      void saveLocalModelCacheMeta({
        modelId,
        dtype,
        device: preferredDevice,
        ...loadMeta,
      });

      return {
        generator,
        device: preferredDevice,
        dtype,
        loadMeta,
      } satisfies ModelRuntime;
    } catch {
      emitLoadProgress(options, {
        phase: "initializing",
        message: "WebGPU 不可用，正在回退到 WASM...",
        progress: null,
        loaded: 0,
        total: 0,
      });

      try {
        const fallbackGenerator = await pipeline("text-generation", modelId, {
          device: "wasm",
          dtype,
          progress_callback: progressCallback,
        });

        const loadMeta = extractLoadMeta(fallbackGenerator);
        void saveLocalModelCacheMeta({
          modelId,
          dtype,
          device: "wasm",
          ...loadMeta,
        });

        return {
          generator: fallbackGenerator,
          device: "wasm",
          dtype,
          loadMeta,
        } satisfies ModelRuntime;
      } catch (fallbackError) {
        persistDownloadRecord({
          file: "runtime",
          status: "failed",
          loadedBytes: 0,
          totalBytes: 0,
          force: true,
        });

        throw fallbackError;
      }
    }
  })();

  runtimes.set(key, loading);
  const runtime = await loading;
  loadedModelIds.add(modelId);

  if (options?.isLoadCancelled?.()) {
    throw new Error("Model load cancelled");
  }

  return runtime;
}

function normalizeGeneratedText(output: unknown): string {
  if (Array.isArray(output) && output.length > 0) {
    const first = output[0] as any;

    if (typeof first?.generated_text === "string") {
      return first.generated_text.trim();
    }

    if (Array.isArray(first?.generated_text)) {
      const last = first.generated_text.at(-1);
      if (last?.content && typeof last.content === "string") {
        return last.content.trim();
      }
    }
  }

  if (
    typeof output === "object" &&
    output !== null &&
    "generated_text" in output
  ) {
    const value = (output as { generated_text?: unknown }).generated_text;
    if (typeof value === "string") {
      return value.trim();
    }
  }

  return "";
}

function cleanGeneratedAssistantText(rawText: string, prompt?: string | null) {
  let text = rawText.replace(/\r\n/g, "\n").trim();

  if (prompt && text.startsWith(prompt)) {
    text = text.slice(prompt.length).trim();
  }

  // Some base models continue writing the whole transcript. Keep only assistant content.
  if (/^(System|User|Assistant|用户|助手)\s*[:：]/i.test(text)) {
    const segments = text.split(/\n\s*(?:Assistant|助手)\s*[:：]\s*/i);
    if (segments.length > 1) {
      text = segments.at(-1)?.trim() ?? text;
    }
  }

  const nextUserIndex = text.search(/\n\s*(?:User|用户)\s*[:：]/i);
  if (nextUserIndex > 0) {
    text = text.slice(0, nextUserIndex).trim();
  }

  text = text.replace(/^(?:Assistant|助手)\s*[:：]\s*/i, "").trim();

  return text;
}

function looksLikeGarbledText(text: string) {
  const normalized = text.trim();
  if (normalized.length < 18) {
    return false;
  }

  const cjkCount = (normalized.match(/[\u4e00-\u9fff]/g) ?? []).length;
  const alphaCount = (normalized.match(/[A-Za-z]/g) ?? []).length;
  const digitCount = (normalized.match(/[0-9]/g) ?? []).length;
  const symbolCount = (
    normalized.match(/[^\u4e00-\u9fffA-Za-z0-9\s.,!?，。！？:：;；'"()\-]/g) ?? []
  ).length;

  if (cjkCount === 0 && alphaCount <= 2 && symbolCount >= 6) {
    return true;
  }

  const totalCount = cjkCount + alphaCount + digitCount + symbolCount;
  if (totalCount === 0) {
    return false;
  }

  const symbolRatio = symbolCount / totalCount;
  return symbolRatio > 0.45 && cjkCount < 2;
}

function hasChineseText(text: string) {
  return /[\u4e00-\u9fff]/.test(text);
}

function supportsThinkingToggle(modelId: string) {
  return /onnx-community\/Qwen3(?:\.5)?-/i.test(modelId);
}

function stripThinkTags(text: string): string {
  return text.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
}

function extractReasoningAndAnswer(text: string): {
  reasoningText: string;
  answerText: string;
} {
  const reasoningParts: string[] = [];
  const answerText = text
    .replace(/<think>([\s\S]*?)<\/think>/gi, (_, block: string) => {
      const normalized = block.trim();
      if (normalized.length > 0) {
        reasoningParts.push(normalized);
      }
      return "";
    })
    .trim();

  return {
    reasoningText: reasoningParts.join("\n\n"),
    answerText,
  };
}

function createGenerationOptions(params: {
  modelId: string;
  reasoningMode: LocalReasoningMode;
  signal?: AbortSignal;
  deterministic?: boolean;
}) {
  const { modelId, reasoningMode, signal, deterministic = false } = params;
  const tokenizerEncodeKwargs = supportsThinkingToggle(modelId)
    ? { enable_thinking: reasoningMode === "thinking" }
    : undefined;

  if (deterministic) {
    return {
      max_new_tokens: 140,
      do_sample: false,
      return_full_text: false,
      tokenizer_encode_kwargs: tokenizerEncodeKwargs,
      signal,
    };
  }

  return {
    max_new_tokens: 180,
    do_sample: true,
    temperature: 0.7,
    top_p: 0.9,
    return_full_text: false,
    tokenizer_encode_kwargs: tokenizerEncodeKwargs,
    signal,
  };
}

function isChatTemplateMissingError(error: unknown): boolean {
  if (!(error instanceof Error)) {
    return false;
  }

  return /apply_chat_template|chat_template/i.test(error.message);
}

function buildPlainPrompt(
  messages: Array<{ role: "system" | "user" | "assistant"; content: string }>
): string {
  const labelByRole: Record<"system" | "user" | "assistant", string> = {
    system: "System",
    user: "User",
    assistant: "Assistant",
  };

  const lines = messages.map(
    (message) => `${labelByRole[message.role]}: ${message.content}`
  );

  if (lines.length === 0) {
    return "User: Hello\nAssistant:";
  }

  const lastRole = messages.at(-1)?.role;
  if (lastRole !== "assistant") {
    lines.push("Assistant:");
  }

  return lines.join("\n\n");
}

function extractLoadMeta(generator: any): LocalLoadMeta {
  const metadata = Array.isArray(generator?.metadata)
    ? generator.metadata
    : [];

  if (metadata.length === 0) {
    return {
      bytesTotal: 0,
      fileCount: 0,
      fromCacheCount: 0,
    };
  }

  let bytesTotal = 0;
  let fromCacheCount = 0;

  for (const row of metadata) {
    if (typeof row?.size === "number") {
      bytesTotal += row.size;
    }
    if (row?.fromCache === true) {
      fromCacheCount += 1;
    }
  }

  return {
    bytesTotal,
    fileCount: metadata.length,
    fromCacheCount,
  };
}

export async function generateLocalAssistantResponse(params: {
  modelId: string;
  messages: ChatMessage[];
  reasoningMode?: LocalReasoningMode;
  signal?: AbortSignal;
  onLoadProgress?: (event: LocalModelLoadProgressEvent) => void;
  isLoadCancelled?: () => boolean;
}): Promise<{
  text: string;
  reasoningText: string;
  dtype: LocalDtype;
  device: "webgpu" | "wasm";
  loadMeta: LocalLoadMeta;
}> {
  const {
    modelId,
    messages,
    reasoningMode = "normal",
    signal,
    onLoadProgress,
    isLoadCancelled,
  } = params;
  const runtime = await getRuntime(modelId, {
    onLoadProgress,
    isLoadCancelled,
  });

  const promptMessages = messages
    .filter(
      (message) =>
        message.role === "system" ||
        message.role === "user" ||
        message.role === "assistant"
    )
    .map((message) => ({
      role: message.role as "system" | "user" | "assistant",
      content: getTextFromMessage(message),
    }))
    .filter((message) => message.content.length > 0);

  const generationOptions = createGenerationOptions({
    modelId,
    reasoningMode,
    signal,
  });

  let output: unknown;
  let plainPromptUsed: string | null = null;
  let generationInput: typeof promptMessages | string = promptMessages;

  try {
    output = await runtime.generator(promptMessages, generationOptions);
  } catch (error) {
    if (!isChatTemplateMissingError(error)) {
      throw error;
    }

    const plainPrompt = buildPlainPrompt(promptMessages);
    plainPromptUsed = plainPrompt;
    generationInput = plainPrompt;
    output = await runtime.generator(plainPrompt, generationOptions);
  }

  if (signal?.aborted) {
    throw new Error("Generation aborted");
  }

  let rawText = cleanGeneratedAssistantText(
    normalizeGeneratedText(output),
    plainPromptUsed
  );

  let parsedReasoning = extractReasoningAndAnswer(rawText);
  let reasoningText =
    reasoningMode === "thinking" ? parsedReasoning.reasoningText : "";
  let text =
    parsedReasoning.answerText ||
    (reasoningMode === "normal" ? stripThinkTags(rawText) : rawText);

  if (looksLikeGarbledText(text)) {
    try {
      const retryOutput = await runtime.generator(
        generationInput,
        createGenerationOptions({
          modelId,
          reasoningMode,
          signal,
          deterministic: true,
        })
      );

      const retriedRawText = cleanGeneratedAssistantText(
        normalizeGeneratedText(retryOutput),
        plainPromptUsed
      );
      const retriedParsed = extractReasoningAndAnswer(retriedRawText);
      const retriedAnswer =
        retriedParsed.answerText ||
        (reasoningMode === "normal"
          ? stripThinkTags(retriedRawText)
          : retriedRawText);

      if (!looksLikeGarbledText(retriedAnswer) && retriedAnswer.length > 0) {
        text = retriedAnswer;
        if (reasoningMode === "thinking") {
          reasoningText = retriedParsed.reasoningText;
        }
      }
    } catch {
      // Keep original text and fall through to model suggestion handling.
    }
  }

  if (looksLikeGarbledText(text)) {
    const latestUserText = promptMessages
      .filter((message) => message.role === "user")
      .at(-1)?.content;

    if (
      latestUserText &&
      hasChineseText(latestUserText) &&
      !CHINESE_PRIORITY_MODEL_IDS.has(modelId)
    ) {
      text =
        "当前模型对中文支持不稳定，建议切换到 Qwen2.5 0.5B、Qwen3.5 0.8B 或 Qwen3.5 2B (INT4) 后再试。";
    }
  }

  return {
    text: text || "我目前还在本地模型冷启动中，请再试一次。",
    reasoningText,
    dtype: runtime.dtype,
    device: runtime.device,
    loadMeta: runtime.loadMeta,
  };
}

export async function prepareLocalModel(params: {
  modelId: string;
  onLoadProgress?: (event: LocalModelLoadProgressEvent) => void;
  isLoadCancelled?: () => boolean;
}): Promise<{
  dtype: LocalDtype;
  device: "webgpu" | "wasm";
  loadMeta: LocalLoadMeta;
}> {
  const runtime = await getRuntime(params.modelId, {
    onLoadProgress: params.onLoadProgress,
    isLoadCancelled: params.isLoadCancelled,
  });

  return {
    dtype: runtime.dtype,
    device: runtime.device,
    loadMeta: runtime.loadMeta,
  };
}

export function hasLoadedLocalModel(modelId: string): boolean {
  return loadedModelIds.has(modelId);
}
