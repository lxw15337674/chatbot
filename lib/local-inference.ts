import {
  getLocalModelDtypeCandidates,
  type LocalModelDtype,
} from "@/lib/ai/models";
import { getTextFromMessage } from "@/lib/utils";
import {
  clearLocalModelDownloadRecords,
  getLocalModelCacheMeta,
  saveLocalModelAssetState,
  saveLocalModelCacheMeta,
  upsertLocalModelDownloadRecord,
} from "./local-chat-store";
import type { ChatMessage } from "./types";

type LocalDtype = LocalModelDtype;

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

type CacheCheckResult = {
  allCached: boolean;
  files: Array<{
    file: string;
    cached: boolean;
  }>;
};

type VerifiedLocalModelAssetState = {
  modelId: string;
  status: "missing" | "partial" | "complete";
  dtype: LocalDtype | null;
  bytesTotal: number;
  fileCount: number;
  cachedFileCount: number;
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
const LOCAL_INFERENCE_DEBUG_PREFIX = "[local-inference]";

function logLocalInference(message: string, details?: Record<string, unknown>) {
  if (details) {
    console.info(LOCAL_INFERENCE_DEBUG_PREFIX, message, details);
    return;
  }

  console.info(LOCAL_INFERENCE_DEBUG_PREFIX, message);
}

function emitLoadProgress(
  options: RuntimeLoadOptions | undefined,
  event: LocalModelLoadProgressEvent
) {
  if (options?.isLoadCancelled?.()) {
    return;
  }

  options?.onLoadProgress?.(event);
}

function hasWebGpuSupport() {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}

function initializeTransformersEnv(env: {
  allowRemoteModels?: boolean;
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

function resolveDtypeCandidates(modelId: string): LocalDtype[] {
  return getLocalModelDtypeCandidates(modelId);
}

type TransformersInternals = {
  env: {
    allowRemoteModels?: boolean;
    useBrowserCache?: boolean;
    useWasmCache?: boolean;
    useCustomCache?: boolean;
    cacheKey?: string;
  };
  pipeline: typeof import("@huggingface/transformers").pipeline;
  getModelFile: (
    pathOrRepoId: string,
    filename: string,
    fatal?: boolean,
    options?: Record<string, unknown>,
    returnPath?: boolean
  ) => Promise<string | Uint8Array>;
  get_pipeline_files: (
    task: string,
    modelId: string,
    options?: Record<string, unknown>
  ) => Promise<string[]>;
  is_pipeline_cached_files: (
    task: string,
    modelId: string,
    options?: Record<string, unknown>
  ) => Promise<CacheCheckResult>;
  get_file_metadata: (
    pathOrRepoId: string,
    filename: string,
    options?: Record<string, unknown>
  ) => Promise<{
    exists: boolean;
    size?: number;
    fromCache?: boolean;
  }>;
};

let transformersInternalsPromise: Promise<TransformersInternals> | null = null;

async function getTransformersInternals(): Promise<TransformersInternals> {
  if (!transformersInternalsPromise) {
    transformersInternalsPromise = (async () => {
      const [
        transformersModule,
        hubModule,
        getPipelineFilesModule,
        isCachedModule,
        getFileMetadataModule,
      ] = await Promise.all([
        import("@huggingface/transformers"),
        // @ts-expect-error Internal Transformers.js source import with no bundled typings.
        import("../node_modules/@huggingface/transformers/src/utils/hub.js"),
        // @ts-expect-error Internal Transformers.js source import with no bundled typings.
        import("../node_modules/@huggingface/transformers/src/utils/model_registry/get_pipeline_files.js"),
        // @ts-expect-error Internal Transformers.js source import with no bundled typings.
        import("../node_modules/@huggingface/transformers/src/utils/model_registry/is_cached.js"),
        // @ts-expect-error Internal Transformers.js source import with no bundled typings.
        import("../node_modules/@huggingface/transformers/src/utils/model_registry/get_file_metadata.js"),
      ]);

      initializeTransformersEnv(transformersModule.env);

      return {
        env: transformersModule.env,
        pipeline: transformersModule.pipeline,
        getModelFile: hubModule.getModelFile,
        get_pipeline_files: getPipelineFilesModule.get_pipeline_files,
        is_pipeline_cached_files: isCachedModule.is_pipeline_cached_files,
        get_file_metadata: getFileMetadataModule.get_file_metadata,
      } satisfies TransformersInternals;
    })();
  }

  return transformersInternalsPromise;
}

function isMissingLocalAssetError(error: unknown): boolean {
  if (!(error instanceof Error)) {
    return false;
  }

  return /could not locate file|local files only|local_files_only/i.test(
    error.message
  );
}

function shouldFallbackToWasm(error: unknown): boolean {
  if (!(error instanceof Error)) {
    return false;
  }

  return /webgpu|gpu adapter|gpu device|wasm backend|backend.*not available|not supported/i.test(
    error.message
  );
}

async function getLocalPipelineCacheStatus(
  modelId: string,
  dtype: LocalDtype
): Promise<CacheCheckResult> {
  try {
    const internals = await getTransformersInternals();

    return await internals.is_pipeline_cached_files("text-generation", modelId, {
      dtype,
      local_files_only: true,
      revision: "main",
    });
  } catch (error) {
    logLocalInference("assets:local-cache-check:error", {
      modelId,
      dtype,
      message: error instanceof Error ? error.message : String(error),
    });

    return {
      allCached: false,
      files: [],
    };
  }
}

async function getCachedFilesMeta(modelId: string, files: string[]) {
  const internals = await getTransformersInternals();
  let bytesTotal = 0;

  await Promise.all(
    files.map(async (file) => {
      const metadata = await internals.get_file_metadata(modelId, file, {
        local_files_only: true,
        revision: "main",
      });

      if (typeof metadata.size === "number") {
        bytesTotal += metadata.size;
      }
    })
  );

  return {
    bytesTotal,
    fileCount: files.length,
  };
}

export async function getVerifiedLocalModelAssetState(
  modelId: string
): Promise<VerifiedLocalModelAssetState> {
  const cacheMeta = await getLocalModelCacheMeta(modelId).catch(() => null);
  const dtypeCandidates = Array.from(
    new Set([
      cacheMeta?.dtype,
      ...resolveDtypeCandidates(modelId),
    ].filter((value): value is LocalDtype => Boolean(value)))
  );

  let bestPartial: VerifiedLocalModelAssetState | null = null;

  for (const dtype of dtypeCandidates) {
    const cacheStatus = await getLocalPipelineCacheStatus(modelId, dtype);
    const cachedFiles = cacheStatus.files.filter((entry) => entry.cached);

    if (cacheStatus.allCached && cacheStatus.files.length > 0) {
      const cachedMeta = await getCachedFilesMeta(
        modelId,
        cacheStatus.files.map((entry) => entry.file)
      );
      const snapshot = {
        modelId,
        status: "complete",
        dtype,
        bytesTotal: cachedMeta.bytesTotal,
        fileCount: cachedMeta.fileCount,
        cachedFileCount: cacheStatus.files.length,
      } satisfies VerifiedLocalModelAssetState;

      await saveLocalModelAssetState({
        modelId,
        status: "complete",
        bytesTotal: snapshot.bytesTotal,
        fileCount: snapshot.fileCount,
      });

      return snapshot;
    }

    if (cachedFiles.length > 0) {
      const cachedMeta = await getCachedFilesMeta(
        modelId,
        cachedFiles.map((entry) => entry.file)
      );
      const snapshot = {
        modelId,
        status: "partial",
        dtype,
        bytesTotal: cachedMeta.bytesTotal,
        fileCount: cacheStatus.files.length,
        cachedFileCount: cachedFiles.length,
      } satisfies VerifiedLocalModelAssetState;

      if (
        !bestPartial ||
        snapshot.cachedFileCount > bestPartial.cachedFileCount
      ) {
        bestPartial = snapshot;
      }
    }
  }

  if (bestPartial) {
    await saveLocalModelAssetState({
      modelId,
      status: "partial",
      bytesTotal: bestPartial.bytesTotal,
      fileCount: bestPartial.fileCount,
    });
    return bestPartial;
  }

  await saveLocalModelAssetState({
    modelId,
    status: "missing",
    bytesTotal: 0,
    fileCount: 0,
  });

  return {
    modelId,
    status: "missing",
    dtype: null,
    bytesTotal: 0,
    fileCount: 0,
    cachedFileCount: 0,
  };
}

export async function ensureLocalModelAssets(params: {
  modelId: string;
  onLoadProgress?: (event: LocalModelLoadProgressEvent) => void;
  isLoadCancelled?: () => boolean;
}): Promise<{
  dtype: LocalDtype;
  bytesTotal: number;
  fileCount: number;
}> {
  const { modelId } = params;
  const existing = await getVerifiedLocalModelAssetState(modelId);

  if (existing.status === "complete" && existing.dtype) {
    logLocalInference("assets:already-complete", {
      modelId,
      dtype: existing.dtype,
      bytesTotal: existing.bytesTotal,
      fileCount: existing.fileCount,
    });

    return {
      dtype: existing.dtype,
      bytesTotal: existing.bytesTotal,
      fileCount: existing.fileCount,
    };
  }

  const internals = await getTransformersInternals();
  const dtypeCandidates = Array.from(
    new Set([
      existing.dtype,
      ...resolveDtypeCandidates(modelId),
    ].filter((value): value is LocalDtype => Boolean(value)))
  );

  let lastError: unknown = null;

  for (const dtype of dtypeCandidates) {
    if (params.isLoadCancelled?.()) {
      throw new Error("Model load cancelled");
    }

    const modelKey = `${modelId}:${dtype}`;
    const localCacheStatus = await getLocalPipelineCacheStatus(modelId, dtype);
    let requiredFiles = localCacheStatus.files.map((entry) => entry.file);

    if (requiredFiles.length === 0) {
      try {
        requiredFiles = await internals.get_pipeline_files(
          "text-generation",
          modelId,
          {
            dtype,
            revision: "main",
          }
        );
      } catch (error) {
        lastError = error;
        logLocalInference("assets:get-required-files:error", {
          modelId,
          dtype,
          message: error instanceof Error ? error.message : String(error),
        });
        continue;
      }
    }

    const cachedByFile = new Map(
      localCacheStatus.files.map((entry) => [entry.file, entry.cached])
    );
    const missingFiles = requiredFiles.filter(
      (file) => !cachedByFile.get(file)
    );

    if (missingFiles.length === 0) {
      const verified = await getVerifiedLocalModelAssetState(modelId);
      if (verified.status === "complete" && verified.dtype) {
        return {
          dtype: verified.dtype,
          bytesTotal: verified.bytesTotal,
          fileCount: verified.fileCount,
        };
      }
      continue;
    }

    await clearLocalModelDownloadRecords(modelKey);
    await saveLocalModelAssetState({
      modelId,
      status: "partial",
      fileCount: requiredFiles.length,
    });

    for (const file of requiredFiles) {
      if (cachedByFile.get(file)) {
        void upsertLocalModelDownloadRecord({
          modelKey,
          modelId,
          dtype,
          file,
          status: "done",
          loadedBytes: 0,
          totalBytes: 0,
        });
      }
    }

    const progressByFile = new Map<
      string,
      {
        loaded: number;
        total: number;
        done: boolean;
      }
    >();

    const emitAggregateProgress = (currentFile?: string) => {
      let loaded = 0;
      let total = 0;
      let completedFiles = 0;

      for (const file of missingFiles) {
        const progress = progressByFile.get(file);
        if (!progress) {
          continue;
        }

        loaded += progress.loaded;
        total += progress.total;
        if (progress.done) {
          completedFiles += 1;
        }
      }

      const progressValue =
        total > 0
          ? Math.max(0, Math.min(100, (loaded / total) * 100))
          : Math.max(
              0,
              Math.min(100, (completedFiles / missingFiles.length) * 100)
            );

      emitLoadProgress(params, {
        phase: "downloading",
        message: `正在下载模型文件 (${completedFiles}/${missingFiles.length})${currentFile ? ` · ${currentFile}` : ""}`,
        progress: Number.isFinite(progressValue) ? progressValue : null,
        loaded,
        total,
      });
    };

    logLocalInference("assets:download:start", {
      modelId,
      dtype,
      requiredFiles,
      missingFiles,
    });

    try {
      for (const file of missingFiles) {
        if (params.isLoadCancelled?.()) {
          throw new Error("Model load cancelled");
        }

        progressByFile.set(file, {
          loaded: 0,
          total: 0,
          done: false,
        });
        emitAggregateProgress(file);

        await internals.getModelFile(modelId, file, true, {
          revision: "main",
          local_files_only: false,
          progress_callback: (info: HubProgressInfo) => {
            if (params.isLoadCancelled?.()) {
              return;
            }

            const fileName = info.file ?? file;
            const previous = progressByFile.get(fileName) ?? {
              loaded: 0,
              total: 0,
              done: false,
            };
            const next = {
              loaded:
                typeof info.loaded === "number"
                  ? Math.max(previous.loaded, info.loaded)
                  : previous.loaded,
              total:
                typeof info.total === "number" && info.total > 0
                  ? Math.max(previous.total, info.total)
                  : previous.total,
              done:
                info.status === "done" || info.status === "ready"
                  ? true
                  : previous.done,
            };

            progressByFile.set(fileName, next);

            if (
              info.status === "initiate" ||
              info.status === "download" ||
              info.status === "progress" ||
              info.status === "progress_total"
            ) {
              void saveLocalModelAssetState({
                modelId,
                status: "partial",
                fileCount: requiredFiles.length,
              });
              void upsertLocalModelDownloadRecord({
                modelKey,
                modelId,
                dtype,
                file: fileName,
                status: "downloading",
                loadedBytes: next.loaded,
                totalBytes: next.total,
              });
            }

            if (info.status === "done" || info.status === "ready") {
              void upsertLocalModelDownloadRecord({
                modelKey,
                modelId,
                dtype,
                file: fileName,
                status: "done",
                loadedBytes: next.total > 0 ? next.total : next.loaded,
                totalBytes: next.total,
              });
            }

            emitAggregateProgress(fileName);
          },
        });

        const fileProgress = progressByFile.get(file);
        progressByFile.set(file, {
          loaded: fileProgress?.total ?? fileProgress?.loaded ?? 0,
          total: fileProgress?.total ?? fileProgress?.loaded ?? 0,
          done: true,
        });
        await upsertLocalModelDownloadRecord({
          modelKey,
          modelId,
          dtype,
          file,
          status: "done",
          loadedBytes: fileProgress?.total ?? fileProgress?.loaded ?? 0,
          totalBytes: fileProgress?.total ?? fileProgress?.loaded ?? 0,
        });
        emitAggregateProgress(file);
      }

      const verified = await getVerifiedLocalModelAssetState(modelId);
      if (verified.status === "complete" && verified.dtype) {
        logLocalInference("assets:download:complete", {
          modelId,
          dtype: verified.dtype,
          bytesTotal: verified.bytesTotal,
          fileCount: verified.fileCount,
        });

        return {
          dtype: verified.dtype,
          bytesTotal: verified.bytesTotal,
          fileCount: verified.fileCount,
        };
      }
    } catch (error) {
      lastError = error;
      logLocalInference("assets:download:error", {
        modelId,
        dtype,
        message: error instanceof Error ? error.message : String(error),
      });

      await saveLocalModelAssetState({
        modelId,
        status: "partial",
        fileCount: requiredFiles.length,
      });

      if (params.isLoadCancelled?.()) {
        throw error;
      }
    }
  }

  throw lastError instanceof Error
    ? lastError
    : new Error(`Failed to download required assets for ${modelId}`);
}

async function getRuntime(
  modelId: string,
  options?: RuntimeLoadOptions
): Promise<ModelRuntime> {
  const preferredDevice = hasWebGpuSupport() ? "webgpu" : "wasm";
  const key = `${modelId}:runtime`;
  const existing = runtimes.get(key);
  if (existing) {
    logLocalInference("getRuntime:reuse-existing-runtime", {
      modelId,
      cacheKey: key,
    });
    const runtime = await existing;
    if (options?.isLoadCancelled?.()) {
      logLocalInference("getRuntime:cancelled-after-reuse", {
        modelId,
        cacheKey: key,
      });
      throw new Error("Model load cancelled");
    }
    logLocalInference("getRuntime:reuse-existing-runtime:done", {
      modelId,
      device: runtime.device,
      dtype: runtime.dtype,
    });
    return runtime;
  }

  const verifiedAssets = await getVerifiedLocalModelAssetState(modelId);
  const dtypeCandidates = Array.from(
    new Set([
      verifiedAssets.dtype,
      ...resolveDtypeCandidates(modelId),
    ].filter((value): value is LocalDtype => Boolean(value)))
  );
  logLocalInference("getRuntime:start", {
    modelId,
    preferredDevice,
    dtypeCandidates,
    verifiedAssetStatus: verifiedAssets.status,
    cacheKey: key,
  });

  emitLoadProgress(options, {
    phase: "initializing",
    message: "正在初始化推理引擎...",
    progress: null,
    loaded: 0,
    total: 0,
  });

  const loading = (async () => {
    const { pipeline } = await getTransformersInternals();

    const createRuntimeForDevice = async (
      device: "webgpu" | "wasm"
    ): Promise<ModelRuntime> => {
      let lastError: unknown = null;

      for (const dtype of dtypeCandidates) {
        try {
          logLocalInference("getRuntime:create-pipeline", {
            modelId,
            device,
            dtype,
            localFilesOnly: true,
          });
          const generator = await pipeline("text-generation", modelId, {
            device,
            dtype,
            local_files_only: true,
          });

          const loadMeta = extractLoadMeta(generator);
          logLocalInference("getRuntime:create-pipeline:done", {
            modelId,
            device,
            dtype,
            bytesTotal: loadMeta.bytesTotal,
            fileCount: loadMeta.fileCount,
            fromCacheCount: loadMeta.fromCacheCount,
          });
          void saveLocalModelAssetState({
            modelId,
            status: "complete",
            bytesTotal: loadMeta.bytesTotal,
            fileCount: loadMeta.fileCount,
          });
          void saveLocalModelCacheMeta({
            modelId,
            dtype,
            device,
            ...loadMeta,
          });

          return {
            generator,
            device,
            dtype,
            loadMeta,
          } satisfies ModelRuntime;
        } catch (error) {
          lastError = error;
          logLocalInference("getRuntime:create-pipeline:error", {
            modelId,
            device,
            dtype,
            message: error instanceof Error ? error.message : String(error),
          });

          if (isMissingLocalAssetError(error)) {
            await saveLocalModelAssetState({
              modelId,
              status: "partial",
            });
          }
        }
      }

      await saveLocalModelAssetState({
        modelId,
        status: "partial",
      });
      throw lastError instanceof Error
        ? lastError
        : new Error(`Failed to create runtime for ${modelId}`);
    };

    try {
      return await createRuntimeForDevice(preferredDevice);
    } catch (primaryError) {
      if (preferredDevice === "wasm" || !shouldFallbackToWasm(primaryError)) {
        throw primaryError;
      }

      logLocalInference("getRuntime:create-pipeline:fallback-to-wasm", {
        modelId,
        message:
          primaryError instanceof Error ? primaryError.message : String(primaryError),
      });
      emitLoadProgress(options, {
        phase: "initializing",
        message: "WebGPU 不可用，正在回退到 WASM...",
        progress: null,
        loaded: 0,
        total: 0,
      });

      return createRuntimeForDevice("wasm");
    }
  })();

  runtimes.set(key, loading);

  try {
    const runtime = await loading;
    loadedModelIds.add(modelId);
    logLocalInference("getRuntime:ready", {
      modelId,
      device: runtime.device,
      dtype: runtime.dtype,
    });

    if (options?.isLoadCancelled?.()) {
      logLocalInference("getRuntime:cancelled-after-ready", {
        modelId,
      });
      throw new Error("Model load cancelled");
    }

    return runtime;
  } catch (error) {
    logLocalInference("getRuntime:error", {
      modelId,
      message: error instanceof Error ? error.message : String(error),
    });
    runtimes.delete(key);
    throw error;
  }
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
  logLocalInference("generate:start", {
    modelId,
    reasoningMode,
    messageCount: messages.length,
  });
  const runtime = await getRuntime(modelId, {
    onLoadProgress,
    isLoadCancelled,
  });
  logLocalInference("generate:runtime-ready", {
    modelId,
    device: runtime.device,
    dtype: runtime.dtype,
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
    logLocalInference("generate:invoke-generator", {
      modelId,
      inputType: "messages",
      promptMessageCount: promptMessages.length,
    });
    output = await runtime.generator(promptMessages, generationOptions);
  } catch (error) {
    if (!isChatTemplateMissingError(error)) {
      logLocalInference("generate:invoke-generator:error", {
        modelId,
        message: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }

    const plainPrompt = buildPlainPrompt(promptMessages);
    plainPromptUsed = plainPrompt;
    generationInput = plainPrompt;
    logLocalInference("generate:invoke-generator:fallback-plain-prompt", {
      modelId,
      promptLength: plainPrompt.length,
    });
    output = await runtime.generator(plainPrompt, generationOptions);
  }

  if (signal?.aborted) {
    logLocalInference("generate:aborted-after-generator", {
      modelId,
    });
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
      logLocalInference("generate:retry-deterministic", {
        modelId,
      });
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
      logLocalInference("generate:retry-deterministic:failed", {
        modelId,
      });
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
  const assets = await ensureLocalModelAssets({
    modelId: params.modelId,
    onLoadProgress: params.onLoadProgress,
    isLoadCancelled: params.isLoadCancelled,
  });

  return {
    dtype: assets.dtype,
    device: hasWebGpuSupport() ? "webgpu" : "wasm",
    loadMeta: {
      bytesTotal: assets.bytesTotal,
      fileCount: assets.fileCount,
      fromCacheCount: 0,
    },
  };
}

export function hasLoadedLocalModel(modelId: string): boolean {
  return loadedModelIds.has(modelId);
}
