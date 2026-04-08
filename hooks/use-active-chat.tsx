"use client";

import type { UseChatHelpers } from "@ai-sdk/react";
import { usePathname } from "next/navigation";
import {
  createContext,
  useCallback,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useDataStream } from "@/components/chat/data-stream-provider";
import { toast } from "@/components/chat/toast";
import type { VisibilityType } from "@/components/chat/visibility-selector";
import { allowedModelIds, DEFAULT_CHAT_MODEL } from "@/lib/ai/models";
import type { Vote } from "@/lib/db/schema";
import {
  generateLocalAssistantResponse,
  hasLoadedLocalModel,
  prepareLocalModel,
  type LocalModelLoadPhase,
  type LocalModelLoadProgressEvent,
  type LocalReasoningMode,
} from "@/lib/local-inference";
import {
  type LocalMemoryCategory,
  getLocalChatById,
  listLocalMemories,
  getLocalMessages,
  getLocalSetting,
  replaceLocalMessages,
  saveLocalChat,
  saveLocalSetting,
  upsertLocalMemory,
} from "@/lib/local-chat-store";
import type { ChatMessage } from "@/lib/types";
import { generateUUID } from "@/lib/utils";

export type ModelLoadProgressState = {
  sessionId: string;
  modelId: string;
  phase: LocalModelLoadPhase;
  progress: number | null;
  loadedBytes: number;
  totalBytes: number;
  message: string;
  startedAt: number;
  updatedAt: number;
  canCancel: boolean;
};

export type ReasoningMode = LocalReasoningMode;

type ActiveChatContextValue = {
  chatId: string;
  messages: ChatMessage[];
  setMessages: UseChatHelpers<ChatMessage>["setMessages"];
  sendMessage: UseChatHelpers<ChatMessage>["sendMessage"];
  status: UseChatHelpers<ChatMessage>["status"];
  stop: UseChatHelpers<ChatMessage>["stop"];
  regenerate: UseChatHelpers<ChatMessage>["regenerate"];
  addToolApprovalResponse: UseChatHelpers<ChatMessage>["addToolApprovalResponse"];
  input: string;
  setInput: Dispatch<SetStateAction<string>>;
  visibilityType: VisibilityType;
  isReadonly: boolean;
  isLoading: boolean;
  votes: Vote[] | undefined;
  currentModelId: string;
  setCurrentModelId: (id: string) => void;
  reasoningMode: ReasoningMode;
  setReasoningMode: (mode: ReasoningMode) => void;
  modelLoadProgress: ModelLoadProgressState | null;
  cancelModelLoad: () => void;
  showCreditCardAlert: boolean;
  setShowCreditCardAlert: Dispatch<SetStateAction<boolean>>;
};

const ActiveChatContext = createContext<ActiveChatContextValue | null>(null);

function getMessageTimestamp(message: ChatMessage): number {
  const createdAt = message.metadata?.createdAt;
  if (!createdAt) {
    return Date.now();
  }
  const parsed = Date.parse(createdAt);
  return Number.isNaN(parsed) ? Date.now() : parsed;
}

function extractFirstUserText(messages: ChatMessage[]): string {
  const firstUser = messages.find((message) => message.role === "user");
  if (!firstUser) {
    return "New Chat";
  }

  const text = firstUser.parts
    .filter((part) => part.type === "text")
    .map((part) => part.text)
    .join(" ")
    .trim();

  if (!text) {
    return "New Chat";
  }

  return text.length > 60 ? `${text.slice(0, 60)}...` : text;
}

function resolveModelId(candidate: string | null | undefined): string {
  if (candidate && allowedModelIds.has(candidate)) {
    return candidate;
  }

  return DEFAULT_CHAT_MODEL;
}

function resolveReasoningMode(
  candidate: string | null | undefined
): ReasoningMode {
  return candidate === "thinking" ? "thinking" : "normal";
}

function getMessageText(message: ChatMessage): string {
  return message.parts
    .filter((part) => part.type === "text")
    .map((part) => part.text)
    .join("\n")
    .trim();
}

async function buildMemorySystemMessage(
  chatId: string
): Promise<ChatMessage | null> {
  const [preferences, facts, sessionContexts] = await Promise.all([
    listLocalMemories("preference"),
    listLocalMemories("fact"),
    listLocalMemories("session-context"),
  ]);

  const sections: string[] = [];

  if (preferences.length > 0) {
    sections.push(
      `用户偏好：${preferences
        .slice(0, 5)
        .map((item) => `${item.key}=${item.value}`)
        .join("；")}`
    );
  }

  if (facts.length > 0) {
    sections.push(
      `已知事实：${facts
        .slice(0, 5)
        .map((item) => `${item.key}=${item.value}`)
        .join("；")}`
    );
  }

  const relatedContext = sessionContexts.filter(
    (item) => item.sourceChatId === chatId
  );
  const contextPool = relatedContext.length > 0 ? relatedContext : sessionContexts;

  if (contextPool.length > 0) {
    sections.push(
      `上下文：${contextPool
        .slice(0, 3)
        .map((item) => item.value)
        .join("；")}`
    );
  }

  if (sections.length === 0) {
    return null;
  }

  return {
    id: `memory-system-${chatId}`,
    role: "system",
    parts: [
      {
        type: "text",
        text: `请结合以下长期记忆回答，若与当前用户输入冲突，以当前输入为准。\n${sections.join("\n")}`,
      },
    ],
    metadata: {
      createdAt: new Date().toISOString(),
    },
  };
}

function extractMemoryCandidates(text: string): Array<{
  category: LocalMemoryCategory;
  key: string;
  value: string;
  confidence: number;
}> {
  const candidates: Array<{
    category: LocalMemoryCategory;
    key: string;
    value: string;
    confidence: number;
  }> = [];

  const normalized = text.trim();
  if (!normalized) {
    return candidates;
  }

  const nameMatch = normalized.match(/我叫\s*([^，。,.!?\s]{2,20})/);
  if (nameMatch?.[1]) {
    candidates.push({
      category: "fact",
      key: "user-name",
      value: nameMatch[1],
      confidence: 0.95,
    });
  }

  const roleMatch = normalized.match(/我是\s*([^，。,.!?]{2,24})/);
  if (roleMatch?.[1]) {
    candidates.push({
      category: "fact",
      key: "user-role",
      value: roleMatch[1].trim(),
      confidence: 0.85,
    });
  }

  const languageMatch = normalized.match(/请用\s*(中文|英文|英语)\s*([回答回复输出]?)/);
  if (languageMatch?.[1]) {
    candidates.push({
      category: "preference",
      key: "response-language",
      value: languageMatch[1].includes("英") ? "en" : "zh",
      confidence: 0.95,
    });
  }

  if (/简短|简洁|一句话/.test(normalized)) {
    candidates.push({
      category: "preference",
      key: "response-style",
      value: "concise",
      confidence: 0.8,
    });
  }

  if (/详细|展开|尽可能完整/.test(normalized)) {
    candidates.push({
      category: "preference",
      key: "response-style",
      value: "detailed",
      confidence: 0.8,
    });
  }

  if (/这轮|当前任务|这个项目|我们项目|先做/.test(normalized)) {
    candidates.push({
      category: "session-context",
      key: "active-context",
      value: normalized.slice(0, 240),
      confidence: 0.75,
    });
  }

  return candidates;
}

function splitTextForStreaming(text: string): string[] {
  const tokens = text.split(/(\s+)/).filter((token) => token.length > 0);
  const chunks: string[] = [];
  let buffer = "";

  for (const token of tokens) {
    buffer += token;
    if (buffer.length >= 24 || /[.!?\n。！？]$/.test(buffer)) {
      chunks.push(buffer);
      buffer = "";
    }
  }

  if (buffer.length > 0) {
    chunks.push(buffer);
  }

  return chunks.length > 0 ? chunks : [text];
}

async function wait(ms: number) {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

function extractChatId(pathname: string): string | null {
  const match = pathname.match(/\/chat\/([^/]+)/);
  return match ? match[1] : null;
}

export function ActiveChatProvider({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const { setDataStream } = useDataStream();

  const chatIdFromUrl = extractChatId(pathname);
  const isNewChat = !chatIdFromUrl;
  const newChatIdRef = useRef(generateUUID());
  const prevPathnameRef = useRef(pathname);

  if (isNewChat && prevPathnameRef.current !== pathname) {
    newChatIdRef.current = generateUUID();
  }
  prevPathnameRef.current = pathname;

  const chatId = chatIdFromUrl ?? newChatIdRef.current;

  const [currentModelId, setCurrentModelId] = useState(DEFAULT_CHAT_MODEL);
  const [reasoningMode, setReasoningMode] = useState<ReasoningMode>("normal");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [status, setStatus] = useState<UseChatHelpers<ChatMessage>["status"]>(
    "ready"
  );
  const [modelLoadProgress, setModelLoadProgress] =
    useState<ModelLoadProgressState | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [visibilityType, setVisibilityType] =
    useState<VisibilityType>("private");

  const currentModelIdRef = useRef(currentModelId);
  const reasoningModeRef = useRef(reasoningMode);
  const messagesRef = useRef<ChatMessage[]>(messages);
  const abortControllerRef = useRef<AbortController | null>(null);
  const modelLoadProgressRef = useRef<ModelLoadProgressState | null>(null);
  const loadSessionSequenceRef = useRef(0);
  const activeLoadSessionRef = useRef<string | null>(null);
  const cancelledLoadSessionsRef = useRef(new Set<string>());
  const hideModelLoadTimerRef = useRef<ReturnType<typeof setTimeout> | null>(
    null
  );

  useEffect(() => {
    currentModelIdRef.current = currentModelId;
  }, [currentModelId]);

  useEffect(() => {
    reasoningModeRef.current = reasoningMode;
  }, [reasoningMode]);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  useEffect(() => {
    modelLoadProgressRef.current = modelLoadProgress;
  }, [modelLoadProgress]);

  const [input, setInput] = useState("");
  const [showCreditCardAlert, setShowCreditCardAlert] = useState(false);

  useEffect(() => {
    let isMounted = true;

    const load = async () => {
      setIsLoading(true);

      try {
        const [chat, rows, savedModel, savedReasoningMode] = await Promise.all([
          getLocalChatById(chatId),
          getLocalMessages<ChatMessage>(chatId),
          getLocalSetting("local-chat-model"),
          getLocalSetting("local-reasoning-mode"),
        ]);

        if (!isMounted) {
          return;
        }

        setMessages(rows.map((row) => row.message));
        setVisibilityType(chat?.visibility ?? "private");

        setCurrentModelId(resolveModelId(chat?.modelId ?? savedModel));
        setReasoningMode(resolveReasoningMode(savedReasoningMode));
      } catch {
        if (isMounted) {
          setMessages([]);
          setVisibilityType("private");
          setCurrentModelId(DEFAULT_CHAT_MODEL);
          setReasoningMode("normal");
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    load();

    return () => {
      isMounted = false;
    };
  }, [chatId]);

  useEffect(() => {
    void saveLocalSetting("local-chat-model", currentModelId);
  }, [currentModelId]);

  useEffect(() => {
    void saveLocalSetting("local-reasoning-mode", reasoningMode);
  }, [reasoningMode]);

  const persistConversation = async (nextMessages: ChatMessage[]) => {
    if (nextMessages.length === 0) {
      return;
    }

    await saveLocalChat({
      chatId,
      title: extractFirstUserText(nextMessages),
      visibility: visibilityType,
      modelId: currentModelIdRef.current,
    });

    await replaceLocalMessages(
      chatId,
      nextMessages.map((message) => ({
        id: message.id,
        createdAt: getMessageTimestamp(message),
        message,
      }))
    );
  };

  const clearModelLoadHideTimer = useCallback(() => {
    if (hideModelLoadTimerRef.current !== null) {
      clearTimeout(hideModelLoadTimerRef.current);
      hideModelLoadTimerRef.current = null;
    }
  }, []);

  const isLoadSessionActive = useCallback((sessionId: string) => {
    return (
      activeLoadSessionRef.current === sessionId &&
      !cancelledLoadSessionsRef.current.has(sessionId)
    );
  }, []);

  const scheduleModelLoadHide = useCallback(
    (sessionId: string, delay = 1400) => {
      clearModelLoadHideTimer();
      hideModelLoadTimerRef.current = setTimeout(() => {
        setModelLoadProgress((previous) => {
          if (!previous || previous.sessionId !== sessionId) {
            return previous;
          }

          if (
            previous.phase === "ready" ||
            previous.phase === "cancelled" ||
            previous.phase === "failed"
          ) {
            if (activeLoadSessionRef.current === sessionId) {
              activeLoadSessionRef.current = null;
            }
            return null;
          }

          return previous;
        });
      }, delay);
    },
    [clearModelLoadHideTimer]
  );

  const startModelLoadSession = useCallback(
    (modelId: string) => {
      clearModelLoadHideTimer();
      cancelledLoadSessionsRef.current.clear();
      loadSessionSequenceRef.current += 1;

      const sessionId = `${Date.now()}-${loadSessionSequenceRef.current}`;
      const now = Date.now();

      activeLoadSessionRef.current = sessionId;
      setModelLoadProgress({
        sessionId,
        modelId,
        phase: "initializing",
        progress: null,
        loadedBytes: 0,
        totalBytes: 0,
        message: "正在初始化推理引擎...",
        startedAt: now,
        updatedAt: now,
        canCancel: true,
      });

      return sessionId;
    },
    [clearModelLoadHideTimer]
  );

  const applyModelLoadProgress = useCallback(
    (sessionId: string, event: LocalModelLoadProgressEvent) => {
      if (!isLoadSessionActive(sessionId)) {
        return;
      }

      setModelLoadProgress((previous) => {
        if (!previous || previous.sessionId !== sessionId) {
          return previous;
        }

        return {
          ...previous,
          phase: event.phase,
          progress:
            typeof event.progress === "number"
              ? event.progress
              : previous.progress,
          loadedBytes:
            typeof event.loaded === "number"
              ? Math.max(previous.loadedBytes, event.loaded)
              : previous.loadedBytes,
          totalBytes:
            typeof event.total === "number" && event.total > 0
              ? Math.max(previous.totalBytes, event.total)
              : previous.totalBytes,
          message: event.message,
          updatedAt: Date.now(),
          canCancel:
            event.phase === "downloading" || event.phase === "initializing",
        };
      });
    },
    [isLoadSessionActive]
  );

  const markModelLoadComplete = useCallback(
    (sessionId: string) => {
      if (!isLoadSessionActive(sessionId)) {
        return;
      }

      setModelLoadProgress((previous) => {
        if (!previous || previous.sessionId !== sessionId) {
          return previous;
        }

        return {
          ...previous,
          phase: "ready",
          progress: 100,
          loadedBytes:
            previous.totalBytes > 0 ? previous.totalBytes : previous.loadedBytes,
          message: "模型已就绪",
          updatedAt: Date.now(),
          canCancel: false,
        };
      });
      scheduleModelLoadHide(sessionId);
    },
    [isLoadSessionActive, scheduleModelLoadHide]
  );

  const markModelLoadFailed = useCallback(
    (sessionId: string, message: string) => {
      if (!isLoadSessionActive(sessionId)) {
        return;
      }

      setModelLoadProgress((previous) => {
        if (!previous || previous.sessionId !== sessionId) {
          return previous;
        }

        return {
          ...previous,
          phase: "failed",
          progress: null,
          message,
          updatedAt: Date.now(),
          canCancel: false,
        };
      });
      scheduleModelLoadHide(sessionId, 2800);
    },
    [isLoadSessionActive, scheduleModelLoadHide]
  );

  const cancelModelLoad = useCallback(() => {
    const activeSessionId = activeLoadSessionRef.current;
    if (!activeSessionId) {
      return;
    }

    cancelledLoadSessionsRef.current.add(activeSessionId);
    activeLoadSessionRef.current = null;
    clearModelLoadHideTimer();

    setModelLoadProgress((previous) => {
      if (!previous || previous.sessionId !== activeSessionId) {
        return previous;
      }

      return {
        ...previous,
        phase: "cancelled",
        progress: null,
        message: "已取消当前加载并切换模型",
        updatedAt: Date.now(),
        canCancel: false,
      };
    });

    scheduleModelLoadHide(activeSessionId, 1200);
  }, [clearModelLoadHideTimer, scheduleModelLoadHide]);

  const prepareModel = useCallback(
    async (modelId: string) => {
      const sessionId = startModelLoadSession(modelId);

      try {
        await prepareLocalModel({
          modelId,
          onLoadProgress: (event) => applyModelLoadProgress(sessionId, event),
          isLoadCancelled: () => !isLoadSessionActive(sessionId),
        });

        markModelLoadComplete(sessionId);
      } catch (error) {
        if (!isLoadSessionActive(sessionId)) {
          return;
        }

        markModelLoadFailed(
          sessionId,
          error instanceof Error ? error.message : "模型加载失败，请重试"
        );
      }
    },
    [
      applyModelLoadProgress,
      isLoadSessionActive,
      markModelLoadComplete,
      markModelLoadFailed,
      startModelLoadSession,
    ]
  );

  const handleModelChange = useCallback(
    (modelId: string) => {
      if (!allowedModelIds.has(modelId)) {
        setCurrentModelId(DEFAULT_CHAT_MODEL);
        return;
      }

      if (modelId === currentModelIdRef.current) {
        return;
      }

      cancelModelLoad();
      setCurrentModelId(modelId);
    },
    [cancelModelLoad]
  );

  const runAssistantGeneration = async (baseMessages: ChatMessage[]) => {
    const modelId = currentModelIdRef.current;
    const controller = new AbortController();
    abortControllerRef.current = controller;
    setStatus("submitted");

    let loadSessionId: string | null = null;
    const hasModelLoaded = hasLoadedLocalModel(modelId);

    if (!hasModelLoaded) {
      const currentLoad = modelLoadProgressRef.current;
      if (
        currentLoad &&
        currentLoad.modelId === modelId &&
        (currentLoad.phase === "downloading" ||
          currentLoad.phase === "initializing")
      ) {
        loadSessionId = currentLoad.sessionId;
      } else {
        loadSessionId = startModelLoadSession(modelId);
      }
    }

    try {
      const memorySystemMessage = await buildMemorySystemMessage(chatId);
      const inferenceMessages = memorySystemMessage
        ? [memorySystemMessage, ...baseMessages]
        : baseMessages;

      const result = await generateLocalAssistantResponse({
        modelId,
        messages: inferenceMessages,
        reasoningMode: reasoningModeRef.current,
        signal: controller.signal,
        onLoadProgress: loadSessionId
          ? (event) => applyModelLoadProgress(loadSessionId as string, event)
          : undefined,
        isLoadCancelled: loadSessionId
          ? () => !isLoadSessionActive(loadSessionId as string)
          : undefined,
      });

      if (loadSessionId) {
        markModelLoadComplete(loadSessionId);
      }

      if (controller.signal.aborted) {
        setStatus("ready");
        return;
      }

      const assistantId = generateUUID();
      const createdAt = new Date().toISOString();
      const chunks = splitTextForStreaming(result.text);
      const reasoningPart =
        reasoningModeRef.current === "thinking" &&
        result.reasoningText.trim().length > 0
          ? ([
              {
                type: "reasoning",
                text: result.reasoningText,
                state: "done",
              },
            ] as any[])
          : [];

      setStatus("streaming");
      setMessages([
        ...baseMessages,
        {
          id: assistantId,
          role: "assistant",
          parts: [...reasoningPart, { type: "text", text: "" }] as any,
          metadata: {
            createdAt,
          },
        },
      ]);

      let streamedText = "";
      for (const chunk of chunks) {
        if (controller.signal.aborted) {
          setStatus("ready");
          setMessages(baseMessages);
          await persistConversation(baseMessages);
          return;
        }

        streamedText += chunk;
        setMessages([
          ...baseMessages,
          {
            id: assistantId,
            role: "assistant",
            parts: [...reasoningPart, { type: "text", text: streamedText }] as any,
            metadata: {
              createdAt,
            },
          },
        ]);

        await wait(20);
      }

      const assistantMessage: ChatMessage = {
        id: assistantId,
        role: "assistant",
        parts: [...reasoningPart, { type: "text", text: result.text }] as any,
        metadata: {
          createdAt,
        },
      };

      const nextMessages = [...baseMessages, assistantMessage];
      setMessages(nextMessages);
      await persistConversation(nextMessages);
      setStatus("ready");
      setShowCreditCardAlert(false);
    } catch (error) {
      if (controller.signal.aborted) {
        setStatus("ready");
        return;
      }

      if (loadSessionId && isLoadSessionActive(loadSessionId)) {
        markModelLoadFailed(
          loadSessionId,
          error instanceof Error ? error.message : "模型加载失败，请重试"
        );
      }

      setStatus("error");
      toast({
        type: "error",
        description:
          error instanceof Error
            ? error.message
            : "Local model generation failed",
      });
    } finally {
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
    }
  };

  const sendMessage: UseChatHelpers<ChatMessage>["sendMessage"] = async (
    message
  ) => {
    if (!message) {
      return;
    }

    const role =
      "role" in message && message.role ? (message.role as ChatMessage["role"]) : "user";

    const parts =
      "parts" in message && Array.isArray(message.parts)
        ? (message.parts as ChatMessage["parts"])
        : "text" in message && typeof message.text === "string"
          ? ([{ type: "text", text: message.text }] as ChatMessage["parts"])
          : [];

    if (parts.length === 0) {
      return;
    }

    const createdAt = new Date().toISOString();
    const nextUserMessage: ChatMessage = {
      id:
        "id" in message && typeof message.id === "string"
          ? message.id
          : generateUUID(),
      role,
      parts,
      metadata: {
        createdAt,
      },
    };

    const nextMessages = [...messagesRef.current, nextUserMessage];
    setMessages(nextMessages);
    setDataStream([]);

    await persistConversation(nextMessages);

    if (nextUserMessage.role === "user") {
      const candidates = extractMemoryCandidates(getMessageText(nextUserMessage));
      if (candidates.length > 0) {
        await Promise.all(
          candidates.map((candidate) =>
            upsertLocalMemory({
              category: candidate.category,
              key: candidate.key,
              value: candidate.value,
              confidence: candidate.confidence,
              sourceChatId: chatId,
            })
          )
        );
      }
    }

    if (nextUserMessage.role === "user") {
      await runAssistantGeneration(nextMessages);
    }
  };

  const stop: UseChatHelpers<ChatMessage>["stop"] = async () => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setStatus("ready");
  };

  const regenerate: UseChatHelpers<ChatMessage>["regenerate"] = async () => {
    const current = [...messagesRef.current];
    const lastMessage = current.at(-1);

    if (lastMessage?.role === "assistant") {
      current.pop();
    }

    setMessages(current);
    await persistConversation(current);

    if (current.length > 0) {
      await runAssistantGeneration(current);
    }
  };

  const addToolApprovalResponse: UseChatHelpers<ChatMessage>["addToolApprovalResponse"] =
    async () => undefined;

  const resumeStream: UseChatHelpers<ChatMessage>["resumeStream"] =
    async () => undefined;

  const prevChatIdRef = useRef(chatId);
  useEffect(() => {
    if (prevChatIdRef.current !== chatId) {
      prevChatIdRef.current = chatId;
      if (isNewChat) {
        setMessages([]);
      }
      setStatus("ready");
      abortControllerRef.current?.abort();
      abortControllerRef.current = null;
      cancelModelLoad();
    }
  }, [cancelModelLoad, chatId, isNewChat, setMessages]);

  useEffect(() => {
    return () => {
      clearModelLoadHideTimer();
    };
  }, [clearModelLoadHideTimer]);

  const hasAppendedQueryRef = useRef(false);
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const query = params.get("query");
    if (query && !hasAppendedQueryRef.current) {
      hasAppendedQueryRef.current = true;
      window.history.replaceState(
        {},
        "",
        `${process.env.NEXT_PUBLIC_BASE_PATH ?? ""}/chat/${chatId}`
      );
      sendMessage({
        role: "user" as const,
        parts: [{ type: "text", text: query }],
      });
    }
  }, [sendMessage, chatId]);

  const isReadonly = false;
  const votes: Vote[] | undefined = undefined;

  const value = useMemo<ActiveChatContextValue>(
    () => ({
      chatId,
      messages,
      setMessages,
      sendMessage,
      status,
      stop,
      regenerate,
      addToolApprovalResponse,
      input,
      setInput,
      visibilityType,
      isReadonly,
      isLoading,
      votes,
      currentModelId,
      setCurrentModelId: handleModelChange,
      reasoningMode,
      setReasoningMode,
      modelLoadProgress,
      cancelModelLoad,
      showCreditCardAlert,
      setShowCreditCardAlert,
    }),
    [
      chatId,
      messages,
      setMessages,
      sendMessage,
      status,
      stop,
      regenerate,
      addToolApprovalResponse,
      input,
      visibilityType,
      isReadonly,
      isLoading,
      votes,
      currentModelId,
      handleModelChange,
      reasoningMode,
      modelLoadProgress,
      cancelModelLoad,
      showCreditCardAlert,
    ]
  );

  return (
    <ActiveChatContext.Provider value={value}>
      {children}
    </ActiveChatContext.Provider>
  );
}

export function useActiveChat() {
  const context = useContext(ActiveChatContext);
  if (!context) {
    throw new Error("useActiveChat must be used within ActiveChatProvider");
  }
  return context;
}
