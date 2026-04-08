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
} from "@/lib/local-inference";
import {
  getLocalChatById,
  getLocalMessages,
  getLocalSetting,
  replaceLocalMessages,
  saveLocalChat,
  saveLocalSetting,
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
  const messagesRef = useRef<ChatMessage[]>(messages);
  const abortControllerRef = useRef<AbortController | null>(null);
  const runtimeNoticeRef = useRef<string>("");
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
        const [chat, rows, savedModel] = await Promise.all([
          getLocalChatById(chatId),
          getLocalMessages<ChatMessage>(chatId),
          getLocalSetting("local-chat-model"),
        ]);

        if (!isMounted) {
          return;
        }

        setMessages(rows.map((row) => row.message));
        setVisibilityType(chat?.visibility ?? "private");

        setCurrentModelId(resolveModelId(chat?.modelId ?? savedModel));
      } catch {
        if (isMounted) {
          setMessages([]);
          setVisibilityType("private");
          setCurrentModelId(DEFAULT_CHAT_MODEL);
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
            previous.phase === "cancelled"
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
    },
    [isLoadSessionActive]
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
      void prepareModel(modelId);
    },
    [cancelModelLoad, prepareModel]
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
      const result = await generateLocalAssistantResponse({
        modelId,
        messages: baseMessages,
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

      const runtimeNoticeKey = `${currentModelIdRef.current}:${result.device}:${result.dtype}:${result.loadMeta.fromCacheCount}`;
      if (runtimeNoticeRef.current !== runtimeNoticeKey) {
        runtimeNoticeRef.current = runtimeNoticeKey;
        const cacheLabel =
          result.loadMeta.fileCount > 0
            ? `${result.loadMeta.fromCacheCount}/${result.loadMeta.fileCount} 命中缓存`
            : "缓存信息暂不可用";

        toast({
          type: "success",
          description: `本地模型已就绪：${result.device.toUpperCase()} · ${result.dtype} · ${cacheLabel}`,
        });
      }

      if (controller.signal.aborted) {
        setStatus("ready");
        return;
      }

      const assistantId = generateUUID();
      const createdAt = new Date().toISOString();
      const chunks = splitTextForStreaming(result.text);

      setStatus("streaming");
      setMessages([
        ...baseMessages,
        {
          id: assistantId,
          role: "assistant",
          parts: [{ type: "text", text: "" }],
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
            parts: [{ type: "text", text: streamedText }],
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
        parts: [{ type: "text", text: result.text }],
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
