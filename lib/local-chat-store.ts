export type LocalVisibilityType = "private" | "public";

export type LocalChatRecord = {
  id: string;
  title: string;
  createdAt: Date;
  updatedAt: Date;
  visibility: LocalVisibilityType;
  modelId: string;
};

type StoredChatRecord = {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  visibility: LocalVisibilityType;
  modelId: string;
};

type StoredMessageRecord<TMessage> = {
  id: string;
  chatId: string;
  createdAt: number;
  message: TMessage;
};

type StoredSetting = {
  key: string;
  value: string;
};

export type LocalModelCacheMeta = {
  modelId: string;
  dtype: string;
  device: "webgpu" | "wasm";
  bytesTotal: number;
  fileCount: number;
  fromCacheCount: number;
  loadedAt: Date;
};

type StoredModelCacheMeta = {
  modelId: string;
  dtype: string;
  device: "webgpu" | "wasm";
  bytesTotal: number;
  fileCount: number;
  fromCacheCount: number;
  loadedAt: number;
};

export type LocalModelDownloadStatus = "downloading" | "done" | "failed";

export type LocalModelDownloadRecord = {
  id: string;
  modelKey: string;
  modelId: string;
  dtype: string;
  file: string;
  status: LocalModelDownloadStatus;
  loadedBytes: number;
  totalBytes: number;
  updatedAt: Date;
};

export type LocalMemoryCategory = "preference" | "fact" | "session-context";

export type LocalMemoryRecord = {
  id: string;
  category: LocalMemoryCategory;
  key: string;
  value: string;
  sourceChatId: string | null;
  confidence: number;
  updatedAt: Date;
};

type StoredModelDownloadRecord = {
  id: string;
  modelKey: string;
  modelId: string;
  dtype: string;
  file: string;
  status: LocalModelDownloadStatus;
  loadedBytes: number;
  totalBytes: number;
  updatedAt: number;
};

type StoredMemoryRecord = {
  id: string;
  category: LocalMemoryCategory;
  key: string;
  value: string;
  sourceChatId: string | null;
  confidence: number;
  updatedAt: number;
};

const DB_NAME = "chatbot-local";
const DB_VERSION = 4;
const CHATS_STORE = "chats";
const MESSAGES_STORE = "messages";
const SETTINGS_STORE = "settings";
const MODEL_META_STORE = "model_cache_meta";
const MODEL_DOWNLOAD_STORE = "model_download_state";
const MEMORY_STORE = "long_term_memory";
const EVENT_NAME = "local-chat-updated";

function emitChatUpdated() {
  if (typeof window === "undefined") {
    return;
  }
  window.dispatchEvent(new CustomEvent(EVENT_NAME));
}

export function subscribeLocalChatUpdates(handler: () => void) {
  if (typeof window === "undefined") {
    return () => undefined;
  }

  const listener = () => handler();
  window.addEventListener(EVENT_NAME, listener);

  return () => window.removeEventListener(EVENT_NAME, listener);
}

function requestToPromise<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function txDone(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
    transaction.onabort = () => reject(transaction.error);
  });
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === "undefined") {
      reject(new Error("IndexedDB is not available in this environment"));
      return;
    }

    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = () => {
      const db = request.result;

      if (!db.objectStoreNames.contains(CHATS_STORE)) {
        const chats = db.createObjectStore(CHATS_STORE, { keyPath: "id" });
        chats.createIndex("updatedAt", "updatedAt", { unique: false });
      }

      if (!db.objectStoreNames.contains(MESSAGES_STORE)) {
        const messages = db.createObjectStore(MESSAGES_STORE, {
          keyPath: "id",
        });
        messages.createIndex("chatId", "chatId", { unique: false });
        messages.createIndex("chatId_createdAt", ["chatId", "createdAt"], {
          unique: false,
        });
      }

      if (!db.objectStoreNames.contains(SETTINGS_STORE)) {
        db.createObjectStore(SETTINGS_STORE, { keyPath: "key" });
      }

      if (!db.objectStoreNames.contains(MODEL_META_STORE)) {
        const modelMeta = db.createObjectStore(MODEL_META_STORE, {
          keyPath: "modelId",
        });
        modelMeta.createIndex("loadedAt", "loadedAt", { unique: false });
      }

      if (!db.objectStoreNames.contains(MODEL_DOWNLOAD_STORE)) {
        const modelDownloads = db.createObjectStore(MODEL_DOWNLOAD_STORE, {
          keyPath: "id",
        });
        modelDownloads.createIndex("modelKey", "modelKey", {
          unique: false,
        });
        modelDownloads.createIndex("updatedAt", "updatedAt", {
          unique: false,
        });
      }

      if (!db.objectStoreNames.contains(MEMORY_STORE)) {
        const memories = db.createObjectStore(MEMORY_STORE, {
          keyPath: "id",
        });
        memories.createIndex("category", "category", { unique: false });
        memories.createIndex("updatedAt", "updatedAt", { unique: false });
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function toLocalChat(record: StoredChatRecord): LocalChatRecord {
  return {
    ...record,
    createdAt: new Date(record.createdAt),
    updatedAt: new Date(record.updatedAt),
  };
}

export async function listLocalChats(): Promise<LocalChatRecord[]> {
  const db = await openDatabase();
  const tx = db.transaction(CHATS_STORE, "readonly");
  const store = tx.objectStore(CHATS_STORE);
  const rows = (await requestToPromise(store.getAll())) as StoredChatRecord[];
  await txDone(tx);

  return rows
    .map(toLocalChat)
    .sort((a, b) => b.updatedAt.getTime() - a.updatedAt.getTime());
}

export async function getLocalChatById(
  chatId: string
): Promise<LocalChatRecord | null> {
  const db = await openDatabase();
  const tx = db.transaction(CHATS_STORE, "readonly");
  const store = tx.objectStore(CHATS_STORE);
  const row = (await requestToPromise(
    store.get(chatId)
  )) as StoredChatRecord | undefined;
  await txDone(tx);

  return row ? toLocalChat(row) : null;
}

export async function saveLocalChat(params: {
  chatId: string;
  title: string;
  visibility: LocalVisibilityType;
  modelId: string;
}): Promise<void> {
  const db = await openDatabase();
  const existing = await getLocalChatById(params.chatId);
  const now = Date.now();

  const tx = db.transaction(CHATS_STORE, "readwrite");
  const store = tx.objectStore(CHATS_STORE);

  const nextRecord: StoredChatRecord = {
    id: params.chatId,
    title: params.title,
    visibility: params.visibility,
    modelId: params.modelId,
    createdAt: existing?.createdAt.getTime() ?? now,
    updatedAt: now,
  };

  store.put(nextRecord);
  await txDone(tx);
  emitChatUpdated();
}

export async function setLocalChatVisibility(
  chatId: string,
  visibility: LocalVisibilityType
): Promise<void> {
  const existing = await getLocalChatById(chatId);
  if (!existing) {
    return;
  }

  const db = await openDatabase();
  const tx = db.transaction(CHATS_STORE, "readwrite");
  const store = tx.objectStore(CHATS_STORE);

  const nextRecord: StoredChatRecord = {
    id: existing.id,
    title: existing.title,
    visibility,
    modelId: existing.modelId,
    createdAt: existing.createdAt.getTime(),
    updatedAt: Date.now(),
  };

  store.put(nextRecord);
  await txDone(tx);
  emitChatUpdated();
}

export async function replaceLocalMessages<TMessage>(
  chatId: string,
  messages: Array<{ id: string; createdAt: number; message: TMessage }>
): Promise<void> {
  const db = await openDatabase();
  const tx = db.transaction(MESSAGES_STORE, "readwrite");
  const store = tx.objectStore(MESSAGES_STORE);
  const index = store.index("chatId");
  const existingKeys = (await requestToPromise(
    index.getAllKeys(IDBKeyRange.only(chatId))
  )) as IDBValidKey[];

  for (const key of existingKeys) {
    store.delete(key);
  }

  for (const row of messages) {
    const payload: StoredMessageRecord<TMessage> = {
      id: row.id,
      chatId,
      createdAt: row.createdAt,
      message: row.message,
    };
    store.put(payload);
  }

  await txDone(tx);
  emitChatUpdated();
}

export async function getLocalMessages<TMessage>(
  chatId: string
): Promise<Array<{ id: string; createdAt: number; message: TMessage }>> {
  const db = await openDatabase();
  const tx = db.transaction(MESSAGES_STORE, "readonly");
  const store = tx.objectStore(MESSAGES_STORE);
  const index = store.index("chatId");

  const rows = (await requestToPromise(
    index.getAll(IDBKeyRange.only(chatId))
  )) as StoredMessageRecord<TMessage>[];
  await txDone(tx);

  return rows
    .sort((a, b) => a.createdAt - b.createdAt)
    .map((row) => ({
      id: row.id,
      createdAt: row.createdAt,
      message: row.message,
    }));
}

export async function deleteLocalChat(chatId: string): Promise<void> {
  const db = await openDatabase();
  const tx = db.transaction([CHATS_STORE, MESSAGES_STORE], "readwrite");

  tx.objectStore(CHATS_STORE).delete(chatId);

  const messageStore = tx.objectStore(MESSAGES_STORE);
  const index = messageStore.index("chatId");
  const messageKeys = (await requestToPromise(
    index.getAllKeys(IDBKeyRange.only(chatId))
  )) as IDBValidKey[];

  for (const key of messageKeys) {
    messageStore.delete(key);
  }

  await txDone(tx);
  emitChatUpdated();
}

export async function clearLocalChats(): Promise<void> {
  const db = await openDatabase();
  const tx = db.transaction([CHATS_STORE, MESSAGES_STORE], "readwrite");
  tx.objectStore(CHATS_STORE).clear();
  tx.objectStore(MESSAGES_STORE).clear();
  await txDone(tx);
  emitChatUpdated();
}

export async function saveLocalSetting(
  key: string,
  value: string
): Promise<void> {
  const db = await openDatabase();
  const tx = db.transaction(SETTINGS_STORE, "readwrite");
  tx.objectStore(SETTINGS_STORE).put({ key, value } satisfies StoredSetting);
  await txDone(tx);
}

export async function getLocalSetting(key: string): Promise<string | null> {
  const db = await openDatabase();
  const tx = db.transaction(SETTINGS_STORE, "readonly");
  const record = (await requestToPromise(
    tx.objectStore(SETTINGS_STORE).get(key)
  )) as StoredSetting | undefined;
  await txDone(tx);

  return record?.value ?? null;
}

function toLocalMemoryRecord(row: StoredMemoryRecord): LocalMemoryRecord {
  return {
    ...row,
    updatedAt: new Date(row.updatedAt),
  };
}

export async function upsertLocalMemory(params: {
  category: LocalMemoryCategory;
  key: string;
  value: string;
  sourceChatId?: string | null;
  confidence?: number;
  updatedAt?: number;
}): Promise<void> {
  const db = await openDatabase();
  const tx = db.transaction(MEMORY_STORE, "readwrite");
  const store = tx.objectStore(MEMORY_STORE);

  const normalizedKey = params.key.trim().toLowerCase();
  if (normalizedKey.length === 0) {
    await txDone(tx);
    return;
  }

  const row: StoredMemoryRecord = {
    id: `${params.category}:${normalizedKey}`,
    category: params.category,
    key: normalizedKey,
    value: params.value.trim(),
    sourceChatId: params.sourceChatId ?? null,
    confidence:
      typeof params.confidence === "number"
        ? Math.max(0, Math.min(1, params.confidence))
        : 0.8,
    updatedAt: params.updatedAt ?? Date.now(),
  };

  if (!row.value) {
    await txDone(tx);
    return;
  }

  store.put(row);
  await txDone(tx);
}

export async function listLocalMemories(
  category?: LocalMemoryCategory
): Promise<LocalMemoryRecord[]> {
  const db = await openDatabase();
  const tx = db.transaction(MEMORY_STORE, "readonly");
  const store = tx.objectStore(MEMORY_STORE);

  const rows = category
    ? ((await requestToPromise(
        store.index("category").getAll(IDBKeyRange.only(category))
      )) as StoredMemoryRecord[])
    : ((await requestToPromise(store.getAll())) as StoredMemoryRecord[]);

  await txDone(tx);

  return rows
    .map(toLocalMemoryRecord)
    .sort((a, b) => b.updatedAt.getTime() - a.updatedAt.getTime());
}

export async function deleteLocalMemory(memoryId: string): Promise<void> {
  const db = await openDatabase();
  const tx = db.transaction(MEMORY_STORE, "readwrite");
  tx.objectStore(MEMORY_STORE).delete(memoryId);
  await txDone(tx);
}

export async function clearLocalMemories(
  category?: LocalMemoryCategory
): Promise<void> {
  const db = await openDatabase();
  const tx = db.transaction(MEMORY_STORE, "readwrite");
  const store = tx.objectStore(MEMORY_STORE);

  if (!category) {
    store.clear();
    await txDone(tx);
    return;
  }

  const keys = (await requestToPromise(
    store.index("category").getAllKeys(IDBKeyRange.only(category))
  )) as IDBValidKey[];

  for (const key of keys) {
    store.delete(key);
  }

  await txDone(tx);
}

function toLocalModelCacheMeta(
  row: StoredModelCacheMeta
): LocalModelCacheMeta {
  return {
    ...row,
    loadedAt: new Date(row.loadedAt),
  };
}

export async function saveLocalModelCacheMeta(
  payload: Omit<LocalModelCacheMeta, "loadedAt"> & { loadedAt?: number }
): Promise<void> {
  const db = await openDatabase();
  const tx = db.transaction(MODEL_META_STORE, "readwrite");
  const store = tx.objectStore(MODEL_META_STORE);

  const row: StoredModelCacheMeta = {
    ...payload,
    loadedAt: payload.loadedAt ?? Date.now(),
  };

  store.put(row);
  await txDone(tx);
}

export async function getLocalModelCacheMeta(
  modelId: string
): Promise<LocalModelCacheMeta | null> {
  const db = await openDatabase();
  const tx = db.transaction(MODEL_META_STORE, "readonly");
  const store = tx.objectStore(MODEL_META_STORE);
  const row = (await requestToPromise(
    store.get(modelId)
  )) as StoredModelCacheMeta | undefined;
  await txDone(tx);

  return row ? toLocalModelCacheMeta(row) : null;
}

function toLocalModelDownloadRecord(
  row: StoredModelDownloadRecord
): LocalModelDownloadRecord {
  return {
    ...row,
    updatedAt: new Date(row.updatedAt),
  };
}

export async function upsertLocalModelDownloadRecord(params: {
  modelKey: string;
  modelId: string;
  dtype: string;
  file: string;
  status: LocalModelDownloadStatus;
  loadedBytes: number;
  totalBytes: number;
  updatedAt?: number;
}): Promise<void> {
  const db = await openDatabase();
  const tx = db.transaction(MODEL_DOWNLOAD_STORE, "readwrite");
  const store = tx.objectStore(MODEL_DOWNLOAD_STORE);

  const normalizedFile = params.file || "unknown";

  const row: StoredModelDownloadRecord = {
    id: `${params.modelKey}::${normalizedFile}`,
    modelKey: params.modelKey,
    modelId: params.modelId,
    dtype: params.dtype,
    file: normalizedFile,
    status: params.status,
    loadedBytes: Math.max(0, params.loadedBytes),
    totalBytes: Math.max(0, params.totalBytes),
    updatedAt: params.updatedAt ?? Date.now(),
  };

  store.put(row);
  await txDone(tx);
}

export async function listLocalModelDownloadRecords(
  modelKey: string
): Promise<LocalModelDownloadRecord[]> {
  const db = await openDatabase();
  const tx = db.transaction(MODEL_DOWNLOAD_STORE, "readonly");
  const store = tx.objectStore(MODEL_DOWNLOAD_STORE);
  const index = store.index("modelKey");

  const rows = (await requestToPromise(
    index.getAll(IDBKeyRange.only(modelKey))
  )) as StoredModelDownloadRecord[];
  await txDone(tx);

  return rows
    .map(toLocalModelDownloadRecord)
    .sort((a, b) => a.file.localeCompare(b.file));
}

export async function getLocalModelDownloadSummary(modelKey: string): Promise<{
  totalFiles: number;
  completedFiles: number;
  hasPartial: boolean;
}> {
  const rows = await listLocalModelDownloadRecords(modelKey);
  const totalFiles = rows.length;
  const completedFiles = rows.filter((row) => row.status === "done").length;

  return {
    totalFiles,
    completedFiles,
    hasPartial: totalFiles > 0 && completedFiles < totalFiles,
  };
}

export async function clearLocalModelDownloadRecords(
  modelKey: string
): Promise<void> {
  const db = await openDatabase();
  const tx = db.transaction(MODEL_DOWNLOAD_STORE, "readwrite");
  const store = tx.objectStore(MODEL_DOWNLOAD_STORE);
  const index = store.index("modelKey");
  const keys = (await requestToPromise(
    index.getAllKeys(IDBKeyRange.only(modelKey))
  )) as IDBValidKey[];

  for (const key of keys) {
    store.delete(key);
  }

  await txDone(tx);
}

export async function listLocalModelCacheMeta(): Promise<
  LocalModelCacheMeta[]
> {
  const db = await openDatabase();
  const tx = db.transaction(MODEL_META_STORE, "readonly");
  const rows = (await requestToPromise(
    tx.objectStore(MODEL_META_STORE).getAll()
  )) as StoredModelCacheMeta[];
  await txDone(tx);

  return rows
    .map(toLocalModelCacheMeta)
    .sort((a, b) => b.loadedAt.getTime() - a.loadedAt.getTime());
}
