export const DEFAULT_CHAT_MODEL = "onnx-community/Qwen3-0.6B-ONNX";

export const titleModel = {
  id: "onnx-community/Qwen2.5-0.5B-Instruct",
  name: "Qwen2.5 0.5B Instruct",
  provider: "onnx-community",
  description: "Local Qwen model for title generation",
};

export type ModelCapabilities = {
  tools: boolean;
  vision: boolean;
  reasoning: boolean;
};

export type ChatModel = {
  id: string;
  name: string;
  provider: string;
  description: string;
  estimatedSizeBytes: number;
  gatewayOrder?: string[];
  reasoningEffort?: "none" | "minimal" | "low" | "medium" | "high";
};

export const chatModels: ChatModel[] = [
  {
    id: "onnx-community/Qwen3-0.6B-ONNX",
    name: "Qwen3 0.6B",
    provider: "onnx-community",
    description: "Default local Qwen model with balanced speed and quality",
    estimatedSizeBytes: Math.round(620 * 1024 * 1024),
  },
  {
    id: "onnx-community/Qwen2.5-0.5B-Instruct",
    name: "Qwen2.5 0.5B Instruct",
    provider: "onnx-community",
    description: "Stable local Qwen model for broad compatibility",
    estimatedSizeBytes: Math.round(483 * 1024 * 1024),
  },
  {
    id: "onnx-community/Qwen3.5-0.8B-ONNX",
    name: "Qwen3.5 0.8B",
    provider: "onnx-community",
    description: "Higher quality Qwen option with larger runtime footprint",
    estimatedSizeBytes: Math.round(1035 * 1024 * 1024),
  },
];

export const localModelCapabilities: Record<string, ModelCapabilities> =
  Object.fromEntries(
    chatModels.map((model) => [
      model.id,
      {
        tools: false,
        vision: false,
        reasoning: false,
      },
    ])
  );

export async function getCapabilities(): Promise<
  Record<string, ModelCapabilities>
> {
  return localModelCapabilities;
}

export const isDemo = process.env.IS_DEMO === "1";

type GatewayModel = {
  id: string;
  name: string;
  type?: string;
  tags?: string[];
};

export type GatewayModelWithCapabilities = ChatModel & {
  capabilities: ModelCapabilities;
};

export async function getAllGatewayModels(): Promise<
  GatewayModelWithCapabilities[]
> {
  return chatModels.map((model) => ({
    ...model,
    capabilities: localModelCapabilities[model.id],
  }));
}

export function getActiveModels(): ChatModel[] {
  return chatModels;
}

export const allowedModelIds = new Set(chatModels.map((m) => m.id));

export const modelsByProvider = chatModels.reduce(
  (acc, model) => {
    if (!acc[model.provider]) {
      acc[model.provider] = [];
    }
    acc[model.provider].push(model);
    return acc;
  },
  {} as Record<string, ChatModel[]>
);
