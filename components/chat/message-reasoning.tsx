"use client";

import { useEffect, useState } from "react";
import {
  Reasoning,
  ReasoningContent,
  ReasoningTrigger,
} from "../ai-elements/reasoning";

type MessageReasoningProps = {
  isLoading: boolean;
  reasoning: string;
};

export function MessageReasoning({
  isLoading,
  reasoning,
}: MessageReasoningProps) {
  const [hasBeenStreaming, setHasBeenStreaming] = useState(isLoading);

  useEffect(() => {
    if (isLoading) {
      setHasBeenStreaming(true);
    }
  }, [isLoading]);

  const statusLabel = isLoading
    ? "思考中..."
    : hasBeenStreaming
      ? "思考已停止"
      : "思考完成";

  return (
    <Reasoning
      className="rounded-lg border border-border/40 bg-card/40 px-3 py-2"
      data-testid="message-reasoning"
      defaultOpen={false}
      isStreaming={isLoading}
    >
      <div className="border-border/40 border-l pl-3">
        <ReasoningTrigger
          className="text-[12px] text-muted-foreground/85"
          getThinkingMessage={() => <span>{statusLabel}</span>}
        />
        <ReasoningContent className="mt-2 text-muted-foreground/80">
          {reasoning}
        </ReasoningContent>
      </div>
    </Reasoning>
  );
}
