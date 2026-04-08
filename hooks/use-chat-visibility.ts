"use client";

import { useEffect, useState } from "react";
import {
  getLocalChatById,
  setLocalChatVisibility,
} from "@/lib/local-chat-store";
import type { VisibilityType } from "@/components/chat/visibility-selector";

export function useChatVisibility({
  chatId,
  initialVisibilityType,
}: {
  chatId: string;
  initialVisibilityType: VisibilityType;
}) {
  const [visibilityType, setVisibilityTypeState] =
    useState<VisibilityType>(initialVisibilityType);

  useEffect(() => {
    let isMounted = true;

    const load = async () => {
      const chat = await getLocalChatById(chatId);
      if (isMounted && chat) {
        setVisibilityTypeState(chat.visibility);
      }
    };

    void load();

    return () => {
      isMounted = false;
    };
  }, [chatId]);

  const setVisibilityType = (updatedVisibilityType: VisibilityType) => {
    setVisibilityTypeState(updatedVisibilityType);
    void setLocalChatVisibility(chatId, updatedVisibilityType);
  };

  return { visibilityType, setVisibilityType };
}
