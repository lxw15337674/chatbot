"use client";

import { formatDistanceToNow } from "date-fns";
import { Trash2Icon } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  clearLocalMemories,
  deleteLocalMemory,
  listLocalMemories,
  type LocalMemoryCategory,
  type LocalMemoryRecord,
} from "@/lib/local-chat-store";
import { cn } from "@/lib/utils";
import { Button } from "../ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";

const CATEGORY_OPTIONS: Array<{
  label: string;
  value: LocalMemoryCategory | "all";
}> = [
  { label: "全部", value: "all" },
  { label: "偏好", value: "preference" },
  { label: "事实", value: "fact" },
  { label: "上下文", value: "session-context" },
];

function categoryLabel(category: LocalMemoryCategory): string {
  if (category === "preference") {
    return "偏好";
  }
  if (category === "fact") {
    return "事实";
  }
  return "上下文";
}

export function MemoryManagerDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [selectedCategory, setSelectedCategory] = useState<
    LocalMemoryCategory | "all"
  >("all");
  const [memories, setMemories] = useState<LocalMemoryRecord[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const loadMemories = async () => {
    setIsLoading(true);
    try {
      const rows =
        selectedCategory === "all"
          ? await listLocalMemories()
          : await listLocalMemories(selectedCategory);
      setMemories(rows);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (!open) {
      return;
    }

    void loadMemories();
  }, [open, selectedCategory]);

  const hasMemories = memories.length > 0;

  const headerDescription = useMemo(() => {
    if (selectedCategory === "all") {
      return "管理自动提取的偏好、事实和会话上下文。";
    }

    return `当前仅显示${categoryLabel(selectedCategory)}类记忆。`;
  }, [selectedCategory]);

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-h-[80vh] max-w-2xl overflow-hidden p-0" showCloseButton={true}>
        <DialogHeader className="border-b border-border/50 px-5 py-4">
          <DialogTitle>长期记忆</DialogTitle>
          <DialogDescription>{headerDescription}</DialogDescription>
        </DialogHeader>

        <div className="flex items-center gap-2 border-b border-border/40 px-5 py-3">
          {CATEGORY_OPTIONS.map((option) => (
            <button
              className={cn(
                "rounded-md px-2.5 py-1 text-[12px] transition-colors",
                selectedCategory === option.value
                  ? "bg-muted text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              )}
              key={option.value}
              onClick={() => setSelectedCategory(option.value)}
              type="button"
            >
              {option.label}
            </button>
          ))}
        </div>

        <div className="max-h-[52vh] overflow-y-auto px-5 py-4">
          {isLoading ? (
            <div className="text-[12px] text-muted-foreground">正在加载记忆...</div>
          ) : !hasMemories ? (
            <div className="text-[12px] text-muted-foreground">
              当前没有可展示的记忆。
            </div>
          ) : (
            <div className="space-y-2">
              {memories.map((memory) => (
                <div
                  className="rounded-lg border border-border/40 bg-card/60 px-3 py-2"
                  key={memory.id}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0 space-y-1">
                      <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
                        <span>{categoryLabel(memory.category)}</span>
                        <span>·</span>
                        <span>{memory.key}</span>
                        <span>·</span>
                        <span>
                          {formatDistanceToNow(memory.updatedAt, {
                            addSuffix: true,
                          })}
                        </span>
                      </div>
                      <div className="wrap-break-word text-[13px] leading-relaxed text-foreground/90">
                        {memory.value}
                      </div>
                    </div>

                    <Button
                      className="h-7 w-7 rounded-md"
                      onClick={async () => {
                        await deleteLocalMemory(memory.id);
                        await loadMemories();
                      }}
                      size="icon-xs"
                      variant="ghost"
                    >
                      <Trash2Icon className="size-3.5" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <DialogFooter className="border-t border-border/50 px-5 py-3 sm:justify-between">
          <Button
            onClick={async () => {
              if (selectedCategory === "all") {
                await clearLocalMemories();
              } else {
                await clearLocalMemories(selectedCategory);
              }
              await loadMemories();
            }}
            size="sm"
            variant="destructive"
          >
            清空当前分类
          </Button>
          <Button onClick={() => onOpenChange(false)} size="sm" variant="outline">
            关闭
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
