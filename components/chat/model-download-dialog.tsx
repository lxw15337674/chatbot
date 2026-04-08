"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import type { ModelLoadProgressState } from "@/hooks/use-active-chat";
import { chatModels } from "@/lib/ai/models";
import { cn } from "@/lib/utils";

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "0 B";
  }

  const units = ["B", "KB", "MB", "GB"];
  const exponent = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1
  );
  const value = bytes / 1024 ** exponent;

  return `${value.toFixed(exponent === 0 ? 0 : 1)} ${units[exponent]}`;
}

function getLoadProgressWidth(
  modelLoadProgress: ModelLoadProgressState
): number {
  if (modelLoadProgress.phase === "downloading") {
    if (typeof modelLoadProgress.progress === "number") {
      return Math.max(6, Math.min(98, modelLoadProgress.progress));
    }
    if (modelLoadProgress.totalBytes > 0) {
      const ratio =
        (modelLoadProgress.loadedBytes / modelLoadProgress.totalBytes) * 100;
      return Math.max(6, Math.min(98, ratio));
    }
    return 40;
  }

  if (modelLoadProgress.phase === "initializing") {
    if (typeof modelLoadProgress.progress === "number") {
      return Math.max(92, Math.min(99, modelLoadProgress.progress));
    }
    return modelLoadProgress.totalBytes > 0 ? 96 : 88;
  }

  if (modelLoadProgress.phase === "ready") {
    return 100;
  }

  return 0;
}

function getProgressToneClass(
  modelLoadProgress: ModelLoadProgressState
): string {
  if (modelLoadProgress.phase === "failed") {
    return "bg-red-500/80";
  }

  if (modelLoadProgress.phase === "cancelled") {
    return "bg-muted-foreground/35";
  }

  return "bg-foreground";
}

function getProgressSummary(modelLoadProgress: ModelLoadProgressState): string {
  const summaryParts: string[] = [];

  if (typeof modelLoadProgress.progress === "number") {
    summaryParts.push(`${Math.round(modelLoadProgress.progress)}%`);
  }

  if (modelLoadProgress.totalBytes > 0) {
    summaryParts.push(
      `${formatBytes(modelLoadProgress.loadedBytes)} / ${formatBytes(modelLoadProgress.totalBytes)}`
    );
  }

  return summaryParts.join(" · ");
}

const phaseLabelMap: Record<ModelLoadProgressState["phase"], string> = {
  downloading: "下载中",
  initializing: "初始化",
  ready: "已就绪",
  failed: "失败",
  cancelled: "已取消",
};

const phaseVariantMap: Record<
  ModelLoadProgressState["phase"],
  "default" | "secondary" | "destructive" | "outline" | "ghost" | "link"
> = {
  downloading: "default",
  initializing: "secondary",
  ready: "secondary",
  failed: "destructive",
  cancelled: "outline",
};

export function ModelDownloadDialog({
  modelLoadProgress,
  onCancel,
}: {
  modelLoadProgress: ModelLoadProgressState | null;
  onCancel?: () => void;
}) {
  if (!modelLoadProgress || modelLoadProgress.phase !== "downloading") {
    return null;
  }

  const modelName =
    chatModels.find((model) => model.id === modelLoadProgress.modelId)?.name ??
    modelLoadProgress.modelId;
  const progressSummary = getProgressSummary(modelLoadProgress);

  return (
    <Dialog open={true}>
      <DialogContent className="max-w-md" showCloseButton={false}>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            模型下载中
            <Badge variant={phaseVariantMap[modelLoadProgress.phase]}>
              {phaseLabelMap[modelLoadProgress.phase]}
            </Badge>
          </DialogTitle>
          <DialogDescription>
            {modelName} · {modelLoadProgress.message}
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-2">
          {progressSummary && (
            <div className="text-xs text-muted-foreground">
              {progressSummary}
            </div>
          )}

          <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted/70">
            <div
              className={cn(
                "h-full rounded-full transition-[width] duration-300 ease-out",
                getProgressToneClass(modelLoadProgress),
                (modelLoadProgress.phase === "downloading" ||
                  modelLoadProgress.phase === "initializing") &&
                  "animate-pulse"
              )}
              style={{
                width: `${getLoadProgressWidth(modelLoadProgress)}%`,
              }}
            />
          </div>
        </div>

        {modelLoadProgress.canCancel && onCancel && (
          <DialogFooter>
            <Button onClick={onCancel} type="button" variant="outline">
              取消下载
            </Button>
          </DialogFooter>
        )}
      </DialogContent>
    </Dialog>
  );
}
