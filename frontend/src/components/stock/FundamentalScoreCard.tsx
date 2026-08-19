"use client";

import { ProgressRing } from "@/components/ui/ProgressRing";
import { Skeleton } from "@/components/ui/Skeleton";
import { cn } from "@/lib/cn";
import type { FundamentalScore, CategoryScore } from "@/types/fundamentals";

interface FundamentalScoreCardProps {
  score?: FundamentalScore | null;
  loading?: boolean;
}

export function FundamentalScoreCard({ score, loading }: FundamentalScoreCardProps) {
  if (loading) {
    return (
      <div className="flex flex-col items-center py-6">
        <Skeleton className="size-28 rounded-full" />
        <Skeleton className="h-5 w-20 mt-4" />
        <Skeleton className="h-4 w-32 mt-2" />
      </div>
    );
  }

  if (!score) return null;

  return (
    <div className="flex flex-col items-center py-4">
      <ProgressRing value={score.total} size={130} strokeWidth={8} label="Score" />
      <p
        className={cn(
          "mt-3 text-sm font-semibold",
          score.total >= 75 ? "text-signal-bullish" : score.total >= 40 ? "text-signal-neutral" : "text-signal-bearish"
        )}
      >
        {score.rating}
      </p>
      <p className="text-[11px] text-neutral-500 mt-0.5">
        {score.total}/100
      </p>
      {score.categories && score.categories.length > 0 && (
        <div className="mt-6 w-full space-y-3.5">
          {score.categories.map((cat: CategoryScore) => {
            const pct = Math.min((cat.score / Math.max(cat.weight, 1)) * 100, 100);
            const barColor = pct >= 70 ? "var(--color-signal-bullish)" : pct >= 40 ? "var(--color-signal-neutral)" : "var(--color-signal-bearish)";
            return (
              <div key={cat.name}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-neutral-500">{cat.name}</span>
                  <span className="font-mono text-xs text-neutral-400">{cat.score}/{cat.weight}</span>
                </div>
                <div className="h-1.5 rounded-full bg-neutral-800 overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-1000 ease-out"
                    style={{ width: `${pct}%`, backgroundColor: barColor }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
