"use client";

import { useMemo } from "react";
import { cn } from "@/lib/cn";
import { SIGNAL_COLORS } from "@/lib/constants";
import { formatPrice, formatPercent } from "@/lib/formatters";
import { Skeleton } from "@/components/ui/Skeleton";
import { Badge } from "@/components/ui/Badge";
import type { TradeSignal as TradeSignalType } from "@/types/signal";

interface TradeSignalProps {
  signal?: TradeSignalType | null;
  price?: { price: number; change: number; changePercent: number } | null;
  loading?: boolean;
}

export function TradeSignal({ signal, price, loading }: TradeSignalProps) {
  const confidenceLevel = useMemo(() => {
    if (!signal) return "neutral";
    if (signal.confidence >= 70) return "high";
    if (signal.confidence >= 40) return "medium";
    return "low";
  }, [signal]);

  const confidenceColor = useMemo(() => {
    switch (confidenceLevel) {
      case "high": return { text: "text-signal-bullish", bar: "var(--color-signal-bullish)", bg: "rgba(0,200,83,0.08)" };
      case "medium": return { text: "text-signal-neutral", bar: "var(--color-signal-neutral)", bg: "rgba(255,193,7,0.08)" };
      case "low":
      default: return { text: "text-signal-bearish", bar: "var(--color-signal-bearish)", bg: "rgba(255,82,82,0.08)" };
    }
  }, [confidenceLevel]);

  if (loading) {
    return (
      <div className="px-8 py-4">
        <Skeleton className="h-36 rounded-2xl" />
      </div>
    );
  }

  if (!signal) return null;

  const colors = SIGNAL_COLORS[signal.direction];
  const bullishCount = signal.reasons.filter((r) => r.impact === "bullish").length;
  const bearishCount = signal.reasons.filter((r) => r.impact === "bearish").length;
  const totalCount = signal.reasons.length;

  return (
    <div className="px-8 py-4">
      <div
        className={cn(
          "relative overflow-hidden rounded-2xl border p-6 transition-all duration-300",
          "hover:shadow-[0_8px_30px_rgba(0,0,0,0.35)]",
          colors.bg,
          colors.border,
        )}
      >
        <div className="absolute top-0 right-0 w-64 h-64 opacity-5 pointer-events-none">
          <div className="absolute top-[-60px] right-[-60px] w-48 h-48 rounded-full"
            style={{ background: `radial-gradient(circle, ${signal.direction === "bullish" ? "var(--color-signal-bullish)" : signal.direction === "bearish" ? "var(--color-signal-bearish)" : "var(--color-signal-neutral)"}, transparent)` }}
          />
        </div>

        <div className="relative z-10">
          <div className="flex items-start justify-between flex-wrap gap-4 mb-5">
            <div className="flex-shrink-0">
              <div className="flex items-center gap-3 mb-1.5">
                <span className={cn("text-[28px] font-bold font-mono tracking-tight", colors.text)}>
                  {signal.action}
                </span>
                <Badge variant={signal.direction === "bullish" ? "bullish" : signal.direction === "bearish" ? "bearish" : "neutral"}>
                  {signal.direction.toUpperCase()}
                </Badge>
              </div>
              {price ? (
                <div className="flex items-center gap-2.5">
                  <span className="font-mono text-base font-semibold text-neutral-50">
                    {formatPrice(price.price)}
                  </span>
                  <span
                    className={cn(
                      "font-mono text-sm font-medium",
                      price.change >= 0 ? "text-signal-bullish" : "text-signal-bearish"
                    )}
                  >
                    {price.change >= 0 ? "+" : ""}
                    {formatPercent(price.changePercent)}
                  </span>
                  <span
                    className={cn(
                      "font-mono text-xs",
                      price.change >= 0 ? "text-signal-bullish" : "text-signal-bearish"
                    )}
                  >
                    {price.change >= 0 ? "+" : ""}₹{Math.abs(price.change).toFixed(2)}
                  </span>
                </div>
              ) : null}
            </div>

            <div className="flex items-center gap-5">
              <div className="flex flex-col items-center">
                <div
                  className="relative size-[68px] flex-shrink-0"
                >
                  <svg width={68} height={68} viewBox="0 0 68 68" className="transform -rotate-90">
                    <circle cx="34" cy="34" r="29" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="3" />
                    <circle
                      cx="34" cy="34" r="29" fill="none" stroke={confidenceColor.bar} strokeWidth="3" strokeLinecap="round"
                      strokeDasharray={2 * Math.PI * 29}
                      strokeDashoffset={2 * Math.PI * 29 * (1 - (signal.confidence ?? 0) / 100)}
                    />
                  </svg>
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <span className="text-sm font-bold text-neutral-50">{signal.confidence}%</span>
                  </div>
                </div>
                <span className="text-[10px] font-medium text-neutral-500 uppercase tracking-[1px] mt-1">
                  Confidence
                </span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5 mb-5">
            <div className="rounded-xl bg-white/[0.03] border border-white/[0.06] px-3.5 py-2.5">
              <span className="text-[10px] font-medium text-neutral-500 uppercase tracking-[0.5px]">Signal Strength</span>
              <div className="flex items-center gap-2 mt-1">
                <div className="flex-1 h-1.5 rounded-full bg-neutral-800 overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-1000"
                    style={{
                      width: `${signal.confidence}%`,
                      backgroundColor: confidenceColor.bar,
                    }}
                  />
                </div>
                <span className={cn("text-xs font-mono font-semibold", confidenceColor.text)}>
                  {signal.confidence}%
                </span>
              </div>
            </div>
            <div className="rounded-xl bg-white/[0.03] border border-white/[0.06] px-3.5 py-2.5">
              <span className="text-[10px] font-medium text-neutral-500 uppercase tracking-[0.5px]">Bullish Signals</span>
              <div className="flex items-center gap-2 mt-1">
                <span className="text-sm font-bold text-signal-bullish">{bullishCount}/{totalCount}</span>
                <div className="flex-1 h-1.5 rounded-full bg-neutral-800 overflow-hidden">
                  <div className="h-full rounded-full bg-signal-bullish" style={{ width: `${totalCount > 0 ? (bullishCount / totalCount) * 100 : 0}%` }} />
                </div>
              </div>
            </div>
            <div className="rounded-xl bg-white/[0.03] border border-white/[0.06] px-3.5 py-2.5">
              <span className="text-[10px] font-medium text-neutral-500 uppercase tracking-[0.5px]">Bearish Signals</span>
              <div className="flex items-center gap-2 mt-1">
                <span className="text-sm font-bold text-signal-bearish">{bearishCount}/{totalCount}</span>
                <div className="flex-1 h-1.5 rounded-full bg-neutral-800 overflow-hidden">
                  <div className="h-full rounded-full bg-signal-bearish" style={{ width: `${totalCount > 0 ? (bearishCount / totalCount) * 100 : 0}%` }} />
                </div>
              </div>
            </div>
            <div className="rounded-xl bg-white/[0.03] border border-white/[0.06] px-3.5 py-2.5">
              <span className="text-[10px] font-medium text-neutral-500 uppercase tracking-[0.5px]">Generated</span>
              <div className="mt-1">
                <span className="text-xs font-mono text-neutral-400">
                  {new Date(signal.generatedAt).toLocaleString("en-IN", {
                    day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit",
                  })}
                </span>
              </div>
            </div>
          </div>

          <div>
            <span className="text-[11px] font-semibold text-neutral-500 uppercase tracking-[0.8px] mb-2.5 block">
              Technical Analysis
            </span>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {signal.reasons.map((reason, i) => (
                <div
                  key={i}
                  className={cn(
                    "flex items-start gap-2.5 rounded-xl px-3.5 py-2.5 border transition-colors",
                    reason.impact === "bullish"
                      ? "bg-signal-bullish/[0.04] border-signal-bullish/10"
                      : reason.impact === "bearish"
                        ? "bg-signal-bearish/[0.04] border-signal-bearish/10"
                        : "bg-white/[0.02] border-white/[0.06]"
                  )}
                >
                  <span
                    className={cn(
                      "mt-0.5 shrink-0 size-2 rounded-full",
                      reason.impact === "bullish" ? "bg-signal-bullish" : reason.impact === "bearish" ? "bg-signal-bearish" : "bg-signal-neutral"
                    )}
                  />
                  <div className="min-w-0">
                    <span className="text-sm font-semibold text-neutral-200">{reason.factor}: </span>
                    <span className="text-sm text-neutral-500">{reason.detail}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
