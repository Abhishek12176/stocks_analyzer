"use client";

import { useMemo } from "react";
import { MetricCard } from "@/components/ui/MetricCard";
import { Badge } from "@/components/ui/Badge";
import type { TechnicalIndicators } from "@/types/stock";

interface MetricGridProps {
  indicators?: TechnicalIndicators | null;
  loading?: boolean;
}

export function MetricGrid({ indicators, loading }: MetricGridProps) {
  const rsiStatus = useMemo(() => {
    if (!indicators?.rsi) return { label: "", color: "var(--color-text-muted)", trend: "neutral" as const };
    if (indicators.rsi >= 70) return { label: "Overbought", color: "var(--color-signal-neutral)", trend: "down" as const };
    if (indicators.rsi <= 30) return { label: "Oversold", color: "var(--color-signal-bullish)", trend: "up" as const };
    return { label: "Neutral", color: "var(--color-accent-500)", trend: "neutral" as const };
  }, [indicators?.rsi]);

  const macdStatus = useMemo(() => {
    if (!indicators?.macd) return { label: "", trend: "neutral" as const };
    return indicators.macd > 0
      ? { label: "Bullish", trend: "up" as const }
      : { label: "Bearish", trend: "down" as const };
  }, [indicators?.macd]);

  const sma20Trend = useMemo(() => {
    if (!indicators?.sma20 || !indicators?.sma50) return "neutral" as const;
    return indicators.sma20 > indicators.sma50 ? ("up" as const) : ("down" as const);
  }, [indicators?.sma20, indicators?.sma50]);

  if (loading) {
    return (
      <div className="px-8 py-6">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <MetricCard key={i} label="" value="" loading />
          ))}
        </div>
      </div>
    );
  }

  if (!indicators) return null;

  return (
    <div className="px-8 py-6">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <MetricCard
          label="RSI (14) · Relative Strength Index"
          value={indicators.rsi?.toFixed(1) ?? "N/A"}
          statusLabel={rsiStatus.label}
          statusColor={rsiStatus.color}
          trend={rsiStatus.trend}
        />
        <MetricCard
          label="MACD · Moving Average Convergence Divergence"
          value={indicators.macd?.toFixed(2) ?? "N/A"}
          statusLabel={macdStatus.label}
          statusColor={macdStatus.trend === "up" ? "var(--color-signal-bullish)" : "var(--color-signal-bearish)"}
          trend={macdStatus.trend}
        />
        <MetricCard
          label="SMA (20) · Simple Moving Average"
          value={indicators.sma20?.toFixed(2) ?? "N/A"}
          statusLabel={indicators.sma20 && indicators.sma50 ? (indicators.sma20 > indicators.sma50 ? "Above 50" : "Below 50") : undefined}
          statusColor={sma20Trend === "up" ? "var(--color-signal-bullish)" : "var(--color-signal-bearish)"}
          trend={sma20Trend}
        />
        <MetricCard
          label="SMA (50) · Simple Moving Average"
          value={indicators.sma50?.toFixed(2) ?? "N/A"}
          badge={
            indicators.sma20 && indicators.sma50 ? (
              <Badge
                variant={indicators.sma20 > indicators.sma50 ? "bullish" : "bearish"}
              >
                {indicators.sma20 > indicators.sma50 ? "Golden Cross" : "Death Cross"}
              </Badge>
            ) : undefined
          }
        />
      </div>
    </div>
  );
}
