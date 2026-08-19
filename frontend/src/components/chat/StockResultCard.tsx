"use client";

import Link from "next/link";
import { cn } from "@/lib/cn";
import { formatPrice, formatPercent } from "@/lib/formatters";
import { SparklineChart } from "@/components/chart/SparklineChart";
import type { ChatStock } from "@/types/chat";

const signalConfig = {
  BUY: { color: "var(--color-signal-bullish)", bg: "rgba(0,200,83,0.12)", border: "rgba(0,200,83,0.25)" },
  SELL: { color: "var(--color-signal-bearish)", bg: "rgba(255,82,82,0.12)", border: "rgba(255,82,82,0.25)" },
  HOLD: { color: "var(--color-signal-neutral)", bg: "rgba(255,193,7,0.12)", border: "rgba(255,193,7,0.25)" },
  NEUTRAL: { color: "var(--color-text-secondary)", bg: "rgba(148,163,184,0.12)", border: "rgba(148,163,184,0.25)" },
} as const;

interface StockResultCardProps {
  stock: ChatStock;
  index?: number;
}

export function StockResultCard({ stock, index = 0 }: StockResultCardProps) {
  const sig = signalConfig[stock.signal] ?? signalConfig.NEUTRAL;
  const isUp = stock.changePercent >= 0;
  const sparkColor = stock.confidence >= 60 ? "var(--color-signal-bullish)" : "var(--color-signal-bearish)";

  return (
    <Link
      href={`/stock/${stock.symbol}?exchange=NSE`}
      className="group block rounded-xl border border-neutral-800 bg-neutral-900/70 p-3 transition-all duration-200 hover:border-accent-500/30 hover:bg-neutral-900"
      aria-label={`Open ${stock.symbol} analysis`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="font-mono text-[13px] font-bold text-neutral-50 group-hover:text-accent-400 transition-colors truncate">
            {stock.symbol}
          </p>
          <p className="text-[10px] text-neutral-500">{stock.name}</p>
        </div>
        <span
          className="shrink-0 inline-flex items-center px-1.5 py-0.5 rounded text-[9px] font-bold tracking-wide"
          style={{ color: sig.color, backgroundColor: sig.bg, border: `1px solid ${sig.border}` }}
        >
          {stock.signal}
        </span>
      </div>

      <div className="mt-1.5 flex items-baseline justify-between gap-2">
        <span className="font-mono text-sm font-bold text-neutral-50">
          {formatPrice(stock.currentPrice)}
        </span>
        <span className={cn("font-mono text-[11px] font-semibold", isUp ? "text-signal-bullish" : "text-signal-bearish")}>
          {isUp ? "▲" : "▼"} {formatPercent(stock.changePercent)}
        </span>
      </div>

      <div className="mt-1 flex items-center justify-between gap-2">
        <div className="flex items-center gap-1">
          <span
            className="inline-block size-1.5 rounded-full"
            style={{ backgroundColor: sig.color }}
          />
          <span className="font-mono text-[10px] font-bold text-neutral-300">
            {stock.confidence.toFixed(0)}%
          </span>
        </div>
        <SparklineChart data={stock.sparkline} color={sparkColor} width={56} height={18} />
      </div>

      {stock.prediction20d ? (
        <p className="mt-1.5 text-[10px] leading-snug text-neutral-500 line-clamp-2 border-t border-neutral-800/60 pt-1.5">
          {stock.prediction20d}
        </p>
      ) : null}
    </Link>
  );
}
