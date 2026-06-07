"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { SparklineChart } from "@/components/chart/SparklineChart";
import { cn } from "@/lib/cn";
import { formatPrice, formatPercent } from "@/lib/formatters";
import type { SignalStock } from "@/types/signals";

const signalConfig = {
  BUY: { label: "BUY", color: "var(--color-signal-bullish)", bg: "rgba(0,200,83,0.12)", border: "rgba(0,200,83,0.25)" },
  SELL: { label: "SELL", color: "var(--color-signal-bearish)", bg: "rgba(255,82,82,0.12)", border: "rgba(255,82,82,0.25)" },
  HOLD: { label: "HOLD", color: "var(--color-signal-neutral)", bg: "rgba(255,193,7,0.12)", border: "rgba(255,193,7,0.25)" },
  NEUTRAL: { label: "NEUTRAL", color: "var(--color-text-secondary)", bg: "rgba(148,163,184,0.12)", border: "rgba(148,163,184,0.25)" },
};

const confidenceColor = (c: number) => {
  if (c >= 80) return "var(--color-signal-bullish)";
  if (c >= 60) return "var(--color-accent-500)";
  if (c >= 40) return "var(--color-signal-neutral)";
  return "var(--color-signal-bearish)";
};

interface SignalCardProps {
  stock: SignalStock;
  index: number;
}

export function SignalCard({ stock, index }: SignalCardProps) {
  const sig = signalConfig[stock.signal];
  const isUp = stock.changePercent >= 0;
  const sparkColor = stock.confidence >= 60 ? "var(--color-signal-bullish)" : "var(--color-signal-bearish)";

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        duration: 0.35,
        delay: index * 0.035,
        ease: [0.16, 1, 0.3, 1],
      }}
    >
      <Link
        href={`/stock/${stock.symbol}?exchange=NSE`}
        className={cn(
          "group block rounded-2xl border border-neutral-800",
          "bg-neutral-900/60 backdrop-blur-sm p-5",
          "transition-all duration-300",
          "hover:border-neutral-700/60 hover:bg-neutral-900/80",
          "hover:shadow-[0_8px_30px_rgba(0,0,0,0.3)]"
        )}
      >
        <div className="flex items-start justify-between mb-3">
          <div className="min-w-0 flex-1">
            <h4 className="text-sm font-semibold text-neutral-50 truncate leading-snug">
              {stock.name}
            </h4>
            <div className="flex items-center gap-2 mt-0.5">
              <span className="text-[11px] font-mono font-medium text-neutral-500">
                {stock.symbol}
              </span>
              <span className="text-[10px] font-semibold text-neutral-500 px-1.5 py-0.5 rounded border border-neutral-800">
                NSE
              </span>
            </div>
          </div>
          <span
            className="ml-3 shrink-0 inline-flex items-center px-2.5 py-1 rounded-full text-[10px] font-bold tracking-wide"
            style={{
              color: sig.color,
              backgroundColor: sig.bg,
              border: `1px solid ${sig.border}`,
            }}
          >
            {sig.label}
          </span>
        </div>

        <div className="flex items-baseline justify-between mb-3">
          <span className="font-mono text-lg font-bold text-neutral-50">
            {formatPrice(stock.currentPrice)}
          </span>
          <span
            className={cn(
              "font-mono text-sm font-semibold",
              isUp ? "text-signal-bullish" : "text-signal-bearish"
            )}
          >
            {isUp ? "▲" : "▼"} {formatPercent(stock.changePercent)}
          </span>
        </div>

        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-1.5">
            <span
              className="inline-block size-2 rounded-full"
              style={{ backgroundColor: confidenceColor(stock.confidence) }}
            />
            <span className="font-mono text-xs font-bold text-neutral-300">
              {stock.confidence.toFixed(0)}%
            </span>
            <span className="text-[10px] text-neutral-500">Confidence</span>
          </div>
          <SparklineChart
            data={stock.sparkline}
            color={sparkColor}
            width={72}
            height={24}
          />
        </div>

        <div className="flex items-center gap-3 text-[11px] pt-2 border-t border-neutral-800/60">
          <div className="flex items-center gap-1.5">
            <span className="text-neutral-500">RSI</span>
            <span
              className={cn(
                "font-mono font-semibold",
                stock.rsi > 70
                  ? "text-signal-bearish"
                  : stock.rsi < 30
                    ? "text-signal-bullish"
                    : "text-neutral-300"
              )}
            >
              {stock.rsi.toFixed(1)}
            </span>
          </div>
          <span className="text-neutral-700">|</span>
          <div className="flex items-center gap-1.5">
            <span className="text-neutral-500">MACD</span>
            <span
              className={cn(
                "font-mono font-semibold",
                stock.macd > 0 ? "text-signal-bullish" : "text-signal-bearish"
              )}
            >
              {stock.macd.toFixed(2)}
            </span>
          </div>
          <span className="text-neutral-700">|</span>
          <div className="flex items-center gap-1.5">
            <span className="text-neutral-500">Trend</span>
            <span
              className={cn(
                "font-medium",
                stock.trend === "Strong Trend"
                  ? "text-signal-bullish"
                  : "text-neutral-400"
              )}
            >
              {stock.trend === "Strong Trend" ? "Strong ↑" : "Weak"}
            </span>
          </div>
        </div>
      </Link>
    </motion.div>
  );
}
