"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { cn } from "@/lib/cn";
import type { BasketData } from "@/types/basket";

interface BasketCardProps {
  basket: BasketData;
  index: number;
}

const sentimentConfig = {
  bullish: { label: "Bullish", color: "var(--color-signal-bullish)", bg: "rgba(0,200,83,0.1)", border: "rgba(0,200,83,0.2)" },
  bearish: { label: "Bearish", color: "var(--color-signal-bearish)", bg: "rgba(255,82,82,0.1)", border: "rgba(255,82,82,0.2)" },
  neutral: { label: "Neutral", color: "var(--color-signal-neutral)", bg: "rgba(255,193,7,0.1)", border: "rgba(255,193,7,0.2)" },
};

export function BasketCard({ basket, index }: BasketCardProps) {
  const sent = sentimentConfig[basket.sentiment];
  const trend = basket.averageReturn >= 0 ? "up" : "down";

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: index * 0.08, ease: [0.16, 1, 0.3, 1] }}
    >
      <Link
        href={`/basket/${basket.id}`}
        className={cn(
          "group relative block overflow-hidden rounded-2xl border border-neutral-800",
          "bg-neutral-900/60 backdrop-blur-sm p-6",
          "transition-all duration-300",
          "hover:border-neutral-700/60 hover:bg-neutral-900/70",
          "hover:shadow-[0_8px_30px_rgba(0,0,0,0.3)]"
        )}
      >
        <div className="flex items-start justify-between mb-4">
          <div className="flex-1 min-w-0">
            <h3 className="text-base font-semibold text-neutral-50 truncate">
              {basket.name}
            </h3>
            <p className="mt-1 text-sm text-neutral-500 line-clamp-2 leading-relaxed">
              {basket.description}
            </p>
          </div>
          <div className="flex items-center gap-3 ml-4 shrink-0">
            <span
              className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-semibold"
              style={{
                color: sent.color,
                backgroundColor: sent.bg,
                border: `1px solid ${sent.border}`,
              }}
            >
              <span
                className="size-1.5 rounded-full"
                style={{ backgroundColor: sent.color }}
              />
              {sent.label}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-5">
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-semibold uppercase tracking-[1px] text-neutral-500">
              Stocks
            </span>
            <span className="font-mono text-lg font-bold text-neutral-50">
              {basket.totalStocks}
            </span>
          </div>
          <div className="h-8 w-px bg-neutral-800" />
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-semibold uppercase tracking-[1px] text-neutral-500">
              Avg Return
            </span>
            <span
              className={cn(
                "font-mono text-lg font-bold",
                trend === "up" ? "text-signal-bullish" : "text-signal-bearish"
              )}
            >
              {basket.averageReturn >= 0 ? "+" : ""}
              {basket.averageReturn.toFixed(2)}%
            </span>
            <span
              className={cn(
                "text-lg",
                trend === "up" ? "text-signal-bullish" : "text-signal-bearish"
              )}
            >
              {trend === "up" ? "↑" : "↓"}
            </span>
          </div>
          <div className="flex-1" />
          <div className="flex items-center gap-1 text-sm font-medium text-accent-500 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
            <span className="text-accent-500">View Basket</span>
            <svg className="size-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
            </svg>
          </div>
        </div>

        <div
          className="absolute top-0 right-0 size-32 rounded-full blur-[80px] opacity-[0.04] pointer-events-none"
          style={{ backgroundColor: sent.color }}
        />
      </Link>
    </motion.div>
  );
}
