"use client";

import { use } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { useBaskets } from "@/hooks/useBaskets";
import { cn } from "@/lib/cn";
import { formatPrice, formatPercent } from "@/lib/formatters";
import type { BasketCategory } from "@/types/basket";

const BASKET_IDS: BasketCategory[] = [
  "ai-growth",
  "momentum",
  "strong-buy",
  "banking",
  "it",
  "auto",
  "pharma",
  "energy",
];

const sentimentConfig = {
  bullish: {
    label: "Bullish",
    color: "var(--color-signal-bullish)",
    bg: "rgba(0,200,83,0.1)",
    border: "rgba(0,200,83,0.2)",
    text: "text-signal-bullish",
  },
  bearish: {
    label: "Bearish",
    color: "var(--color-signal-bearish)",
    bg: "rgba(255,82,82,0.1)",
    border: "rgba(255,82,82,0.2)",
    text: "text-signal-bearish",
  },
  neutral: {
    label: "Neutral",
    color: "var(--color-signal-neutral)",
    bg: "rgba(255,193,7,0.1)",
    border: "rgba(255,193,7,0.2)",
    text: "text-signal-neutral",
  },
};

const signalBadge = {
  BUY: { label: "BUY", color: "var(--color-signal-bullish)", bg: "rgba(0,200,83,0.1)" },
  SELL: { label: "SELL", color: "var(--color-signal-bearish)", bg: "rgba(255,82,82,0.1)" },
  HOLD: { label: "HOLD", color: "var(--color-signal-neutral)", bg: "rgba(255,193,7,0.1)" },
  NEUTRAL: {
    label: "NEUTRAL",
    color: "var(--color-text-secondary)",
    bg: "rgba(148,163,184,0.1)",
  },
};

export default function BasketDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);

  if (!BASKET_IDS.includes(id as BasketCategory)) {
    return (
      <div className="mx-auto max-w-7xl px-6 py-8">
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <h3 className="text-base font-semibold text-neutral-300">
            Basket not found
          </h3>
          <Link
            href="/basket"
            className="mt-4 text-sm font-medium text-accent-500 hover:underline"
          >
            ← Back to Baskets
          </Link>
        </div>
      </div>
    );
  }

  return <BasketContent id={id as BasketCategory} />;
}

function BasketContent({ id }: { id: BasketCategory }) {
  const { data: baskets, isLoading, isError, refetch } = useBaskets();
  const basket = baskets?.find((b) => b.id === id);

  if (isLoading) {
    return (
      <div className="mx-auto max-w-7xl px-6 py-8">
        <div className="shimmer h-6 w-32 rounded-lg" />
        <div className="shimmer mt-2 h-8 w-64 rounded-lg" />
        <div className="shimmer mt-1 h-4 w-96 rounded" />

        <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <div
              key={i}
              className="rounded-2xl border border-neutral-800 bg-neutral-900/60 p-5"
            >
              <div className="shimmer h-4 w-32 rounded" />
              <div className="shimmer mt-2 h-3 w-24 rounded" />
              <div className="mt-3 flex items-center gap-3">
                <div className="shimmer h-7 w-20 rounded" />
                <div className="shimmer h-7 w-16 rounded" />
              </div>
              <div className="mt-3 flex gap-4">
                <div className="shimmer h-3 w-12 rounded" />
                <div className="shimmer h-3 w-12 rounded" />
                <div className="shimmer h-3 w-16 rounded" />
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (isError || !basket) {
    return (
      <div className="mx-auto max-w-7xl px-6 py-8">
        <Link
          href="/basket"
          className="inline-flex items-center gap-1 text-sm text-neutral-500 hover:text-neutral-50 transition-colors"
        >
          <svg className="size-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
          </svg>
          Back to Baskets
        </Link>
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <div className="mb-4 size-12 rounded-xl border border-dashed border-neutral-700 flex items-center justify-center text-neutral-600">
            <svg className="size-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
            </svg>
          </div>
          <h3 className="text-base font-semibold text-neutral-300">
            Unable to load basket
          </h3>
          <p className="mt-1 text-sm text-neutral-500">
            Please try again later.
          </p>
          <button
            onClick={() => refetch()}
            className="mt-6 rounded-xl bg-accent-500/10 px-5 py-2.5 text-sm font-medium text-accent-500 border border-accent-500/20 hover:bg-accent-500/20 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  const sent = sentimentConfig[basket.sentiment];
  const trend = basket.averageReturn >= 0 ? "up" : "down";

  return (
    <div className="mx-auto max-w-7xl px-6 py-8">
      <motion.div
        initial={{ opacity: 0, x: -10 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.3 }}
      >
        <Link
          href="/basket"
          className="inline-flex items-center gap-1 text-sm text-neutral-500 hover:text-neutral-50 transition-colors"
        >
          <svg className="size-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
          </svg>
          Back to Baskets
        </Link>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.1 }}
        className="mt-4"
      >
        <div className="flex items-start justify-between flex-wrap gap-4">
          <div>
            <h1 className="text-2xl font-bold text-neutral-50">{basket.name}</h1>
            <p className="mt-1 text-sm text-neutral-500 max-w-xl">
              {basket.description}
            </p>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-right">
              <span className="text-[11px] font-semibold uppercase tracking-[1px] text-neutral-500">
                Sentiment
              </span>
              <div
                className="mt-1 inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold"
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
              </div>
            </div>
            <div className="h-10 w-px bg-neutral-800" />
            <div className="text-right">
              <span className="text-[11px] font-semibold uppercase tracking-[1px] text-neutral-500">
                Avg Return
              </span>
              <p
                className={cn(
                  "mt-0.5 font-mono text-lg font-bold",
                  trend === "up" ? "text-signal-bullish" : "text-signal-bearish"
                )}
              >
                {basket.averageReturn >= 0 ? "+" : ""}
                {basket.averageReturn.toFixed(2)}%
              </p>
            </div>
          </div>
        </div>

        <div className="mt-4 flex items-center gap-4 text-sm text-neutral-500">
          <span>
            <span className="text-neutral-400 font-medium">{basket.totalStocks}</span>{" "}
            stocks
          </span>
          <span className="text-neutral-700">·</span>
          <span>{basket.sector}</span>
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.2 }}
        className="mt-8 grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3"
      >
        {basket.stocks.map((stock, i) => {
          const sig = signalBadge[stock.signal];
          const isUp = stock.changePercent >= 0;
          return (
            <motion.div
              key={stock.symbol}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                duration: 0.35,
                delay: 0.25 + i * 0.05,
                ease: [0.16, 1, 0.3, 1],
              }}
            >
              <Link
                href={`/stock/${stock.symbol}`}
                className={cn(
                  "group block rounded-2xl border border-neutral-800",
                  "bg-neutral-900/60 backdrop-blur-sm p-5",
                  "transition-all duration-300",
                  "hover:border-neutral-700/60 hover:bg-neutral-900/70",
                  "hover:shadow-[0_8px_30px_rgba(0,0,0,0.3)]"
                )}
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="min-w-0 flex-1">
                    <h4 className="text-sm font-semibold text-neutral-50 truncate">
                      {stock.name}
                    </h4>
                    <span className="text-[11px] font-mono text-neutral-500">
                      {stock.symbol}
                    </span>
                  </div>
                  <span
                    className="ml-3 shrink-0 inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold tracking-wide"
                    style={{
                      color: sig.color,
                      backgroundColor: sig.bg,
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
                    {formatPercent(stock.changePercent)}
                  </span>
                </div>

                <div className="flex items-center gap-3 text-[11px]">
                  <div className="flex items-center gap-1.5">
                    <span className="text-neutral-500">RSI</span>
                    <span className="font-mono font-semibold text-neutral-300">
                      {stock.rsi?.toFixed(1) ?? "N/A"}
                    </span>
                  </div>
                  <span className="text-neutral-700">|</span>
                  <div className="flex items-center gap-1.5">
                    <span className="text-neutral-500">MACD</span>
                    <span className="font-mono font-semibold text-neutral-300">
                      {stock.macd?.toFixed(2) ?? "N/A"}
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
        })}
      </motion.div>

      {basket.stocks.length === 0 && (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <div className="mb-4 size-12 rounded-xl border border-dashed border-neutral-700 flex items-center justify-center text-neutral-600">
            <svg className="size-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
            </svg>
          </div>
          <h3 className="text-base font-semibold text-neutral-300">
            No stocks available
          </h3>
          <p className="mt-1 text-sm text-neutral-500">
            Stock data could not be loaded for this basket.
          </p>
        </div>
      )}

      <p className="mt-10 text-center text-[11px] text-neutral-600 leading-relaxed max-w-2xl mx-auto">
        Basket data is generated using technical indicators and is for
        educational purposes only. It is not investment advice.
      </p>
    </div>
  );
}
