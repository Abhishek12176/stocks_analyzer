"use client";

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { useBaskets } from "@/hooks/useBaskets";
import { BasketCard } from "@/components/basket/BasketCard";
import { cn } from "@/lib/cn";
import type { BasketSummary } from "@/types/basket";

function KpiCard({
  label,
  value,
  accent,
}: {
  label: string;
  value: string | number;
  accent?: string;
}) {
  return (
    <div
      className={cn(
        "rounded-2xl border border-neutral-800 bg-neutral-900/60 backdrop-blur-sm p-5",
        "transition-all duration-300"
      )}
    >
      <span className="text-[11px] font-semibold text-neutral-500 uppercase tracking-[1.2px]">
        {label}
      </span>
      <p
        className="mt-1.5 font-mono text-[26px] font-bold tracking-tight"
        style={{ color: accent ?? "var(--color-text-primary)" }}
      >
        {value}
      </p>
    </div>
  );
}

export default function BasketPage() {
  const { data: baskets, isLoading, isError, refetch } = useBaskets();
  const [search, setSearch] = useState("");

  const summary: BasketSummary = useMemo(() => {
    if (!baskets || baskets.length === 0)
      return { totalBaskets: 0, totalStocks: 0, bullishBaskets: 0, bearishBaskets: 0 };
    return {
      totalBaskets: baskets.length,
      totalStocks: baskets.reduce((s, b) => s + b.totalStocks, 0),
      bullishBaskets: baskets.filter((b) => b.sentiment === "bullish").length,
      bearishBaskets: baskets.filter((b) => b.sentiment === "bearish").length,
    };
  }, [baskets]);

  const filteredBaskets = useMemo(() => {
    if (!baskets) return [];
    if (!search.trim()) return baskets;
    const q = search.toLowerCase();
    return baskets.filter(
      (b) =>
        b.name.toLowerCase().includes(q) ||
        b.sector.toLowerCase().includes(q)
    );
  }, [baskets, search]);

  if (isLoading) {
    return (
      <div className="mx-auto max-w-7xl px-6 py-8">
        <div className="shimmer h-8 w-48 rounded-lg" />
        <div className="shimmer mt-2 h-4 w-96 rounded" />

        <div className="mt-8 grid grid-cols-4 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div
              key={i}
              className="rounded-2xl border border-neutral-800 bg-neutral-900/60 p-5"
            >
              <div className="shimmer h-3 w-20 rounded" />
              <div className="shimmer mt-2 h-7 w-24 rounded" />
            </div>
          ))}
        </div>

        <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {Array.from({ length: 8 }).map((_, i) => (
            <div
              key={i}
              className="rounded-2xl border border-neutral-800 bg-neutral-900/60 p-6"
            >
              <div className="shimmer h-5 w-36 rounded" />
              <div className="shimmer mt-2 h-3 w-full rounded" />
              <div className="shimmer mt-1 h-3 w-3/4 rounded" />
              <div className="mt-4 flex gap-4">
                <div className="shimmer h-8 w-16 rounded" />
                <div className="shimmer h-8 w-24 rounded" />
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="mx-auto max-w-7xl px-6 py-8">
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <div className="mb-4 size-12 rounded-xl border border-dashed border-neutral-700 flex items-center justify-center text-neutral-600">
            <svg className="size-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
            </svg>
          </div>
          <h3 className="text-base font-semibold text-neutral-300">
            Unable to load baskets
          </h3>
          <p className="mt-1 text-sm text-neutral-500 max-w-sm">
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

  return (
    <div className="mx-auto max-w-7xl px-6 py-8">
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      >
        <h1 className="text-2xl font-bold text-neutral-50">Stock Baskets</h1>
        <p className="mt-1 text-sm text-neutral-500">
          Curated groups of stocks based on sectors, momentum, trend strength,
          and technical signals.
        </p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.1 }}
        className="mt-8 grid grid-cols-2 gap-4 sm:grid-cols-4"
      >
        <KpiCard label="Total Baskets" value={summary.totalBaskets} accent="var(--color-accent-500)" />
        <KpiCard label="Stocks Covered" value={summary.totalStocks} accent="var(--color-accent-500)" />
        <KpiCard
          label="Bullish Baskets"
          value={summary.bullishBaskets}
          accent="var(--color-signal-bullish)"
        />
        <KpiCard
          label="Bearish Baskets"
          value={summary.bearishBaskets}
          accent="var(--color-signal-bearish)"
        />
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.15 }}
        className="mt-6"
      >
        <div className="relative">
          <svg
            className="absolute left-4 top-1/2 -translate-y-1/2 size-4 text-neutral-500 pointer-events-none"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <circle cx="11" cy="11" r="8" />
            <path strokeLinecap="round" d="m21 21-4.35-4.35" />
          </svg>
          <input
            type="text"
            placeholder="Search baskets by name or sector..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full rounded-2xl border border-neutral-800 bg-neutral-900/60 py-3 pl-11 pr-4 text-sm text-neutral-50 placeholder-neutral-600 backdrop-blur-sm transition-colors focus:border-neutral-700/60 focus:outline-none"
          />
        </div>
      </motion.div>

      {filteredBaskets.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <div className="mb-4 size-12 rounded-xl border border-dashed border-neutral-700 flex items-center justify-center text-neutral-600">
            <svg className="size-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
            </svg>
          </div>
          <h3 className="text-base font-semibold text-neutral-300">
            No baskets available
          </h3>
          <p className="mt-1 text-sm text-neutral-500 max-w-sm">
            {search.trim()
              ? `No baskets matching "${search}"`
              : "No basket data could be loaded. Stock APIs may be unavailable."}
          </p>
        </div>
      ) : (
        <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {filteredBaskets.map((basket, i) => (
            <BasketCard key={basket.id} basket={basket} index={i} />
          ))}
        </div>
      )}

      <p className="mt-10 text-center text-[11px] text-neutral-600 leading-relaxed max-w-2xl mx-auto">
        Basket data is generated using technical indicators and is for
        educational purposes only. It is not investment advice.
      </p>
    </div>
  );
}
