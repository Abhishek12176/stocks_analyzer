"use client";

import { useState, useRef, useEffect } from "react";
import { motion } from "framer-motion";
import { useSignals } from "@/hooks/useSignals";
import { SignalCard } from "@/components/signals/SignalCard";
import { Tabs } from "@/components/ui/Tabs";
import { Skeleton } from "@/components/ui/Skeleton";
import { cn } from "@/lib/cn";
import type { SignalCategoryId } from "@/types/signals";

function SignalCardSkeleton() {
  return (
    <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 backdrop-blur-sm p-5">
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1 space-y-1.5">
          <Skeleton className="h-4 w-36" />
          <Skeleton className="h-3 w-20" />
        </div>
        <Skeleton className="h-5 w-14 rounded-full" />
      </div>
      <div className="flex items-baseline justify-between mb-3">
        <Skeleton className="h-6 w-20" />
        <Skeleton className="h-4 w-16" />
      </div>
      <div className="flex items-center justify-between mb-3">
        <Skeleton className="h-3 w-24" />
        <Skeleton className="h-6 w-16 rounded" />
      </div>
      <div className="flex items-center gap-3 pt-2 border-t border-neutral-800/60">
        <Skeleton className="h-3 w-16" />
        <Skeleton className="h-3 w-16" />
        <Skeleton className="h-3 w-20" />
      </div>
    </div>
  );
}

export default function SignalsPage() {
  const { categories, totalStocks, isLoading, isError, refetch } = useSignals();
  const [activeTab, setActiveTab] = useState<SignalCategoryId>("strong-buy");
  const scrollRef = useRef<HTMLDivElement>(null);

  const activeCategory = categories.find((c) => c.id === activeTab);
  const displayStocks = activeCategory?.stocks ?? [];

  const tabs = categories.map((c) => ({
    id: c.id,
    label: c.label,
    count: c.count,
  }));

  useEffect(() => {
    if (scrollRef.current) {
      const btn = scrollRef.current.querySelector(
        `[data-tab="${activeTab}"]`
      ) as HTMLElement | null;
      btn?.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
    }
  }, [activeTab]);

  return (
    <div className="mx-auto max-w-7xl px-6 py-8">
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      >
        <h1 className="text-2xl font-bold text-neutral-50">
          Market Signal Center
        </h1>
        <p className="mt-1.5 text-base text-neutral-400">
          Real-time technical signals generated from market analysis.
        </p>
        {!isLoading && !isError && (
          <p className="mt-1.5 text-sm text-neutral-500">
            Scanning <span className="font-mono font-semibold text-neutral-300">{totalStocks}</span> stocks across{" "}
            <span className="font-mono font-semibold text-neutral-300">{categories.length}</span> signal categories
          </p>
        )}
      </motion.div>

      {/* Tabs */}
      <div className="mt-6">
        <div
          ref={scrollRef}
          className="overflow-x-auto scrollbar-hide -mx-6 px-6"
        >
          <div className="min-w-max">
            <Tabs
              tabs={tabs}
              active={activeTab}
              onChange={(id) => setActiveTab(id as SignalCategoryId)}
            />
          </div>
        </div>
      </div>

      {/* Loading */}
      {isLoading && (
        <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
          {Array.from({ length: 8 }).map((_, i) => (
            <SignalCardSkeleton key={i} />
          ))}
        </div>
      )}

      {/* Error */}
      {isError && !isLoading && (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <div className="mb-4 size-12 rounded-xl border border-dashed border-neutral-700 flex items-center justify-center text-neutral-600">
            <svg
              className="size-6"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z"
              />
            </svg>
          </div>
          <h3 className="text-base font-semibold text-neutral-300">
            Unable to load signals
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
      )}

      {/* Data ready */}
      {!isLoading && !isError && (
        <>
          {displayStocks.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-20 text-center">
              <div className="mb-4 size-12 rounded-xl border border-dashed border-neutral-700 flex items-center justify-center text-neutral-600">
                <svg
                  className="size-6"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={1.5}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                  />
                </svg>
              </div>
              <h3 className="text-base font-semibold text-neutral-300">
                No signals available
              </h3>
              <p className="mt-1 text-sm text-neutral-500 max-w-sm">
                No stocks match the criteria for this signal category.
              </p>
            </div>
          ) : (
            <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
              {displayStocks.map((stock, i) => (
                <SignalCard key={stock.symbol} stock={stock} index={i} />
              ))}
            </div>
          )}
        </>
      )}

      <p className="mt-10 text-center text-xs text-neutral-500 leading-relaxed max-w-2xl mx-auto">
        Signals are generated using technical indicators and are for educational
        purposes only. They are not investment advice.
      </p>
    </div>
  );
}
