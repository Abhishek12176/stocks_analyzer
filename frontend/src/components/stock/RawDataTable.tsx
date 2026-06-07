"use client";

import { useState, useMemo } from "react";
import { Skeleton } from "@/components/ui/Skeleton";
import { cn } from "@/lib/cn";
import type { StockPrice } from "@/types/stock";

interface RawDataTableProps {
  history?: StockPrice[] | null;
  loading?: boolean;
}

function formatDisplayDate(dateStr: string): string {
  const d = new Date(dateStr);
  return d.toLocaleDateString("en-IN", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
}

function toInputDate(dateStr: string): string {
  return dateStr.substring(0, 10);
}

export function RawDataTable({ history, loading }: RawDataTableProps) {
  const [selectedDate, setSelectedDate] = useState<string>("");
  const [searchQuery, setSearchQuery] = useState("");

  const filteredHistory = useMemo(() => {
    if (!history) return [];
    if (!searchQuery) return history;
    const q = searchQuery.toLowerCase();
    return history.filter(
      (p) =>
        formatDisplayDate(p.date).toLowerCase().includes(q) ||
        toInputDate(p.date).includes(q)
    );
  }, [history, searchQuery]);

  const selectedPrice = useMemo(() => {
    if (!selectedDate || !history) return null;
    return history.find((p) => toInputDate(p.date) === selectedDate) ?? null;
  }, [selectedDate, history]);

  if (loading) {
    return (
      <div className="space-y-5">
        <Skeleton className="h-10 w-full max-w-sm rounded-xl" />
        <div className="grid grid-cols-5 gap-4">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-24 rounded-2xl" />
          ))}
        </div>
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton key={i} className="h-12 rounded-xl" />
        ))}
      </div>
    );
  }

  if (!history || history.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-sm text-neutral-500">
        No historical price data available
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
        <div className="relative flex-1 max-w-xs">
          <input
            type="date"
            value={selectedDate}
            onChange={(e) => {
              setSelectedDate(e.target.value);
              setSearchQuery("");
            }}
            className={cn(
              "w-full rounded-xl border bg-[rgba(15,23,42,0.6)] backdrop-blur-sm px-4 py-2.5 text-sm text-neutral-200 font-mono",
              "border-neutral-800 focus:border-accent-500/50 focus:outline-none focus:ring-1 focus:ring-accent-500/20",
              "transition-all duration-200"
            )}
          />
          {selectedDate && (
            <button
              onClick={() => setSelectedDate("")}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-neutral-600 hover:text-neutral-400 transition-colors"
            >
              <svg className="size-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>
        <span className="text-xs text-neutral-600">or</span>
        <input
          type="text"
          placeholder="Search date (e.g. May 2025)..."
          value={searchQuery}
          onChange={(e) => {
            setSearchQuery(e.target.value);
            setSelectedDate("");
          }}
          className={cn(
            "flex-1 max-w-xs rounded-xl border bg-[rgba(15,23,42,0.6)] backdrop-blur-sm px-4 py-2.5 text-sm text-neutral-200",
            "border-neutral-800 focus:border-accent-500/50 focus:outline-none focus:ring-1 focus:ring-accent-500/20 placeholder:text-neutral-600",
            "transition-all duration-200"
          )}
        />
      </div>

      {selectedPrice && (
        <div className="rounded-2xl border border-accent-500/20 bg-[rgba(15,23,42,0.6)] backdrop-blur-sm p-5 shadow-[0_0_30px_rgba(59,130,246,0.08)]">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <svg className="size-4 text-accent-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              <span className="text-xs font-semibold text-neutral-400 uppercase tracking-[1px]">
                Selected Date Snapshot
              </span>
            </div>
            <span className="font-mono text-sm font-bold text-accent-400">
              {formatDisplayDate(selectedPrice.date)}
            </span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
            {[
              { label: "Open", value: selectedPrice.open, color: "text-neutral-200" },
              { label: "High", value: selectedPrice.high, color: "text-signal-bullish" },
              { label: "Low", value: selectedPrice.low, color: "text-signal-bearish" },
              { label: "Close", value: selectedPrice.close, color: "text-neutral-50" },
              { label: "Volume", value: selectedPrice.volume, color: "text-accent-400" },
            ].map((item) => (
              <div
                key={item.label}
                className="rounded-xl bg-white/[0.03] border border-neutral-800/60 px-4 py-3"
              >
                <p className="text-[10px] font-semibold text-neutral-500 uppercase tracking-[0.5px]">
                  {item.label}
                </p>
                <p className={cn("font-mono text-lg font-bold mt-1 tabular-nums", item.color)}>
                  {item.label === "Volume"
                    ? item.value.toLocaleString("en-IN")
                    : `₹${item.value.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      <div>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-[13px] font-semibold text-neutral-400 uppercase tracking-[1px]">
            Full Historical Data
          </h3>
          <span className="text-xs text-neutral-600 font-mono">
            {filteredHistory.length} {filteredHistory.length === 1 ? "record" : "records"}
          </span>
        </div>
        <div className="rounded-2xl border border-neutral-800 bg-[rgba(15,23,42,0.6)] overflow-hidden">
          <div className="grid grid-cols-[1fr_100px_100px_100px_100px_120px] gap-2 px-5 py-3.5 bg-white/[0.02] border-b border-neutral-800">
            <span className="text-[11px] font-semibold text-neutral-500 uppercase tracking-[0.5px]">Date</span>
            <span className="text-[11px] font-semibold text-neutral-500 uppercase tracking-[0.5px] text-right">Open</span>
            <span className="text-[11px] font-semibold text-neutral-500 uppercase tracking-[0.5px] text-right">High</span>
            <span className="text-[11px] font-semibold text-neutral-500 uppercase tracking-[0.5px] text-right">Low</span>
            <span className="text-[11px] font-semibold text-neutral-500 uppercase tracking-[0.5px] text-right">Close</span>
            <span className="text-[11px] font-semibold text-neutral-500 uppercase tracking-[0.5px] text-right">Volume</span>
          </div>
          <div className="max-h-[480px] overflow-y-auto scrollbar-thin">
            {filteredHistory.map((price, i) => (
              <div
                key={price.date}
                onClick={() => {
                  setSelectedDate(toInputDate(price.date));
                  setSearchQuery("");
                }}
                className={cn(
                  "grid grid-cols-[1fr_100px_100px_100px_100px_120px] gap-2 px-5 py-3 items-center transition-colors cursor-pointer",
                  "hover:bg-white/[0.02]",
                  i < filteredHistory.length - 1 && "border-b border-neutral-800/50",
                  selectedDate === toInputDate(price.date) && "bg-accent-500/5 border-accent-500/20"
                )}
              >
                <span className="text-sm text-neutral-300 font-medium">
                  {formatDisplayDate(price.date)}
                </span>
                <span className="font-mono text-sm text-right text-neutral-300 tabular-nums">
                  {price.open.toFixed(2)}
                </span>
                <span className="font-mono text-sm text-right text-signal-bullish tabular-nums">
                  {price.high.toFixed(2)}
                </span>
                <span className="font-mono text-sm text-right text-signal-bearish tabular-nums">
                  {price.low.toFixed(2)}
                </span>
                <span className="font-mono text-sm text-right text-neutral-50 font-semibold tabular-nums">
                  {price.close.toFixed(2)}
                </span>
                <span className="font-mono text-sm text-right text-neutral-400 tabular-nums">
                  {price.volume.toLocaleString("en-IN")}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
