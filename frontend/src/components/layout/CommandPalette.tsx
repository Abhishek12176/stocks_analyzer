"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useUiStore } from "@/store/uiStore";
import { useWatchlistStore } from "@/store/watchlistStore";
import { useDebounce } from "@/hooks/useDebounce";
import { cn } from "@/lib/cn";
import { apiGet } from "@/lib/api";
import type { SearchResponse, StockSymbol } from "@/types/api";

export function CommandPalette() {
  const router = useRouter();
  const open = useUiStore((s) => s.commandPaletteOpen);
  const close = useUiStore((s) => s.closeCommandPalette);
  const watchlistItems = useWatchlistStore((s) => s.items);

  const [query, setQuery] = useState("");
  const [results, setResults] = useState<StockSymbol[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const debouncedQuery = useDebounce(query, 300);

  // Keyboard shortcut
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        if (open) close();
        else useUiStore.getState().openCommandPalette();
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [open, close]);

  useEffect(() => {
    if (open) {
      setQuery("");
      setResults([]);
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [open]);

  useEffect(() => {
    if (!debouncedQuery || debouncedQuery.length < 1) {
      setResults([]);
      return;
    }
    setLoading(true);
    apiGet<SearchResponse>("/stock/search", { q: debouncedQuery })
      .then((data) => {
        setResults(data.results);
        setSelectedIndex(0);
      })
      .catch(() => setResults([]))
      .finally(() => setLoading(false));
  }, [debouncedQuery]);

  const navigate = useCallback(
    (symbol: string) => {
      close();
      router.push(`/stock/${symbol}`);
    },
    [router, close]
  );

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex((i) => Math.min(i + 1, items.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter" && items[selectedIndex]) {
      navigate(items[selectedIndex].symbol);
    } else if (e.key === "Escape") {
      close();
    }
  };

  const deduplicated = results.filter(
    (stock, i, arr) => arr.findIndex((s) => s.symbol === stock.symbol) === i
  );
  const items = deduplicated.length > 0
    ? deduplicated
    : watchlistItems.map((w) => ({ symbol: w.symbol, name: w.name, exchange: "NSE" }));

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh]"
      onClick={(e) => {
        if (e.target === e.currentTarget) close();
      }}
    >
      <div className="fixed inset-0 bg-black/60 backdrop-blur-sm" />
      <div className="relative z-10 w-full max-w-xl animate-scale-in">
        <div className="rounded-2xl border border-neutral-800 bg-neutral-900 shadow-2xl overflow-hidden">
          {/* Input */}
          <div className="flex items-center border-b border-neutral-800 px-5">
            <svg className="size-5 text-neutral-500 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input
              ref={inputRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Search stocks or type a command..."
              className="flex-1 bg-transparent px-3 py-4 text-sm text-neutral-100 placeholder-neutral-500 outline-none"
            />
            {loading ? (
              <span className="size-4 border-2 border-accent-500 border-t-transparent rounded-full animate-spin" />
            ) : null}
            <kbd className="ml-2 rounded border border-neutral-700 bg-neutral-800 px-1.5 py-0.5 text-xs text-neutral-500">
              ESC
            </kbd>
          </div>

          {/* Results */}
          <div className="max-h-72 overflow-y-auto p-2">
            {deduplicated.length > 0 ? (
              <div className="space-y-0.5">
                <p className="px-3 py-1.5 text-xs font-medium text-neutral-500 uppercase tracking-wider">
                  Stocks
                </p>
                {deduplicated.map((stock, i) => (
                  <button
                    key={stock.symbol}
                    onClick={() => navigate(stock.symbol)}
                    className={cn(
                      "flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left transition-colors",
                      i === selectedIndex
                        ? "bg-accent-500/10 text-accent-400"
                        : "text-neutral-300 hover:bg-neutral-800"
                    )}
                  >
                    <span className="font-mono text-sm font-semibold">
                      {stock.symbol}
                    </span>
                    <span className="text-sm text-neutral-500 truncate flex-1">
                      {stock.name}
                    </span>
                    <span className="rounded border border-neutral-700 px-1.5 py-0.5 text-xs text-neutral-500">
                      {stock.exchange}
                    </span>
                  </button>
                ))}
              </div>
            ) : query.length === 0 && watchlistItems.length > 0 ? (
              <div className="space-y-0.5">
                <p className="px-3 py-1.5 text-xs font-medium text-neutral-500 uppercase tracking-wider">
                  Watchlist
                </p>
                {watchlistItems.map((stock) => (
                  <button
                    key={stock.symbol}
                    onClick={() => navigate(stock.symbol)}
                    className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-neutral-300 hover:bg-neutral-800 transition-colors"
                  >
                    <span className="font-mono text-sm font-semibold">
                      {stock.symbol}
                    </span>
                    <span className="text-sm text-neutral-500 truncate flex-1">
                      {stock.name}
                    </span>
                  </button>
                ))}
              </div>
            ) : query.length > 0 && deduplicated.length === 0 && !loading ? (
              <div className="py-8 text-center text-sm text-neutral-500">
                No stocks found for &ldquo;{query}&rdquo;
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}
