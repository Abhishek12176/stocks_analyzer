"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { useQuery } from "@tanstack/react-query";
import { useWatchlistStore } from "@/store/watchlistStore";
import { useHistoryStore } from "@/store/historyStore";
import { useUiStore } from "@/store/uiStore";
import { useDebounce } from "@/hooks/useDebounce";
import { apiGet } from "@/lib/api";
import { cn } from "@/lib/cn";
import type { SignalResponse } from "@/types/signal";
import type { PriceResponse } from "@/types/stock";
import type { SearchResponse } from "@/types/api";

const POPULAR_STOCKS = [
  { symbol: "RELIANCE", name: "Reliance Industries Ltd" },
  { symbol: "TCS", name: "Tata Consultancy Services" },
  { symbol: "HDFCBANK", name: "HDFC Bank Ltd" },
  { symbol: "INFY", name: "Infosys Ltd" },
  { symbol: "ICICIBANK", name: "ICICI Bank Ltd" },
  { symbol: "ITC", name: "ITC Ltd" },
];

function SignalBadge({ action }: { action: string | null }) {
  if (!action) return null;
  const colors: Record<string, string> = {
    BUY: "bg-signal-bullish/15 text-signal-bullish border-signal-bullish/30",
    SELL: "bg-signal-bearish/15 text-signal-bearish border-signal-bearish/30",
    HOLD: "bg-signal-neutral/15 text-signal-neutral border-signal-neutral/30",
    NEUTRAL: "bg-neutral-400/15 text-[var(--color-text-secondary)] border-neutral-400/30",
  };
  return (
    <span className={cn("text-[11px] font-bold px-2 py-0.5 rounded-md border", colors[action] || "")}>
      {action}
    </span>
  );
}

function WatchlistCard({ symbol, name }: { symbol: string; name: string }) {
  const { data: priceData } = useQuery<PriceResponse>({
    queryKey: ["stock", "price", symbol, "1mo"],
    queryFn: () => apiGet<PriceResponse>(`/stock/${encodeURIComponent(symbol)}/price`, { period: "1mo" }),
    staleTime: 60_000,
    refetchInterval: 60_000,
  });

  const { data: signalData } = useQuery<SignalResponse>({
    queryKey: ["stock", "signal", symbol],
    queryFn: () => apiGet<SignalResponse>(`/stock/${encodeURIComponent(symbol)}/signal`),
    staleTime: 60_000,
    refetchInterval: 60_000,
  });

  const quote = priceData?.quote;
  const signal = signalData?.signal;

  return (
    <Link href={`/stock/${symbol}`}>
      <motion.div
        whileHover={{ scale: 1.02, borderColor: "rgba(59,130,246,0.3)" }}
        className="rounded-2xl border border-neutral-800 bg-neutral-900/80 backdrop-blur-xl p-4 transition-all duration-200 hover:shadow-[0_20px_60px_rgba(0,0,0,0.35)]"
      >
        <div className="flex items-center justify-between mb-2">
          <span className="font-mono text-base font-bold text-[var(--color-text-primary)]">{symbol}</span>
          {signal && <SignalBadge action={signal.action} />}
        </div>
        <p className="text-sm font-medium text-[var(--color-text-secondary)] truncate mb-3">{name}</p>
        {quote ? (
          <div className="flex items-baseline justify-between">
            <span className="text-xl font-bold text-[var(--color-text-primary)] font-mono">
              ₹{quote.currentPrice.toFixed(2)}
            </span>
            <span className={cn(
              "text-sm font-semibold",
              quote.changePercent >= 0 ? "text-signal-bullish" : "text-signal-bearish"
            )}>
              {quote.changePercent >= 0 ? "▲" : "▼"} {Math.abs(quote.changePercent).toFixed(2)}%
            </span>
          </div>
        ) : (
          <div className="h-6 w-24 rounded bg-neutral-800 animate-pulse" />
        )}
      </motion.div>
    </Link>
  );
}

function SearchBar() {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const debouncedQuery = useDebounce(query, 300);
  const [results, setResults] = useState<{ symbol: string; name: string; exchange: string }[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!debouncedQuery || debouncedQuery.length < 1) {
      setResults([]);
      return;
    }
    setLoading(true);
    apiGet<SearchResponse>("/stock/search", { q: debouncedQuery })
      .then((data) => { setResults(data.results); setActiveIndex(-1); })
      .catch(() => setResults([]))
      .finally(() => setLoading(false));
  }, [debouncedQuery]);

  const handleSelect = (symbol: string) => {
    setQuery("");
    setIsOpen(false);
    router.push(`/stock/${symbol}`);
  };

  return (
    <div className="relative w-full max-w-2xl mx-auto">
      <div className="relative group">
        <svg className="absolute left-4 top-1/2 -translate-y-1/2 size-5 text-neutral-500 group-focus-within:text-accent-500 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <circle cx="11" cy="11" r="8" /><path d="M21 21l-4.35-4.35" />
        </svg>
        <input
          type="text"
          value={query}
          onChange={(e) => { setQuery(e.target.value); setIsOpen(true); }}
          onFocus={() => { if (query.length >= 1) setIsOpen(true); }}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") { e.preventDefault(); setActiveIndex((i) => Math.min(i + 1, results.length - 1)); }
            if (e.key === "ArrowUp") { e.preventDefault(); setActiveIndex((i) => Math.max(i - 1, 0)); }
            if (e.key === "Enter" && activeIndex >= 0) { handleSelect(results[activeIndex].symbol); }
            if (e.key === "Escape") setIsOpen(false);
          }}
          placeholder="Search any stock by symbol or name..."
          className="w-full h-14 pl-12 pr-5 text-base bg-neutral-900/80 border border-neutral-800 rounded-2xl text-neutral-50 placeholder-neutral-500 focus:outline-none focus:border-accent-500/40 focus:ring-2 focus:ring-accent-500/10 transition-all backdrop-blur-xl shadow-[0_20px_60px_rgba(0,0,0,0.35)]"
        />
        {loading && (
          <span className="absolute right-4 top-1/2 -translate-y-1/2 size-5 border-2 border-accent-500 border-t-transparent rounded-full animate-spin" />
        )}
      </div>

      {isOpen && results.length > 0 && (
        <div className="absolute top-full mt-2 w-full rounded-2xl border border-neutral-800 bg-neutral-900 shadow-2xl z-50 overflow-hidden backdrop-blur-xl">
          {results.map((stock, i) => (
            <button
              key={`${stock.symbol}-${stock.exchange}`}
              onClick={() => handleSelect(stock.symbol)}
              className={cn(
                "flex w-full items-center gap-4 px-4 py-3.5 text-left transition-colors",
                i === activeIndex ? "bg-accent-500/10 text-neutral-50" : "text-neutral-500 hover:bg-white/[0.03]"
              )}
            >
              <span className="font-mono text-sm font-bold text-neutral-50">{stock.symbol}</span>
              <span className="text-sm font-medium text-neutral-300 truncate flex-1">{stock.name}</span>
              <span className="text-[10px] font-semibold text-neutral-600 uppercase px-2 py-0.5 rounded border border-neutral-800">{stock.exchange}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export default function DashboardPage() {
  const watchlistItems = useWatchlistStore((s) => s.items);
  const clearWatchlist = useWatchlistStore((s) => s.clear);
  const historyItems = useHistoryStore((s) => s.items);
  const clearHistory = useHistoryStore((s) => s.clear);
  const openCommandPalette = useUiStore((s) => s.openCommandPalette);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4 }}
      className="p-6 space-y-8"
    >
      {/* Hero Section */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="text-center space-y-4 pt-4 pb-2"
      >
        <div className="flex items-center justify-center gap-3 mb-3">
          <div className="relative">
            <div className="absolute inset-0 size-16 rounded-2xl bg-accent-500/20 blur-xl animate-logo-glow" />
            <img src="/logo.png" alt="AVORA" className="size-16 rounded-2xl relative" />
          </div>
        </div>
        <h1 className="text-3xl font-bold text-neutral-50 tracking-[-1px]">
          AVORA Dashboard
        </h1>
        <p className="text-neutral-500 text-sm">
          Research, analyze, and track Indian stocks with institutional-grade tools
        </p>

      </motion.div>

      {/* Search */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1, duration: 0.4 }}
      >
        <SearchBar />
      </motion.div>

      {/* Watchlist */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2, duration: 0.4 }}
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-neutral-50">Watchlist</h2>
          <div className="flex items-center gap-3">
            {watchlistItems.length > 0 && (
              <>
                <Link href="/watchlist" className="text-sm text-accent-500 hover:text-accent-500/80 transition-colors">
                  View all
                </Link>
                <button onClick={clearWatchlist} className="text-sm text-signal-bearish hover:text-signal-bearish/80 transition-colors">
                  Clear
                </button>
              </>
            )}
          </div>
        </div>
        {watchlistItems.length > 0 ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
            {watchlistItems.map((item) => (
              <WatchlistCard key={item.symbol} symbol={item.symbol} name={item.name} />
            ))}
          </div>
        ) : (
          <div className="rounded-2xl border border-dashed border-neutral-800 p-12 text-center">
            <div className="flex justify-center mb-3">
              <svg className="size-12 text-neutral-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
                <path d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
              </svg>
            </div>
            <h3 className="text-base font-semibold text-[var(--color-text-secondary)] mb-1">No stocks watched yet</h3>
            <p className="text-sm text-[var(--color-text-muted)] mb-4">Search for a stock above and add it to your watchlist</p>
            <div className="flex flex-wrap justify-center gap-2">
              {POPULAR_STOCKS.map((s) => (
                <Link
                  key={s.symbol}
                  href={`/stock/${s.symbol}`}
                  className="px-3 py-1.5 text-xs font-semibold rounded-xl border border-neutral-800 text-neutral-500 hover:bg-neutral-800/50 hover:text-neutral-50 transition-all"
                >
                  {s.symbol}
                </Link>
              ))}
            </div>
          </div>
        )}
      </motion.section>

      {/* Recent History */}
      {historyItems.length > 0 && (
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.4 }}
        >
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-neutral-50">Recent Analyses</h2>
            <div className="flex items-center gap-3">
              <Link href="/history" className="text-sm text-accent-500 hover:text-accent-500/80 transition-colors">
                View all
              </Link>
              <button onClick={clearHistory} className="text-sm text-signal-bearish hover:text-signal-bearish/80 transition-colors">
                Clear
              </button>
            </div>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {historyItems.slice(0, 6).map((item) => (
              <Link key={item.symbol} href={`/stock/${item.symbol}`}>
                <motion.div
                  whileHover={{ scale: 1.01, borderColor: "rgba(59,130,246,0.3)" }}
                  className="flex items-center justify-between rounded-2xl border border-neutral-800 bg-neutral-900/80 backdrop-blur-xl p-4 transition-all duration-200 hover:shadow-[0_20px_60px_rgba(0,0,0,0.35)]"
                >
                  <div className="flex items-center gap-3">
                    <div className="flex size-10 items-center justify-center rounded-xl bg-accent-500/10 border border-accent-500/20">
                      <span className="font-mono text-sm font-bold text-accent-500">{item.symbol.charAt(0)}</span>
                    </div>
                    <div>
                      <p className="font-mono text-sm font-bold text-[var(--color-text-primary)]">{item.symbol}</p>
                      <p className="text-xs text-[var(--color-text-muted)]">
                        {new Date(item.visitedAt).toLocaleDateString("en-IN", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    {item.signal && <SignalBadge action={item.signal} />}
                    {item.price && (
                      <span className="font-mono text-sm font-semibold text-[var(--color-text-primary)]">₹{item.price.toFixed(2)}</span>
                    )}
                    {item.score && (
                      <div className="flex items-center gap-1.5 text-xs text-neutral-500">
                        <span className={cn(
                          "font-semibold",
                          item.score >= 75 ? "text-signal-bullish" : item.score >= 50 ? "text-signal-neutral" : "text-signal-bearish"
                        )}>
                          {item.score}
                        </span>
                        <span>/100</span>
                      </div>
                    )}
                  </div>
                </motion.div>
              </Link>
            ))}
          </div>
        </motion.section>
      )}

      {/* Quick Start - Popular Stocks */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4, duration: 0.4 }}
      >
        <h2 className="text-xl font-semibold text-neutral-50 mb-4">Popular Stocks</h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          {POPULAR_STOCKS.map((stock, i) => (
            <Link key={stock.symbol} href={`/stock/${stock.symbol}`}>
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 + i * 0.05, duration: 0.3 }}
                whileHover={{ scale: 1.03, borderColor: "rgba(59,130,246,0.3)" }}
                className="rounded-2xl border border-neutral-800 bg-neutral-900/80 backdrop-blur-xl p-4 text-center transition-all duration-200 hover:shadow-[0_20px_60px_rgba(0,0,0,0.35)] cursor-pointer"
              >
                <div className="flex justify-center mb-2">
                  <div className="flex size-12 items-center justify-center rounded-xl bg-gradient-to-br from-accent-500/20 to-purple-500/20 border border-accent-500/20">
                    <span className="font-mono text-lg font-bold text-accent-500">{stock.symbol.charAt(0)}</span>
                  </div>
                </div>
                <p className="font-mono text-sm font-bold text-[var(--color-text-primary)]">{stock.symbol}</p>
                <p className="text-xs font-semibold text-[var(--color-text-secondary)] truncate mt-0.5">{stock.name}</p>
              </motion.div>
            </Link>
          ))}
        </div>
      </motion.section>

      <div className="h-4" />
    </motion.div>
  );
}
