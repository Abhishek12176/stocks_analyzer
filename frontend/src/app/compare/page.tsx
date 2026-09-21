"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Input } from "@/components/ui/Input";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Skeleton } from "@/components/ui/Skeleton";
import { apiPost } from "@/lib/api";
import { formatPrice, formatRatio } from "@/lib/formatters";
import { compareSchema } from "@/lib/validators";
import type { CompareResponse } from "@/types/api";

const METRIC_ROWS: { key: string; label: string; format: "price" | "ratio"; higherBetter?: boolean }[] = [
  { key: "price", label: "Price", format: "price", higherBetter: undefined },
  { key: "pe", label: "P/E Ratio", format: "ratio", higherBetter: undefined },
  { key: "roe", label: "ROE (%)", format: "ratio", higherBetter: true },
  { key: "de", label: "D/E Ratio", format: "ratio", higherBetter: false },
  { key: "opm", label: "Operating Margin (%)", format: "ratio", higherBetter: true },
  { key: "revenue_growth", label: "Revenue Growth", format: "ratio", higherBetter: true },
  { key: "profit_growth", label: "Profit Growth", format: "ratio", higherBetter: true },
  { key: "score", label: "Fundamental Score", format: "ratio", higherBetter: true },
];

const MAX_SLOTS = 5;
const MIN_SLOTS = 2;

function CompareSkeleton() {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-neutral-800">
            <th className="py-3 pr-6 text-left text-neutral-500 font-medium w-48">
              <Skeleton className="h-4 w-28" />
            </th>
            {Array.from({ length: 3 }).map((_, i) => (
              <th key={i} className="py-3 px-6 text-center">
                <Skeleton className="h-4 w-24 mx-auto mb-1" />
                <Skeleton className="h-3 w-16 mx-auto" />
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: 6 }).map((_, row) => (
            <tr key={row} className="border-b border-neutral-800/50">
              <td className="py-3.5 pr-6">
                <Skeleton className="h-4 w-32" />
              </td>
              {Array.from({ length: 3 }).map((_, col) => (
                <td key={col} className="py-3.5 px-6 text-center">
                  <Skeleton className="h-4 w-20 mx-auto" />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function findBestMetric(
  stocks: CompareResponse["stocks"],
  key: string,
  higherBetter: boolean | undefined
): string | null {
  if (higherBetter === undefined) return null;
  const vals = stocks.map((s) => s.metrics[key] as number | null).filter((v) => v != null) as number[];
  if (vals.length < 2) return null;
  const best = higherBetter ? Math.max(...vals) : Math.min(...vals);
  const bestStock = stocks.find((s) => s.metrics[key] === best);
  return bestStock?.symbol ?? null;
}

export default function ComparePage() {
  const [symbols, setSymbols] = useState<string[]>(["", ""]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<CompareResponse | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);

  const handleSymbolChange = (index: number, value: string) => {
    const next = [...symbols];
    next[index] = value.toUpperCase().replace(/\.NS|\.BO/g, "").trim();
    setSymbols(next);
    setError(null);
  };

  const addSlot = () => {
    if (symbols.length < MAX_SLOTS) setSymbols([...symbols, ""]);
  };

  const removeSlot = (index: number) => {
    if (symbols.length <= MIN_SLOTS) return;
    const next = symbols.filter((_, i) => i !== index);
    setSymbols(next);
  };

  const handleCompare = async () => {
    setError(null);
    setFetchError(null);
    setData(null);

    const filled = symbols.filter((s) => s.length > 0);
    const parsed = compareSchema.safeParse({ symbols: filled });
    if (!parsed.success) {
      setError(parsed.error.issues[0].message);
      return;
    }

    setLoading(true);
    try {
      const res = await apiPost<CompareResponse>("/compare/", {
        symbols: parsed.data.symbols,
      });
      if (!res.stocks || res.stocks.length < 2) {
        setFetchError("Could not fetch data for enough stocks. Try different symbols.");
      } else {
        setData(res);
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to compare stocks";
      setFetchError(msg);
    } finally {
      setLoading(false);
    }
  };

  const activeStocks = data?.stocks ?? [];

  return (
    <div className="mx-auto max-w-7xl px-6 py-8">
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      >
        <h1 className="text-2xl font-bold text-neutral-50">Compare Stocks</h1>
        <p className="mt-1.5 text-base text-neutral-400">
          Compare 2&ndash;5 stocks side-by-side on key fundamentals.
        </p>
      </motion.div>

      <div className="mt-6 flex items-end gap-3 flex-wrap">
        {symbols.map((sym, i) => (
          <div key={i} className="flex items-end gap-2">
            <div className="w-36">
              <label className="mb-1 block text-xs font-medium text-neutral-500">
                Stock {i + 1}
              </label>
              <Input
                placeholder="e.g. TCS"
                value={sym}
                onChange={(e) => handleSymbolChange(i, e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleCompare();
                }}
              />
            </div>
            {symbols.length > MIN_SLOTS && (
              <button
                onClick={() => removeSlot(i)}
                className="mb-1 text-xs text-neutral-600 hover:text-signal-bearish transition-colors"
              >
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
        ))}
        {symbols.length < MAX_SLOTS && (
          <button
            onClick={addSlot}
            className="mb-1 flex h-10 w-10 items-center justify-center rounded-lg border border-dashed border-neutral-700 text-neutral-500 hover:border-neutral-500 hover:text-neutral-300 transition-colors"
          >
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
            </svg>
          </button>
        )}
        <Button onClick={handleCompare} loading={loading} className="mb-0.5">
          Compare
        </Button>
      </div>

      {error && (
        <div className="mt-3 text-sm text-signal-bearish">{error}</div>
      )}

      {fetchError && (
        <div className="mt-4 rounded-xl border border-signal-bearish/20 bg-signal-bearish/5 p-4 text-sm text-signal-bearish">
          {fetchError}
        </div>
      )}

      {loading && (
        <div className="mt-8 rounded-xl border border-neutral-800 bg-neutral-900/60 backdrop-blur-sm p-6">
          <CompareSkeleton />
        </div>
      )}

      {!loading && activeStocks.length >= 2 && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="mt-8 rounded-xl border border-neutral-800 bg-neutral-900/60 backdrop-blur-sm overflow-hidden"
        >
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-neutral-800 bg-neutral-900/80">
                  <th className="py-3.5 pr-6 pl-6 text-left text-xs font-semibold text-neutral-500 uppercase tracking-wider">
                    Metric
                  </th>
                  {activeStocks.map((s) => (
                    <th key={s.symbol} className="py-3.5 px-6 text-center">
                      <a
                        href={`/stock/${s.symbol}`}
                        className="text-sm font-semibold text-accent-400 hover:text-accent-300 transition-colors"
                      >
                        {s.symbol}
                      </a>
                      <p className="text-xs text-neutral-500 mt-0.5">{s.name}</p>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {METRIC_ROWS.map((row) => {
                  const best = findBestMetric(activeStocks, row.key, row.higherBetter);
                  return (
                    <tr key={row.key} className="border-b border-neutral-800/40 hover:bg-neutral-800/30 transition-colors">
                      <td className="py-3 pr-6 pl-6 text-sm font-medium text-neutral-400">
                        {row.label}
                      </td>
                      {activeStocks.map((s) => {
                        const val = s.metrics[row.key] as number | null;
                        const formatted = row.format === "price"
                          ? formatPrice(val)
                          : formatRatio(val);
                        const isBest = best === s.symbol;
                        return (
                          <td key={s.symbol} className="py-3 px-6 text-center">
                            <span
                              className={
                                isBest
                                  ? "font-semibold text-signal-bullish"
                                  : val != null
                                    ? "text-neutral-100"
                                    : "text-neutral-600"
                              }
                            >
                              {formatted}
                            </span>
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </motion.div>
      )}

      {!loading && !data && !fetchError && (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-neutral-800/50">
            <svg className="h-8 w-8 text-neutral-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <h2 className="text-xl font-semibold text-neutral-200">Select stocks to compare</h2>
          <p className="mt-2 max-w-md text-sm text-neutral-500">
            Enter 2&ndash;5 NSE stock symbols above and click <strong>Compare</strong> to see
            a side-by-side breakdown of key fundamental metrics.
          </p>
        </div>
      )}

      <p className="mt-10 text-center text-xs text-neutral-500 leading-relaxed max-w-2xl mx-auto">
        Data sourced from Yahoo Finance and Screener.in. For educational purposes only — not investment advice.
      </p>
    </div>
  );
}
