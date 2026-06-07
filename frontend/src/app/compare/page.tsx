"use client";

import { useState, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiPost } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { EmptyState } from "@/components/ui/EmptyState";
import { Badge } from "@/components/ui/Badge";
import { Skeleton } from "@/components/ui/Skeleton";
import { RadarChart } from "@/components/chart/RadarChart";
import { cn } from "@/lib/cn";
import { formatRatio } from "@/lib/formatters";
import type { CompareResponse } from "@/types/api";
import { symbolSchema } from "@/lib/validators";

const COMPARABLE_METRICS = [
  { key: "peRatio", label: "P/E" },
  { key: "roe", label: "ROE (%)" },
  { key: "debtEquity", label: "D/E" },
  { key: "opm", label: "OPM (%)" },
  { key: "npm", label: "NPM (%)" },
  { key: "revenueGrowth", label: "Rev Growth (%)" },
  { key: "profitGrowth", label: "Profit Growth (%)" },
  { key: "currentRatio", label: "Current Ratio" },
];

export default function ComparePage() {
  const [symbolInput, setSymbolInput] = useState("");
  const [symbols, setSymbols] = useState<string[]>([]);

  const { data, isLoading, isError, refetch } = useQuery<CompareResponse>({
    queryKey: ["compare", symbols],
    queryFn: () => apiPost<CompareResponse>("/compare", { symbols }),
    enabled: symbols.length >= 2,
    staleTime: 60_000,
    retry: false,
  });

  const addSymbol = useCallback(() => {
    const result = symbolSchema.safeParse(symbolInput);
    if (!result.success) return;
    const clean = result.data;
    if (!symbols.includes(clean) && symbols.length < 5) {
      setSymbols((prev) => [...prev, clean]);
    }
    setSymbolInput("");
  }, [symbolInput, symbols]);

  const removeSymbol = useCallback((sym: string) => {
    setSymbols((prev) => prev.filter((s) => s !== sym));
  }, []);

  const radarData = data?.stocks && data.stocks.length > 0
    ? COMPARABLE_METRICS.map((metric) => ({
        label: metric.label,
        values: data.stocks.map((stock) => ({
          symbol: stock.symbol,
          value: (stock.metrics[metric.key] as number) ?? 0,
        })),
      }))
    : [];

  return (
    <div className="animate-fade-in">
      <div className="border-b border-neutral-800 px-8 py-6">
        <h1 className="text-2xl font-semibold text-neutral-100">Compare</h1>
        <p className="mt-1 text-sm text-neutral-500">
          Compare 2-5 stocks side-by-side
        </p>
      </div>

      <div className="p-8 space-y-8">
        <div className="flex items-end gap-3 flex-wrap">
          <div className="flex-1 min-w-[200px]">
            <Input
              placeholder="Enter symbol (e.g. RELIANCE)"
              value={symbolInput}
              onChange={(e) => setSymbolInput(e.target.value.toUpperCase())}
              onKeyDown={(e) => {
                if (e.key === "Enter") addSymbol();
              }}
            />
          </div>
          <Button
            variant="secondary"
            onClick={addSymbol}
            disabled={symbols.length >= 5}
          >
            Add
          </Button>
          <Button
            variant="primary"
            onClick={() => refetch()}
            disabled={symbols.length < 2}
            loading={isLoading}
          >
            Compare
          </Button>
        </div>

        {symbols.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {symbols.map((sym) => (
              <Badge key={sym} variant="accent" className="gap-2">
                {sym}
                <button
                  onClick={() => removeSymbol(sym)}
                  className="text-accent-300 hover:text-accent-100"
                >
                  ×
                </button>
              </Badge>
            ))}
          </div>
        )}

        {symbols.length < 2 && !data && (
          <EmptyState
            title="Select stocks to compare"
            description="Add 2 to 5 stock symbols and click Compare to see them side-by-side."
          />
        )}

        {isLoading && (
          <div className="space-y-4">
            <Skeleton className="h-64 rounded-xl" />
            <Skeleton className="h-48 rounded-xl" />
          </div>
        )}

        {data?.stocks && data.stocks.length > 0 && (
          <>
            <div className="rounded-xl border border-neutral-800 bg-neutral-900/50 p-6 flex justify-center">
              {radarData.length > 0 && (
                <RadarChart
                  data={radarData}
                  symbols={data.stocks.map((s) => s.symbol)}
                  size={320}
                />
              )}
            </div>

            <div className="overflow-x-auto rounded-xl border border-neutral-800">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-neutral-800 bg-neutral-900/50">
                    <th className="text-left px-5 py-3 text-xs font-medium text-neutral-500">
                      Metric
                    </th>
                    {data.stocks.map((stock) => (
                      <th
                        key={stock.symbol}
                        className="text-right px-5 py-3 font-mono text-xs font-semibold text-neutral-200"
                      >
                        {stock.symbol}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {COMPARABLE_METRICS.map((metric, i) => (
                    <tr
                      key={metric.key}
                      className={cn(
                        "transition-colors hover:bg-neutral-800/30",
                        i < COMPARABLE_METRICS.length - 1 && "border-b border-neutral-800/50"
                      )}
                    >
                      <td className="px-5 py-3 text-neutral-400">{metric.label}</td>
                      {data.stocks.map((stock) => (
                        <td
                          key={stock.symbol}
                          className="px-5 py-3 font-mono text-neutral-200 text-right"
                        >
                          {stock.metrics[metric.key] != null
                            ? formatRatio(stock.metrics[metric.key] as number)
                            : "N/A"}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}

        {isError && (
          <p className="text-sm text-signal-bearish">
            Failed to load comparison data. Check the symbols and try again.
          </p>
        )}
      </div>
    </div>
  );
}
