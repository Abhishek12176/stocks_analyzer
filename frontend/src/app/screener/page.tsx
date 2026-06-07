"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiPost } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Select } from "@/components/ui/Select";
import { EmptyState } from "@/components/ui/EmptyState";
import { Skeleton } from "@/components/ui/Skeleton";
import { toast } from "@/components/ui/Toast";
import { cn } from "@/lib/cn";
import { formatPercent, formatRatio } from "@/lib/formatters";
import type { ScreenerResult } from "@/types/api";

interface Filters {
  minROE: string;
  maxDE: string;
  minRevenueGrowth: string;
  minProfitGrowth: string;
  minOPM: string;
  maxPE: string;
  sector: string;
}

const SECTORS = [
  { value: "", label: "All Sectors" },
  { value: "Technology", label: "Technology" },
  { value: "Financial Services", label: "Financial Services" },
  { value: "Consumer Cyclical", label: "Consumer Cyclical" },
  { value: "Healthcare", label: "Healthcare" },
  { value: "Industrials", label: "Industrials" },
  { value: "Basic Materials", label: "Basic Materials" },
  { value: "Energy", label: "Energy" },
  { value: "Utilities", label: "Utilities" },
  { value: "Real Estate", label: "Real Estate" },
  { value: "Consumer Defensive", label: "Consumer Defensive" },
  { value: "Communication Services", label: "Communication Services" },
];

export default function ScreenerPage() {
  const [filters, setFilters] = useState<Filters>({
    minROE: "",
    maxDE: "",
    minRevenueGrowth: "",
    minProfitGrowth: "",
    minOPM: "",
    maxPE: "",
    sector: "",
  });
  const [hasSearched, setHasSearched] = useState(false);

  const buildPayload = () => {
    const p: Record<string, number | string> = {};
    if (filters.minROE) p.minROE = parseFloat(filters.minROE);
    if (filters.maxDE) p.maxDE = parseFloat(filters.maxDE);
    if (filters.minRevenueGrowth) p.minRevenueGrowth = parseFloat(filters.minRevenueGrowth);
    if (filters.minProfitGrowth) p.minProfitGrowth = parseFloat(filters.minProfitGrowth);
    if (filters.minOPM) p.minOPM = parseFloat(filters.minOPM);
    if (filters.maxPE) p.maxPE = parseFloat(filters.maxPE);
    if (filters.sector) p.sector = filters.sector;
    return p;
  };

  const shouldFetch = hasSearched && Object.keys(buildPayload()).length > 0;

  const { data, isLoading, isError, refetch } = useQuery<{ results: ScreenerResult[] }>({
    queryKey: ["screener", filters, hasSearched],
    queryFn: async () => {
      const payload = buildPayload();
      return apiPost<{ results: ScreenerResult[] }>("/screener", payload);
    },
    enabled: shouldFetch,
    staleTime: 60_000,
    retry: false,
  });

  const results = data?.results ?? [];

  const handleRun = () => {
    const hasFilters = Object.values(filters).some((v) => v !== "");
    if (!hasFilters) {
      toast("Add at least one filter criterion", "warning");
      return;
    }
    setHasSearched(true);
    refetch();
  };

  const handleClear = () => {
    setFilters({
      minROE: "",
      maxDE: "",
      minRevenueGrowth: "",
      minProfitGrowth: "",
      minOPM: "",
      maxPE: "",
      sector: "",
    });
    setHasSearched(false);
  };

  return (
    <div className="animate-fade-in">
      <div className="border-b border-neutral-800 px-8 py-6">
        <h1 className="text-2xl font-semibold text-neutral-100">Screener</h1>
        <p className="mt-1 text-sm text-neutral-500">
          Filter stocks by financial criteria
        </p>
      </div>

      <div className="p-8">
        <div className="grid grid-cols-1 lg:grid-cols-[300px_1fr] gap-8">
          <div className="space-y-4">
            <div>
              <label className="block text-xs font-medium text-neutral-500 mb-1.5">
                Min ROE (%)
              </label>
              <Input
                type="number"
                placeholder="e.g. 15"
                value={filters.minROE}
                onChange={(e) => setFilters((f) => ({ ...f, minROE: e.target.value }))}
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-neutral-500 mb-1.5">
                Max D/E
              </label>
              <Input
                type="number"
                placeholder="e.g. 0.5"
                step="0.1"
                value={filters.maxDE}
                onChange={(e) => setFilters((f) => ({ ...f, maxDE: e.target.value }))}
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-neutral-500 mb-1.5">
                Min Revenue Growth (%)
              </label>
              <Input
                type="number"
                placeholder="e.g. 10"
                value={filters.minRevenueGrowth}
                onChange={(e) =>
                  setFilters((f) => ({ ...f, minRevenueGrowth: e.target.value }))
                }
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-neutral-500 mb-1.5">
                Min Profit Growth (%)
              </label>
              <Input
                type="number"
                placeholder="e.g. 10"
                value={filters.minProfitGrowth}
                onChange={(e) =>
                  setFilters((f) => ({ ...f, minProfitGrowth: e.target.value }))
                }
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-neutral-500 mb-1.5">
                Min OPM (%)
              </label>
              <Input
                type="number"
                placeholder="e.g. 20"
                value={filters.minOPM}
                onChange={(e) => setFilters((f) => ({ ...f, minOPM: e.target.value }))}
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-neutral-500 mb-1.5">
                Max P/E
              </label>
              <Input
                type="number"
                placeholder="e.g. 30"
                value={filters.maxPE}
                onChange={(e) => setFilters((f) => ({ ...f, maxPE: e.target.value }))}
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-neutral-500 mb-1.5">
                Sector
              </label>
              <Select
                value={filters.sector}
                onChange={(e) => setFilters((f) => ({ ...f, sector: e.target.value }))}
                options={SECTORS}
              />
            </div>
            <div className="flex gap-2 pt-2">
              <Button variant="primary" onClick={handleRun} loading={isLoading} fullWidth>
                Run Screener
              </Button>
              <Button variant="ghost" onClick={handleClear}>
                Clear
              </Button>
            </div>
          </div>

          <div>
            {!hasSearched && (
              <EmptyState
                title="Set your criteria"
                description="Use the filters on the left to find stocks matching your investment criteria."
              />
            )}

            {isLoading && (
              <div className="space-y-3">
                {Array.from({ length: 5 }).map((_, i) => (
                  <Skeleton key={i} className="h-12 rounded-lg" />
                ))}
              </div>
            )}

            {isError && (
              <p className="text-sm text-signal-bearish">
                Failed to run screener. Please try again.
              </p>
            )}

            {!isLoading && hasSearched && results.length === 0 && (
              <EmptyState
                title="No results"
                description="No stocks matched your criteria. Try broadening your filters."
              />
            )}

            {results.length > 0 && (
              <div className="overflow-x-auto rounded-xl border border-neutral-800">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-neutral-800 bg-neutral-900/50">
                      <th className="text-left px-5 py-3 text-xs font-medium text-neutral-500">#</th>
                      <th className="text-left px-5 py-3 text-xs font-medium text-neutral-500">Symbol</th>
                      <th className="text-left px-5 py-3 text-xs font-medium text-neutral-500">Name</th>
                      <th className="text-right px-5 py-3 text-xs font-medium text-neutral-500">Price</th>
                      <th className="text-right px-5 py-3 text-xs font-medium text-neutral-500">P/E</th>
                      <th className="text-right px-5 py-3 text-xs font-medium text-neutral-500">ROE</th>
                      <th className="text-right px-5 py-3 text-xs font-medium text-neutral-500">D/E</th>
                      <th className="text-right px-5 py-3 text-xs font-medium text-neutral-500">Rev Growth</th>
                      <th className="text-right px-5 py-3 text-xs font-medium text-neutral-500">Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((r, i) => (
                      <tr
                        key={r.symbol}
                        className={cn(
                          "transition-colors hover:bg-neutral-800/30",
                          i < results.length - 1 && "border-b border-neutral-800/50"
                        )}
                      >
                        <td className="px-5 py-3 text-xs text-neutral-500">{i + 1}</td>
                        <td className="px-5 py-3 font-mono font-semibold text-neutral-200">
                          {r.symbol}
                        </td>
                        <td className="px-5 py-3 text-neutral-400 truncate max-w-[160px]">
                          {r.name}
                        </td>
                        <td className="px-5 py-3 font-mono text-neutral-200 text-right">
                          {r.price != null ? `₹${r.price.toLocaleString("en-IN")}` : "N/A"}
                        </td>
                        <td className="px-5 py-3 font-mono text-neutral-200 text-right">
                          {formatRatio(r.pe)}
                        </td>
                        <td className="px-5 py-3 font-mono text-signal-bullish text-right">
                          {formatPercent(r.roe)}
                        </td>
                        <td className="px-5 py-3 font-mono text-neutral-200 text-right">
                          {formatRatio(r.de)}
                        </td>
                        <td className="px-5 py-3 font-mono text-neutral-200 text-right">
                          {formatPercent(r.revenueGrowth)}
                        </td>
                        <td className="px-5 py-3 font-mono text-accent-400 text-right">
                          {r.score != null ? r.score.toFixed(1) : "N/A"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
