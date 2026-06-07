"use client";

import { useState } from "react";
import { CandlestickChart } from "@/components/chart/CandlestickChart";
import { IntradayChart } from "@/components/chart/IntradayChart";
import { RsiChart } from "@/components/chart/RsiChart";
import { MacdChart } from "@/components/chart/MacdChart";
import { Skeleton } from "@/components/ui/Skeleton";
import { cn } from "@/lib/cn";
import { DATE_RANGES } from "@/lib/constants";
import { useIntradayStock } from "@/hooks/useIntradayStock";
import type { StockPrice, TechnicalIndicators } from "@/types/stock";

interface TechnicalPanelProps {
  symbol?: string;
  history?: StockPrice[] | null;
  indicators?: TechnicalIndicators | null;
  loading?: boolean;
}

export function TechnicalPanel({ symbol, history, indicators, loading }: TechnicalPanelProps) {
  const [dateRange, setDateRange] = useState<(typeof DATE_RANGES)[number]>(DATE_RANGES[3]);
  const [showVolume, setShowVolume] = useState(true);
  const [showSMA, setShowSMA] = useState(true);
  const [viewMode, setViewMode] = useState<"daily" | "intraday">("daily");
  const [intradayInterval, setIntradayInterval] = useState("5m");

  const { data: intradayData, isLoading: intradayLoading } = useIntradayStock(
    viewMode === "intraday" ? symbol ?? "" : "",
    intradayInterval
  );

  if (loading && viewMode === "daily") {
    return (
      <div className="space-y-5">
        <Skeleton className="h-9 w-72" />
        <Skeleton className="h-[440px] rounded-2xl" />
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <Skeleton className="h-[200px] rounded-2xl" />
          <Skeleton className="h-[200px] rounded-2xl" />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold text-neutral-300">
            {viewMode === "daily" ? "Daily Chart" : "Intraday Chart"}
          </h3>
          <div className="flex items-center gap-1 p-0.5 rounded-lg bg-neutral-800/50 border border-neutral-700">
            <button
              onClick={() => setViewMode("daily")}
              className={cn(
                "px-3 py-1.5 text-xs font-medium rounded-md transition-all duration-200",
                viewMode === "daily"
                  ? "bg-neutral-700 text-white shadow-sm"
                  : "text-neutral-500 hover:text-neutral-300"
              )}
            >
              Daily
            </button>
            <button
              onClick={() => setViewMode("intraday")}
              className={cn(
                "px-3 py-1.5 text-xs font-medium rounded-md transition-all duration-200",
                viewMode === "intraday"
                  ? "bg-neutral-700 text-white shadow-sm"
                  : "text-neutral-500 hover:text-neutral-300"
              )}
            >
              Intraday
            </button>
          </div>
        </div>
        {viewMode === "daily" && (
          <>
            <div className="flex items-center gap-1.5">
              {DATE_RANGES.map((range) => (
                <button
                  key={range.label}
                  onClick={() => setDateRange(range)}
                  className={cn(
                    "px-3 py-1.5 text-xs font-semibold rounded-lg transition-all duration-200",
                    dateRange.label === range.label
                      ? "bg-accent-500/10 text-accent-400 border border-accent-500/20"
                      : "text-neutral-500 hover:text-neutral-300 border border-transparent"
                  )}
                >
                  {range.label}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-4">
              <label className="flex items-center gap-2 text-xs text-neutral-500 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={showVolume}
                  onChange={() => setShowVolume((v) => !v)}
                  className="rounded border-neutral-700 bg-neutral-800 accent-accent-500"
                />
                Volume
              </label>
              <label className="flex items-center gap-2 text-xs text-neutral-500 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={showSMA}
                  onChange={() => setShowSMA((v) => !v)}
                  className="rounded border-neutral-700 bg-neutral-800 accent-accent-500"
                />
                SMA
              </label>
            </div>
          </>
        )}
      </div>

      {viewMode === "intraday" ? (
        <IntradayChart
          symbol={symbol}
          companyName={intradayData?.company_name}
          currentPrice={intradayData?.current_price}
          change={intradayData?.change}
          changePercent={intradayData?.change_percent}
          lastUpdated={intradayData?.last_updated}
          data={intradayData?.history ?? []}
          interval={intradayInterval}
          onIntervalChange={setIntradayInterval}
          loading={intradayLoading}
          height={480}
        />
      ) : (
        <>
          {(!history || history.length === 0) && !loading ? (
            <div className="flex items-center justify-center h-64 text-neutral-500 text-sm">
              No technical data available
            </div>
          ) : (
            <>
              <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 backdrop-blur-sm p-5 shadow-[0_8px_30px_rgba(0,0,0,0.25)]">
                <CandlestickChart
                  data={
                    dateRange.days > 0 && history
                      ? history.slice(-dateRange.days)
                      : history ?? []
                  }
                  height={420}
                />
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
                <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 backdrop-blur-sm p-5 shadow-[0_8px_30px_rgba(0,0,0,0.25)]">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="text-xs font-medium text-neutral-400">
                      RSI (14) · Relative Strength Index
                    </h4>
                    {indicators?.rsi && (
                      <span
                        className={cn(
                          "text-xs font-mono font-semibold",
                          indicators.rsi >= 70 ? "text-signal-neutral" : indicators.rsi <= 30 ? "text-signal-bullish" : "text-neutral-400"
                        )}
                      >
                        {indicators.rsi.toFixed(1)}
                      </span>
                    )}
                  </div>
                  <RsiChart history={history} height={180} />
                </div>
                <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 backdrop-blur-sm p-5 shadow-[0_8px_30px_rgba(0,0,0,0.25)]">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="text-xs font-medium text-neutral-400">
                      MACD · Moving Average Convergence Divergence
                    </h4>
                    <span className="flex items-center gap-2">
                      {indicators?.macd && (
                        <span
                          className={cn(
                            "text-xs font-mono font-semibold",
                            (indicators.macd ?? 0) >= 0 ? "text-signal-bullish" : "text-signal-bearish"
                          )}
                        >
                          MACD: {indicators.macd.toFixed(2)}
                        </span>
                      )}
                      {indicators?.signal && (
                          <span className="text-xs font-mono text-signal-neutral">
                          Sig: {indicators.signal.toFixed(2)}
                        </span>
                      )}
                    </span>
                  </div>
                  <MacdChart history={history} height={180} />
                </div>
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
