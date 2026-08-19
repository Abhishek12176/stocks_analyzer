"use client";

import { Skeleton } from "@/components/ui/Skeleton";
import { cn } from "@/lib/cn";
import { formatIndianNumber, formatPercent, formatRatio, formatPrice } from "@/lib/formatters";
import type { Fundamentals } from "@/types/fundamentals";

interface FundamentalPanelProps {
  fundamentals?: Fundamentals | null;
  loading?: boolean;
}

function MetricBlock({ label, fullForm, value, className }: { label: string; fullForm?: string; value: string; className?: string }) {
  return (
    <div className={cn("rounded-xl border border-neutral-800 bg-[rgba(15,23,42,0.6)] backdrop-blur-sm p-4", className)}>
      <p className="text-[13px] font-bold text-neutral-200 uppercase tracking-[0.5px]">
        {label}
        {fullForm && <span className="text-neutral-400 lowercase normal-case ml-1 text-[12px] font-medium">({fullForm})</span>}
      </p>
      <p className="font-mono text-xl font-bold text-neutral-50 mt-1.5">{value}</p>
    </div>
  );
}

export function FundamentalPanel({ fundamentals, loading }: FundamentalPanelProps) {
  if (loading) {
    return (
      <div className="space-y-4">
        <div className="grid grid-cols-3 gap-4">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-[72px] rounded-xl" />
          ))}
        </div>
        <div className="grid grid-cols-2 gap-4">
          {Array.from({ length: 2 }).map((_, i) => (
            <Skeleton key={i} className="h-[72px] rounded-xl" />
          ))}
        </div>
        <div className="grid grid-cols-2 gap-4">
          {Array.from({ length: 2 }).map((_, i) => (
            <Skeleton key={i} className="h-[72px] rounded-xl" />
          ))}
        </div>
        <Skeleton className="h-[100px] rounded-xl" />
      </div>
    );
  }

  if (!fundamentals) {
    return (
      <div className="flex items-center justify-center h-48 text-sm text-neutral-500">
        No fundamental data available
      </div>
    );
  }

  const fmtMarketCap = (v: number | null) => formatIndianNumber(v);
  const fmtPercent = (v: number | null) => formatPercent(v);
  const fmtRatio = (v: number | null) => formatRatio(v);
  const fmtPrice = (v: number | null) => formatPrice(v);

  return (
    <div className="space-y-4">
      {/* Row 1: Market Cap | ROE (Return on Equity) | ROCE (Return on Capital Employed) */}
      <div className="grid grid-cols-3 gap-4">
        <MetricBlock label="Market Cap" value={fmtMarketCap(fundamentals.marketCap)} />
        <MetricBlock label="ROE" fullForm="Return on Equity" value={fmtPercent(fundamentals.roe)} />
        <MetricBlock label="ROCE" fullForm="Return on Capital Employed" value={fmtPercent(fundamentals.roce)} />
      </div>

      {/* Row 2: Debt/Equity | OPM (Operating Profit Margin) */}
      <div className="grid grid-cols-2 gap-4">
        <div className="rounded-xl border border-neutral-800 bg-[rgba(15,23,42,0.6)] backdrop-blur-sm p-4">
          <p className="text-[13px] font-bold text-neutral-200 uppercase tracking-[0.5px]">Debt/Equity</p>
          <div className="flex items-center gap-2 mt-1.5">
            <p className="font-mono text-xl font-bold text-neutral-50">{fmtRatio(fundamentals.debtEquity)}</p>
            {fundamentals.deCategory && (
              <span className={cn(
                "text-[11px] font-bold uppercase tracking-wide px-2 py-0.5 rounded",
                fundamentals.deCategory === "Low" && "text-signal-bullish bg-signal-bullish/10",
                fundamentals.deCategory === "Medium" && "text-signal-neutral bg-signal-neutral/10",
                fundamentals.deCategory === "High" && "text-signal-bearish bg-signal-bearish/10",
                fundamentals.deCategory === "N/A" && "text-neutral-500 bg-neutral-800",
              )}>
                {fundamentals.deCategory}
              </span>
            )}
          </div>
        </div>
        <MetricBlock label="OPM" fullForm="Operating Profit Margin" value={fmtPercent(fundamentals.opm)} />
      </div>

      {/* Row 3: P/E Ratio (Price to Earnings Ratio) | EPS (Earnings Per Share) */}
      <div className="grid grid-cols-2 gap-4">
        <MetricBlock label="P/E Ratio" fullForm="Price to Earnings Ratio" value={fmtRatio(fundamentals.peRatio)} />
        <MetricBlock label="EPS" fullForm="Earnings Per Share" value={fmtPrice(fundamentals.eps)} />
      </div>

      {/* Row 4: 📈 Growth (YoY) — Revenue Growth | Profit Growth */}
      <div className="rounded-xl border border-neutral-800 bg-[rgba(15,23,42,0.6)] backdrop-blur-sm p-4">
        <div className="flex items-center gap-1.5 mb-4">
          <span className="text-base">📈</span>
          <span className="text-[13px] font-bold text-neutral-200 uppercase tracking-[1px]">Growth (YoY)</span>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-[13px] font-bold text-neutral-200 uppercase tracking-[0.5px]">Revenue Growth</p>
            <p className="font-mono text-lg font-bold text-neutral-50 mt-1">{fmtPercent(fundamentals.revenueGrowth)}</p>
          </div>
          <div>
            <p className="text-[13px] font-bold text-neutral-200 uppercase tracking-[0.5px]">Profit Growth</p>
            <p className="font-mono text-lg font-bold text-neutral-50 mt-1">{fmtPercent(fundamentals.profitGrowth)}</p>
          </div>
        </div>
      </div>

      {/* Sector badge */}
      {fundamentals.sector && (
        <div className="flex items-center gap-2">
          <span className="text-[13px] font-semibold text-neutral-300">Sector:</span>
          <span className="text-[13px] font-bold text-accent-400 bg-accent-500/10 px-2.5 py-1 rounded-md border border-accent-500/20">{fundamentals.sector}</span>
        </div>
      )}
    </div>
  );
}
