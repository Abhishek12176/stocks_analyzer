"use client";

import { useState, useCallback, useMemo } from "react";
import { Button } from "@/components/ui/Button";
import { Skeleton } from "@/components/ui/Skeleton";
import { cn } from "@/lib/cn";
import type { MajorShareholder } from "@/types/shareholding";

interface MajorShareholdersProps {
  shareholders?: MajorShareholder[] | null;
  source?: string;
  loading?: boolean;
}

const MOCK_SHAREHOLDERS: MajorShareholder[] = [
  { rank: 1, name: "Life Insurance Corporation of India", holdingPercent: 8.45 },
  { rank: 2, name: "HDFC Asset Management Company Ltd", holdingPercent: 4.32 },
  { rank: 3, name: "SBI Mutual Fund", holdingPercent: 3.18 },
  { rank: 4, name: "ICICI Prudential Mutual Fund", holdingPercent: 2.76 },
  { rank: 5, name: "Nippon Life India Asset Management", holdingPercent: 2.14 },
  { rank: 6, name: "Kotak Mahindra Asset Management", holdingPercent: 1.89 },
  { rank: 7, name: "Aditya Birla Sun Life Mutual Fund", holdingPercent: 1.65 },
  { rank: 8, name: "Government of India", holdingPercent: 1.42 },
];

const FALLBACK_SOURCES = new Set(["fallback", "yfinance", "N/A"]);

export function MajorShareholders({ shareholders, source, loading }: MajorShareholdersProps) {
  const [showMock, setShowMock] = useState(false);
  const [sortAsc, setSortAsc] = useState(false);

  const displayData = shareholders && shareholders.length > 0 ? shareholders : showMock ? MOCK_SHAREHOLDERS : null;
  const dataSource = shareholders && shareholders.length > 0 ? source : showMock ? "Sample Data" : undefined;
  const isFallback = FALLBACK_SOURCES.has(source ?? "");

  const sortedData = useMemo(() => {
    if (!displayData) return [];
    const sorted = [...displayData];
    sorted.sort((a, b) =>
      sortAsc ? a.holdingPercent - b.holdingPercent : b.holdingPercent - a.holdingPercent
    );
    return sorted.map((item, i) => ({ ...item, displayRank: i + 1 }));
  }, [displayData, sortAsc]);

  const handleExportCSV = useCallback(() => {
    if (!displayData) return;
    const headers = ["Rank", "Name", "Holding (%)"];
    const rows = displayData.map((s) => [s.rank.toString(), s.name, s.holdingPercent.toFixed(2)]);
    const csv = [headers, ...rows].map((r) => r.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "major-shareholders.csv";
    a.click();
    URL.revokeObjectURL(url);
  }, [displayData]);

  const handleRefresh = useCallback(() => {
    window.location.reload();
  }, []);

  if (loading) {
    return (
      <div className="space-y-3">
        <Skeleton className="h-5 w-48" />
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton key={i} className="h-12 rounded-xl" />
        ))}
      </div>
    );
  }

  if (!displayData) {
    return (
      <div className="rounded-2xl border border-dashed border-neutral-700/60 bg-[rgba(15,23,42,0.4)] p-8">
        <div className="flex flex-col items-center text-center max-w-sm mx-auto">
          <div className="size-14 rounded-2xl bg-[rgba(15,23,42,0.8)] border border-neutral-800 flex items-center justify-center mb-4">
            <svg className="size-6 text-neutral-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </div>
          <h4 className="text-sm font-semibold text-neutral-300 mb-1">
            No Major Shareholder Data Available
          </h4>
          <p className="text-xs text-neutral-500 leading-relaxed mb-5">
            Major shareholder data could not be retrieved from the available sources at this time.
            You can load sample data for a preview.
          </p>
          <div className="flex items-center gap-3">
            <Button variant="secondary" size="sm" onClick={() => setShowMock(true)}>
              <svg className="size-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              Load Sample Data
            </Button>
            <Button variant="ghost" size="sm" onClick={handleRefresh}>
              <svg className="size-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Retry
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs text-neutral-500">Source:</span>
          <span className={cn("text-xs font-semibold text-neutral-400", showMock && "text-accent-400")}>
            {dataSource ?? "Unknown"}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={() => setSortAsc(!sortAsc)}>
            <svg className="size-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={sortAsc ? "M3 4h13M3 8h9m-9 4h6m4 0l4-4m0 0l4 4m-4-4v12" : "M3 4h13M3 8h9m-9 4h9m5-4v12m0 0l-4-4m4 4l4-4"} />
            </svg>
            {sortAsc ? "Lowest First" : "Highest First"}
          </Button>
          <Button variant="ghost" size="sm" onClick={handleExportCSV}>
            <svg className="size-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Export CSV
          </Button>
        </div>
      </div>

      <div className="rounded-2xl border border-neutral-800 bg-[rgba(15,23,42,0.6)] overflow-hidden">
        <div className="grid grid-cols-[44px_1fr_110px] gap-2 px-5 py-3.5 bg-white/[0.02] border-b border-neutral-800">
          <span className="text-[11px] font-semibold text-neutral-500 uppercase tracking-[0.5px]">#</span>
          <span className="text-[11px] font-semibold text-neutral-500 uppercase tracking-[0.5px]">Shareholder Name</span>
          <span className="text-[11px] font-semibold text-neutral-500 uppercase tracking-[0.5px] text-right">Holding (%)</span>
        </div>
        {sortedData.map((holder, i) => (
          <div
            key={`${holder.name}-${i}`}
            className={cn(
              "grid grid-cols-[44px_1fr_110px] gap-2 px-5 py-3 items-center transition-colors",
              "hover:bg-white/[0.02]",
              i < sortedData.length - 1 && "border-b border-neutral-800/50"
            )}
          >
            <span className="font-mono text-xs text-neutral-500">
              {holder.displayRank}
            </span>
            <span className="text-sm text-neutral-200 truncate">
              {holder.name}
            </span>
            <div className="flex items-center justify-end gap-2">
              <div className="w-16 h-1.5 rounded-full bg-neutral-800 overflow-hidden">
                <div
                  className="h-full rounded-full bg-accent-500/50"
                  style={{ width: `${Math.min(holder.holdingPercent * 5, 100)}%` }}
                />
              </div>
              <span className="font-mono text-sm font-semibold text-neutral-100 text-right tabular-nums min-w-[56px]">
                {holder.holdingPercent.toFixed(2)}%
              </span>
            </div>
          </div>
        ))}
      </div>

      {(showMock || isFallback) && (
        <div className="rounded-xl bg-[rgba(15,23,42,0.6)] border border-neutral-800 px-4 py-3">
          <div className="flex items-start gap-2.5">
            <svg className="size-4 text-accent-400 mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div>
              <p className="text-xs font-medium text-neutral-400">
                {isFallback ? "Estimated Shareholder Data" : "Sample Data Preview"}
              </p>
              <p className="text-[10px] text-neutral-600 mt-0.5">
                {isFallback
                  ? "Live shareholder data sources (Screener.in, Moneycontrol, NSE) are currently unavailable. Showing estimated figures based on available market data."
                  : "This is illustrative sample data. Real shareholder data will appear here once the backend fetches from connected sources."}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
