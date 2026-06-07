"use client";

import { cn } from "@/lib/cn";
import { formatPrice, formatPercent } from "@/lib/formatters";
import { Skeleton } from "@/components/ui/Skeleton";
import { Badge } from "@/components/ui/Badge";
import type { StockQuote } from "@/types/stock";

interface CompanyHeaderProps {
  quote?: StockQuote | null;
  loading?: boolean;
}

export function CompanyHeader({ quote, loading }: CompanyHeaderProps) {
  if (loading) {
    return (
      <div className="border-b border-neutral-800/60 px-8 py-6">
        <Skeleton className="h-4 w-48 mb-2" />
        <Skeleton className="h-10 w-52 mb-1.5" />
        <Skeleton className="h-4 w-36" />
      </div>
    );
  }

  if (!quote) return null;

  const isPositive = quote.change >= 0;
  const changeArrow = isPositive ? "▲" : "▼";

  return (
    <div className="border-b border-neutral-800/60 px-8 py-6">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-base font-semibold tracking-tight text-neutral-300">
          {quote.companyName}
        </span>
        <span className="text-neutral-700">·</span>
        <Badge variant="accent" className="text-[10px] px-2 py-0.5">{quote.exchange}</Badge>
      </div>
      <div className="flex items-baseline gap-4">
        <h1 className="font-mono text-4xl font-bold tracking-tight text-neutral-50">
          {formatPrice(quote.currentPrice)}
        </h1>
        <div className="flex items-baseline gap-2.5">
          <span
            className={cn(
              "font-mono text-lg font-semibold",
              isPositive ? "text-signal-bullish" : "text-signal-bearish"
            )}
          >
            {changeArrow} {formatPercent(quote.changePercent)}
          </span>
          <span
            className={cn(
              "font-mono text-sm",
              isPositive ? "text-signal-bullish" : "text-signal-bearish"
            )}
          >
            {isPositive ? "+" : ""}
            {formatPrice(quote.change)}
          </span>
        </div>
      </div>
      <div className="mt-1.5 flex items-center gap-3">
        <span className="text-xs text-neutral-600">
          Updated {new Date(quote.lastUpdated).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" })}
        </span>
        <span className="text-neutral-800">|</span>
        <span className="flex items-center gap-1.5 text-xs text-neutral-500">
          <span className="inline-block size-1.5 rounded-full bg-signal-bullish animate-pulse" />
          Live
        </span>
      </div>
    </div>
  );
}
