"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { motion } from "framer-motion";
import { apiGet } from "@/lib/api";
import { SparklineChart } from "@/components/chart/SparklineChart";
import { Skeleton } from "@/components/ui/Skeleton";
import { Badge } from "@/components/ui/Badge";
import { cn } from "@/lib/cn";
import { formatPrice, formatPercent } from "@/lib/formatters";

interface StockOverviewItem {
  symbol: string;
  companyName: string;
  exchange: string;
  currentPrice: number;
  change: number;
  changePercent: number;
  score: number;
  sparkline: number[];
  signal: {
    action: string;
    direction: string;
    confidence: number;
  };
}

interface MarketOverview {
  bullish: StockOverviewItem[];
  bearish: StockOverviewItem[];
  generatedAt: string;
  totalScanned: number;
}

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.04 },
  },
};

const rowVariants = {
  hidden: { opacity: 0, x: -8 },
  visible: { opacity: 1, x: 0, transition: { duration: 0.25, ease: [0.16, 1, 0.3, 1] as const } },
};

function StockSkeletonRow() {
  return (
    <div className="flex items-center gap-3 px-4 py-3">
      <div className="flex-1 space-y-1.5">
        <Skeleton className="h-3.5 w-20" />
        <Skeleton className="h-3 w-32" />
      </div>
      <Skeleton className="h-3 w-16" />
      <Skeleton className="h-3 w-16" />
      <Skeleton className="h-4 w-12" />
      <Skeleton className="h-7 w-20" />
    </div>
  );
}

function StockCardSkeleton({ title }: { title: string }) {
  return (
    <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 backdrop-blur-sm overflow-hidden">
      <div className="px-5 py-4 border-b border-neutral-800/60">
        <Skeleton className="h-5 w-52" />
      </div>
      <div className="divide-y divide-neutral-800/40">
        {Array.from({ length: 5 }).map((_, i) => (
          <StockSkeletonRow key={i} />
        ))}
      </div>
    </div>
  );
}

function StockRow({ item, index, isBullish }: { item: StockOverviewItem; index: number; isBullish: boolean }) {
  const color = isBullish ? "var(--color-signal-bullish)" : "var(--color-signal-bearish)";
  const bgColor = isBullish ? "bg-signal-bullish-bg" : "bg-signal-bearish-bg";
  const directionLabel = isBullish ? "Bullish" : "Bearish";
  const trendArrow = isBullish ? "▲" : "▼";

  return (
    <Link href={`/stock/${item.symbol}`}>
      <motion.div
        variants={rowVariants}
        whileHover={{ backgroundColor: "rgba(30, 41, 59, 0.4)" }}
        className="flex items-center gap-3 px-4 py-2.5 transition-colors duration-150 cursor-pointer group"
      >
        <span className="w-5 text-center text-[11px] font-mono font-bold text-neutral-500 group-hover:text-neutral-400 transition-colors flex-shrink-0">
          {index + 1}
        </span>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5">
            <span className="font-mono text-sm font-bold text-neutral-50 group-hover:text-accent-400 transition-colors truncate">
              {item.symbol}
            </span>
            <span className="text-[10px] font-semibold text-neutral-400 px-1.5 py-0.5 rounded border border-neutral-800 flex-shrink-0">
              {item.exchange}
            </span>
            <Badge
              variant={isBullish ? "bullish" : "bearish"}
              className="text-[9px] px-1.5 py-0"
            >
              {directionLabel}
            </Badge>
          </div>
          <p className="text-[11px] text-neutral-400 truncate leading-tight mt-0.5">
              {item.companyName}
            </p>
        </div>

        <div className="text-right flex-shrink-0">
          <p className="font-mono text-sm font-semibold text-neutral-50">
            {formatPrice(item.currentPrice)}
          </p>
          <p
            className={cn(
              "font-mono text-[11px] font-medium",
              isBullish ? "text-signal-bullish" : "text-signal-bearish"
            )}
          >
            {trendArrow} {formatPercent(item.changePercent)}
          </p>
        </div>

        <div className="flex items-center gap-2 flex-shrink-0 min-w-[90px] justify-end">
          <div className="flex flex-col items-end">
            <span
              className={cn(
                "font-mono text-xs font-bold",
                isBullish ? "text-signal-bullish" : "text-signal-bearish"
              )}
            >
              {item.signal.confidence.toFixed(0)}%
            </span>
            <span className="text-[10px] text-neutral-500 leading-tight">
              Strength
            </span>
          </div>
          <SparklineChart
            data={item.sparkline}
            color={color}
            width={72}
            height={24}
          />
        </div>
      </motion.div>
    </Link>
  );
}

function StockCard({
  title,
  items,
  isBullish,
  loading,
}: {
  title: string;
  items: StockOverviewItem[];
  isBullish: boolean;
  loading: boolean;
}) {
  const accentColor = isBullish ? "var(--color-signal-bullish)" : "var(--color-signal-bearish)";
  const borderGlow = isBullish ? "rgba(0,200,83,0.15)" : "rgba(255,82,82,0.15)";

  if (loading) return <StockCardSkeleton title={title} />;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      className={cn(
        "rounded-2xl border bg-neutral-900/60 backdrop-blur-sm overflow-hidden",
        "transition-all duration-300 hover:shadow-[0_20px_60px_rgba(0,0,0,0.4)]",
        isBullish
          ? "border-signal-bullish/10 hover:border-signal-bullish/20"
          : "border-signal-bearish/10 hover:border-signal-bearish/20"
      )}
    >
      <div
        className={cn(
          "px-5 py-4 border-b flex items-center justify-between",
          isBullish ? "border-signal-bullish/10" : "border-signal-bearish/10"
        )}
      >
        <div className="flex items-center gap-2.5">
          <span
            className="inline-block size-2 rounded-full flex-shrink-0 animate-pulse"
            style={{ backgroundColor: accentColor, boxShadow: `0 0 6px ${accentColor}` }}
          />
            <h2 className="text-sm font-bold text-neutral-50 tracking-tight">{title}</h2>
        </div>
        <span className="text-[11px] font-mono text-neutral-500">
          {items.length} stocks
        </span>
      </div>

      {items.length === 0 ? (
        <div className="px-5 py-12 text-center">
          <p className="text-sm text-neutral-500">No stocks available</p>
        </div>
      ) : (
        <div className="divide-y divide-neutral-800/40">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
          >
            {items.map((item, i) => (
              <StockRow key={item.symbol} item={item} index={i} isBullish={isBullish} />
            ))}
          </motion.div>
        </div>
      )}
    </motion.div>
  );
}

export default function StocksPage() {
  const { data, isLoading, error } = useQuery<MarketOverview>({
    queryKey: ["market", "overview"],
    queryFn: () => apiGet<MarketOverview>("/market/overview"),
    staleTime: 60_000,
    refetchInterval: 60_000,
  });

  return (
    <div className="animate-fade-in p-6 space-y-6">
      <motion.div
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <div className="flex items-center justify-between mb-1">
          <div>
            <h1 className="text-2xl font-bold text-neutral-50 tracking-[-0.5px]">
              Market Overview
            </h1>
            <p className="text-sm text-neutral-500 mt-1">
              Top bullish and bearish opportunities based on technical signals
            </p>
          </div>
          {data && (
            <div className="flex items-center gap-3 text-xs text-neutral-500">
              <span>
                Scanned{" "}
                <span className="font-mono font-bold text-neutral-400">
                  {data.totalScanned}
                </span>{" "}
                stocks
              </span>
              <span className="size-1 rounded-full bg-neutral-700" />
              <span>
                Updated{" "}
                {new Date(data.generatedAt).toLocaleTimeString("en-IN", {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </span>
            </div>
          )}
        </div>
      </motion.div>

      {error ? (
        <div className="rounded-2xl border border-signal-bearish/20 bg-signal-bearish/5 p-8 text-center">
          <p className="text-sm text-signal-bearish font-medium">
            Failed to load market data. Please try again later.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <StockCard
            title="Top Bullish Stock Today"
            items={data?.bullish ?? []}
            isBullish={true}
            loading={isLoading}
          />
          <StockCard
            title="Top Bearish Stock Today"
            items={data?.bearish ?? []}
            isBullish={false}
            loading={isLoading}
          />
        </div>
      )}
    </div>
  );
}
