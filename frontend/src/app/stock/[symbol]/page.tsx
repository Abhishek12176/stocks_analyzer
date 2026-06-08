"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Skeleton } from "@/components/ui/Skeleton";
import { Tabs } from "@/components/ui/Tabs";
import { CompanyHeader } from "@/components/stock/CompanyHeader";
import { MetricGrid } from "@/components/stock/MetricGrid";
import { TradeSignal } from "@/components/stock/TradeSignal";
import { TechnicalPanel } from "@/components/stock/TechnicalPanel";
import { FundamentalScoreCard } from "@/components/stock/FundamentalScoreCard";
import { FundamentalPanel } from "@/components/stock/FundamentalPanel";
import { ShareholdingChart } from "@/components/stock/ShareholdingChart";
import { MajorShareholders } from "@/components/stock/MajorShareholders";
import { NewsFeed } from "@/components/stock/NewsFeed";
import { RawDataTable } from "@/components/stock/RawDataTable";
import { useWatchlistStore } from "@/store/watchlistStore";
import { useHistoryStore } from "@/store/historyStore";
import { useStockPrice } from "@/hooks/useStock";
import { useSignal } from "@/hooks/useSignal";
import { useNews } from "@/hooks/useNews";
import { useShareholding } from "@/hooks/useShareholding";
import { useFullAnalysis } from "@/hooks/useFullAnalysis";
import { motion, AnimatePresence } from "framer-motion";

const TAB_IDS = ["technical", "raw-data", "fundamentals", "ownership", "news", "signal"] as const;
type TabId = (typeof TAB_IDS)[number];

const TABS = [
  { id: "technical", label: "Technical" },
  { id: "raw-data", label: "Raw Data" },
  { id: "fundamentals", label: "Fundamentals" },
  { id: "ownership", label: "Ownership" },
  { id: "news", label: "News" },
  { id: "signal", label: "Signal" },
];

export default function StockDetailPage() {
  const params = useParams<{ symbol: string }>();
  const symbol = params?.symbol ?? "";

  const [activeTab, setActiveTab] = useState<TabId>("technical");
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    setHydrated(true);
  }, []);

  const watchlist = useWatchlistStore();
  const addToHistory = useHistoryStore((s) => s.add);
  const isWatched = hydrated && watchlist.has(symbol);

  const {
    data: analysisData,
    isLoading: analysisLoading,
    isError: analysisError,
    refetch: refetchAnalysis,
  } = useFullAnalysis(symbol);
  const {
    data: priceHistory,
    isLoading: historyLoading,
  } = useStockPrice(symbol);
  const {
    data: signalData,
    isLoading: signalLoading,
  } = useSignal(symbol);
  const {
    data: newsData,
    isLoading: newsLoading,
  } = useNews(symbol);
  const {
    data: shareholdingData,
    isLoading: shareholdingLoading,
  } = useShareholding(symbol);

  useEffect(() => {
    if (!symbol) return;
    addToHistory({
      symbol,
      name: symbol,
      price: analysisData?.quote.currentPrice ?? null,
      signal: analysisData?.signal.action ?? null,
      score: analysisData?.score.total ?? null,
    });
  }, [symbol, analysisData, addToHistory]);

  const toggleWatchlist = useCallback(() => {
    if (isWatched) {
      watchlist.remove(symbol);
    } else {
      watchlist.add(symbol, symbol);
    }
  }, [symbol, isWatched, watchlist]);

  const isLoading = analysisLoading || historyLoading;
  const hasError = analysisError;

  if (!symbol) {
    return (
      <div className="flex flex-col items-center justify-center py-32 text-center px-6">
        <div className="mb-4 size-12 rounded-xl border border-dashed border-neutral-700 flex items-center justify-center text-neutral-600">
          <svg className="size-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
          </svg>
        </div>
        <h3 className="text-base font-semibold text-neutral-300">No symbol provided</h3>
        <Link
          href="/"
          className="mt-4 text-sm font-medium text-accent-500 hover:underline"
        >
          Return to Dashboard
        </Link>
      </div>
    );
  }

  if (hasError && !isLoading) {
    return (
      <div className="animate-fade-in">
        <div className="flex items-center border-b border-neutral-800/60 px-8 py-3.5">
          <Link
            href="/"
            className="text-sm text-neutral-500 hover:text-neutral-300 transition-colors"
          >
            <svg className="size-4 inline-block mr-1" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
            </svg>
            Dashboard
          </Link>
          <span className="mx-2 text-neutral-700">/</span>
          <span className="font-mono text-sm font-bold text-neutral-50">{symbol}</span>
          <Badge variant="accent" className="ml-2 text-[10px]">NSE</Badge>
        </div>
        <div className="flex flex-col items-center justify-center py-32 text-center px-6">
          <div className="mb-4 size-12 rounded-xl border border-dashed border-neutral-700 flex items-center justify-center text-neutral-600">
            <svg className="size-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
            </svg>
          </div>
          <h3 className="text-base font-semibold text-neutral-300">
            Unable to load stock analysis
          </h3>
          <p className="mt-1 text-sm text-neutral-500 max-w-sm">
            {symbol} data could not be loaded. The stock may be unavailable or the service may be temporarily down.
          </p>
          <button
            onClick={() => refetchAnalysis()}
            className="mt-6 rounded-xl bg-accent-500/10 px-5 py-2.5 text-sm font-medium text-accent-500 border border-accent-500/20 hover:bg-accent-500/20 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="animate-fade-in">
      <div className="flex items-center justify-between border-b border-neutral-800/60 px-8 py-3.5">
        <div className="flex items-center gap-3">
          <Link
            href="/"
            className="text-sm text-neutral-500 hover:text-neutral-300 transition-colors"
          >
            <svg className="size-4 inline-block mr-1" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
            </svg>
            Dashboard
          </Link>
          <span className="text-neutral-700">/</span>
          <span className="font-sans text-xl font-semibold tracking-tight text-neutral-50">
            {symbol}
          </span>
          <Badge variant="accent" className="text-[10px]">NSE</Badge>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={toggleWatchlist}>
            {isWatched ? (
              <span className="text-signal-neutral">★</span>
            ) : (
              <span className="text-neutral-500">☆</span>
            )}
            <span className="ml-1.5">{isWatched ? "Watched" : "Watchlist"}</span>
          </Button>
        </div>
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={symbol}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
        >
          <CompanyHeader
            quote={analysisData?.quote ?? null}
            loading={analysisLoading}
          />

          <MetricGrid
            indicators={analysisData?.indicators ?? null}
            loading={analysisLoading}
          />

          <TradeSignal
            signal={analysisData?.signal ?? null}
            price={analysisData?.quote ? { price: analysisData.quote.currentPrice, change: analysisData.quote.change, changePercent: analysisData.quote.changePercent } : null}
            loading={analysisLoading}
          />
        </motion.div>
      </AnimatePresence>

      <div className="px-8">
        <div className="border-b border-neutral-800/60">
          <Tabs tabs={TABS} active={activeTab} onChange={(id) => setActiveTab(id as TabId)} />
        </div>

        <div className="py-6">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
            >
              {activeTab === "technical" && (
                <TechnicalPanel
                  symbol={symbol}
                  history={priceHistory?.history ?? null}
                  indicators={analysisData?.indicators ?? null}
                  loading={analysisLoading}
                />
              )}

              {activeTab === "fundamentals" && (
                <div className="grid grid-cols-1 lg:grid-cols-[280px_1fr] gap-8">
                  <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 backdrop-blur-sm p-5">
                    <FundamentalScoreCard
                      score={analysisData?.score ?? null}
                      loading={analysisLoading}
                    />
                  </div>
                  <FundamentalPanel
                    fundamentals={analysisData?.fundamentals ?? null}
                    loading={analysisLoading}
                  />
                </div>
              )}

              {activeTab === "ownership" && (
                <div className="space-y-8">
                  <section className="rounded-2xl border border-neutral-800 bg-[rgba(15,23,42,0.6)] backdrop-blur-sm p-5">
                    <h3 className="text-[13px] font-semibold text-neutral-400 uppercase tracking-[1px] mb-5">
                      Quarterly Shareholding Pattern
                    </h3>
                    <ShareholdingChart
                      quarterlyData={shareholdingData?.quarterlyData ?? null}
                      loading={shareholdingLoading}
                    />
                  </section>
                  <section>
                    <h3 className="text-[13px] font-semibold text-neutral-400 uppercase tracking-[1px] mb-4">
                      Major Shareholders
                    </h3>
                    <MajorShareholders
                      shareholders={shareholdingData?.majorShareholders ?? null}
                      source={shareholdingData?.source}
                      loading={shareholdingLoading}
                    />
                  </section>
                </div>
              )}

              {activeTab === "news" && (
                <NewsFeed
                  articles={newsData?.articles ?? null}
                  loading={newsLoading}
                />
              )}

              {activeTab === "signal" && (
                <TradeSignal
                  signal={signalData?.signal ?? analysisData?.signal ?? null}
                  price={signalData?.quote ? { price: signalData.quote.price, change: signalData.quote.change, changePercent: signalData.quote.changePercent } : analysisData?.quote ? { price: analysisData.quote.currentPrice, change: analysisData.quote.change, changePercent: analysisData.quote.changePercent } : null}
                  loading={signalLoading || (!signalData && analysisLoading)}
                />
              )}

              {activeTab === "raw-data" && (
                <RawDataTable
                  history={priceHistory?.history ?? null}
                  loading={analysisLoading}
                />
              )}
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
