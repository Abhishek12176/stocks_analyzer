"use client";

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/lib/api";
import { BASKET_DEFINITIONS, FALLBACK_STOCK_DATA } from "@/lib/basket-data";
import type { PriceResponse } from "@/types/stock";
import type { SignalResponse } from "@/types/signal";
import type { SignalStock, SignalCategory, SignalCategoryId } from "@/types/signals";

const ALL_SYMBOLS = [...new Set(BASKET_DEFINITIONS.flatMap((d) => d.symbols))];

function buildFallbackStock(symbol: string): SignalStock | null {
  const fb = FALLBACK_STOCK_DATA[symbol];
  if (!fb) return null;
  const price = fb.currentPrice ?? 100;
  const isStrong = fb.trend === "Strong Trend";
  const sma20 = isStrong ? price * 0.97 : price * 1.03;
  const sma50 = isStrong ? price * 0.94 : price * 1.06;
  const macd = fb.macd ?? 0;
  const macdSignal = macd > 0 ? macd * 0.75 : macd * 1.25;
  const rsi = fb.rsi ?? 50;
  const conf = fb.confidence ?? 50;
  const sig = conf >= 70 ? "BUY" : conf <= 30 ? "SELL" : "HOLD" as const;
  return {
    symbol,
    name: fb.name ?? symbol,
    currentPrice: price,
    change: fb.change ?? 0,
    changePercent: fb.changePercent ?? 0,
    signal: sig,
    confidence: conf,
    rsi,
    macd,
    macdSignal,
    sma20: Math.round(sma20 * 100) / 100,
    sma50: Math.round(sma50 * 100) / 100,
    trend: isStrong ? "Strong Trend" : "Weak Trend",
    sparkline: [],
  };
}

async function fetchStockData(symbol: string): Promise<SignalStock | null> {
  try {
    const [priceData, signalData] = await Promise.all([
      apiGet<PriceResponse>(`/stock/${encodeURIComponent(symbol)}/price`, {
        period: "1mo",
      }),
      apiGet<SignalResponse>(`/stock/${encodeURIComponent(symbol)}/signal`),
    ]);
    const { quote, indicators, history } = priceData;
    const { signal } = signalData;
    return {
      symbol,
      name: quote.companyName,
      currentPrice: quote.currentPrice,
      change: quote.change,
      changePercent: quote.changePercent,
      signal: signal.action,
      confidence: signal.confidence,
      rsi: indicators.rsi,
      macd: indicators.macd,
      macdSignal: indicators.signal,
      sma20: indicators.sma20,
      sma50: indicators.sma50,
      trend: indicators.sma20 > indicators.sma50 ? "Strong Trend" : "Weak Trend",
      sparkline: history.map((h) => h.close).slice(-30),
    };
  } catch {
    return buildFallbackStock(symbol);
  }
}

function categorizeStock(stock: SignalStock): SignalCategoryId[] {
  const cats: SignalCategoryId[] = [];
  const isStrongTrend = stock.trend === "Strong Trend";

  if (stock.signal === "BUY" && stock.confidence >= 85 && isStrongTrend) {
    cats.push("strong-buy");
  }
  if (stock.signal === "BUY" && stock.confidence >= 70) {
    cats.push("buy");
  }
  if (stock.signal === "HOLD") {
    cats.push("hold");
  }
  if (stock.signal === "SELL") {
    cats.push("sell");
  }
  if (stock.signal === "SELL" && stock.confidence >= 85) {
    cats.push("strong-sell");
  }
  if (stock.rsi < 30) {
    cats.push("rsi-oversold");
  }
  if (stock.rsi > 70) {
    cats.push("rsi-overbought");
  }
  if (stock.macd > stock.macdSignal) {
    cats.push("macd-bullish");
  }
  if (stock.macd < stock.macdSignal) {
    cats.push("macd-bearish");
  }
  if (stock.sma20 > stock.sma50) {
    cats.push("bullish-trend");
  }
  if (stock.sma20 < stock.sma50) {
    cats.push("bearish-trend");
  }
  return cats;
}

const CATEGORY_LABELS: Record<SignalCategoryId, string> = {
  "strong-buy": "Strong Buy",
  buy: "Buy",
  hold: "Hold",
  sell: "Sell",
  "strong-sell": "Strong Sell",
  "rsi-oversold": "RSI Oversold",
  "rsi-overbought": "RSI Overbought",
  "macd-bullish": "MACD Bullish Crossover",
  "macd-bearish": "MACD Bearish Crossover",
  "bullish-trend": "Bullish Trend",
  "bearish-trend": "Bearish Trend",
};

const CATEGORY_ORDER: SignalCategoryId[] = [
  "strong-buy",
  "buy",
  "hold",
  "sell",
  "strong-sell",
  "rsi-oversold",
  "rsi-overbought",
  "macd-bullish",
  "macd-bearish",
  "bullish-trend",
  "bearish-trend",
];

export function useSignals() {
  const { data: allStocks, isLoading, isError, refetch } = useQuery<SignalStock[]>({
    queryKey: ["signals", "all"],
    queryFn: async () => {
      const results = await Promise.allSettled(
        ALL_SYMBOLS.map((s) => fetchStockData(s))
      );
      return results
        .map((r) => (r.status === "fulfilled" ? r.value : null))
        .filter((s): s is SignalStock => s !== null);
    },
    staleTime: 300_000,
    retry: 1,
  });

  const categories = useMemo((): SignalCategory[] => {
    if (!allStocks) return [];
    const map = new Map<SignalCategoryId, SignalStock[]>();
    for (const id of CATEGORY_ORDER) map.set(id, []);

    for (const stock of allStocks) {
      const assigned = categorizeStock(stock);
      for (const catId of assigned) {
        const arr = map.get(catId);
        if (arr) arr.push(stock);
      }
    }

    return CATEGORY_ORDER.map((id) => {
      const stocks = (map.get(id) ?? []).sort(
        (a, b) => b.confidence - a.confidence
      );
      return { id, label: CATEGORY_LABELS[id], stocks, count: stocks.length };
    });
  }, [allStocks]);

  const totalStocks = allStocks?.length ?? 0;

  return { categories, totalStocks, isLoading, isError, refetch };
}
