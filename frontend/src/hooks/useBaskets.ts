"use client";

import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/lib/api";
import { BASKET_DEFINITIONS, FALLBACK_STOCK_DATA } from "@/lib/basket-data";
import type {
  BasketData,
  BasketStock,
  Sentiment,
  BasketCategory,
} from "@/types/basket";
import type { PriceResponse } from "@/types/stock";
import type { SignalResponse } from "@/types/signal";

async function fetchBasketStock(symbol: string): Promise<BasketStock | null> {
  try {
    const [priceData, signalData] = await Promise.all([
      apiGet<PriceResponse>(`/stock/${encodeURIComponent(symbol)}/price`, {
        period: "1mo",
      }),
      apiGet<SignalResponse>(`/stock/${encodeURIComponent(symbol)}/signal`),
    ]);
    const { quote, indicators } = priceData;
    const { signal } = signalData;
    return {
      symbol,
      name: quote.companyName,
      currentPrice: quote.currentPrice,
      change: quote.change,
      changePercent: quote.changePercent,
      signal: signal.action,
      rsi: indicators.rsi,
      macd: indicators.macd,
      trend: indicators.sma20 > indicators.sma50 ? "Strong Trend" : "Weak Trend",
      confidence: signal.confidence,
    };
  } catch {
    const fallback = FALLBACK_STOCK_DATA[symbol];
    if (fallback) {
      return {
        symbol,
        name: fallback.name ?? symbol,
        currentPrice: fallback.currentPrice ?? 100,
        change: fallback.change ?? 0,
        changePercent: fallback.changePercent ?? 0,
        signal: (fallback.confidence ?? 50) >= 60 ? "BUY" : (fallback.confidence ?? 50) <= 30 ? "SELL" : "HOLD",
        rsi: fallback.rsi ?? 50,
        macd: fallback.macd ?? 0,
        trend: fallback.trend ?? "Neutral",
        confidence: fallback.confidence ?? 50,
      };
    }
    return null;
  }
}

function computeSentiment(stocks: BasketStock[]): Sentiment {
  if (stocks.length === 0) return "neutral";
  const bullish = stocks.filter((s) => s.signal === "BUY").length;
  const total = stocks.length;
  if (bullish / total >= 0.5) return "bullish";
  if (bullish / total < 0.3) return "bearish";
  return "neutral";
}

function computeAverageReturn(stocks: BasketStock[]): number {
  if (stocks.length === 0) return 0;
  const sum = stocks.reduce((acc, s) => acc + s.changePercent, 0);
  return sum / stocks.length;
}

function applyBasketLogic(
  basketId: BasketCategory,
  stocks: BasketStock[]
): BasketStock[] {
  switch (basketId) {
    case "ai-growth":
      return stocks
        .filter(
          (s) =>
            s.rsi >= 50 &&
            s.rsi <= 70 &&
            s.macd > 0 &&
            s.trend === "Strong Trend"
        )
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, 10);
    case "momentum":
      return stocks
        .sort((a, b) => {
          const scoreA =
            a.confidence +
            Math.abs(a.changePercent) * 2 +
            (a.signal === "BUY" ? 20 : a.signal === "SELL" ? -10 : 0) +
            (a.trend === "Strong Trend" ? 15 : 0);
          const scoreB =
            b.confidence +
            Math.abs(b.changePercent) * 2 +
            (b.signal === "BUY" ? 20 : b.signal === "SELL" ? -10 : 0) +
            (b.trend === "Strong Trend" ? 15 : 0);
          return scoreB - scoreA;
        })
        .slice(0, 10);
    case "strong-buy":
      return stocks
        .filter((s) => s.signal === "BUY")
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, 12);
    default:
      return stocks
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, 12);
  }
}

export function useBaskets() {
  return useQuery<BasketData[]>({
    queryKey: ["baskets", "all"],
    queryFn: async () => {
      const results = await Promise.all(
        BASKET_DEFINITIONS.map(async (def) => {
          const stockResults = await Promise.allSettled(
            def.symbols.map((s) => fetchBasketStock(s))
          );
          const allStocks = stockResults
            .map((r) => (r.status === "fulfilled" ? r.value : null))
            .filter((s): s is BasketStock => s !== null);

          const filteredStocks = applyBasketLogic(def.id, allStocks);

          return {
            id: def.id,
            name: def.name,
            description: def.description,
            sector: def.sector,
            stocks: filteredStocks,
            sentiment: computeSentiment(filteredStocks),
            averageReturn: computeAverageReturn(filteredStocks),
            totalStocks: filteredStocks.length,
          } satisfies BasketData;
        })
      );
      return results;
    },
    staleTime: 300_000,
    retry: 1,
  });
}

export function useBasket(id: BasketCategory) {
  const { data: baskets } = useBaskets();
  return baskets?.find((b) => b.id === id) ?? null;
}
