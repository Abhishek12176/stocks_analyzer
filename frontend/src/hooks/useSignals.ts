"use client";

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/lib/api";
import type { SignalCategory, SignalCategoryId } from "@/types/signals";

interface SignalsResponse {
  categories: SignalCategory[];
  totalStocks: number;
  generatedAt: string;
}

const CATEGORY_ORDER: SignalCategoryId[] = [
  "strong-buy", "buy", "hold", "sell", "strong-sell",
  "rsi-oversold", "rsi-overbought",
  "macd-bullish", "macd-bearish",
  "bullish-trend", "bearish-trend",
];

export function useSignals() {
  const { data, isLoading, isError, refetch } = useQuery<SignalsResponse>({
    queryKey: ["signals", "all"],
    queryFn: () =>
      apiGet<SignalsResponse>(`/signals/all`),
    staleTime: 300_000,
    retry: 2,
    retryDelay: 1000,
  });

  const categories = useMemo((): SignalCategory[] => {
    if (!data?.categories) return [];
    return CATEGORY_ORDER.map((id) => {
      const found = data.categories.find((c) => c.id === id);
      return found ?? { id, label: id, stocks: [], count: 0 };
    });
  }, [data]);

  const totalStocks = data?.totalStocks ?? 0;

  return { categories, totalStocks, isLoading, isError, refetch };
}
