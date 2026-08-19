import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/lib/api";
import type { NewsResponse } from "@/types/news";

export function useNews(symbol: string, limit: number = 30) {
  return useQuery<NewsResponse>({
    queryKey: ["stock", "news", symbol, limit],
    queryFn: () =>
      apiGet<NewsResponse>(`/stock/${encodeURIComponent(symbol)}/news`, {
        limit: String(limit),
      }),
    enabled: !!symbol,
    staleTime: 900_000,
  });
}
