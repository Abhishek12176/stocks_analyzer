import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/lib/api";
import type { PriceResponse } from "@/types/stock";

export function useStockPrice(symbol: string, period: string = "1y") {
  return useQuery<PriceResponse>({
    queryKey: ["stock", "price", symbol, period],
    queryFn: () =>
      apiGet<PriceResponse>(`/stock/${encodeURIComponent(symbol)}/price`, {
        period,
      }),
    enabled: !!symbol,
    staleTime: 60_000,
    refetchInterval: 60_000,
  });
}
