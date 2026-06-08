import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/lib/api";
import type { IntradayResponse } from "@/types/stock";

export function useIntradayStock(symbol: string, interval: string = "5m") {
  return useQuery<IntradayResponse>({
    queryKey: ["stock", "intraday", symbol, interval],
    queryFn: () =>
      apiGet<IntradayResponse>(`/stock/${encodeURIComponent(symbol)}/intraday`, {
        interval,
      }),
    enabled: !!symbol,
    staleTime: 120_000,
  });
}
