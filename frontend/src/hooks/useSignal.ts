import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/lib/api";
import type { SignalResponse } from "@/types/signal";

export function useSignal(symbol: string) {
  return useQuery<SignalResponse>({
    queryKey: ["stock", "signal", symbol],
    queryFn: () =>
      apiGet<SignalResponse>(`/stock/${encodeURIComponent(symbol)}/signal`),
    enabled: !!symbol,
    staleTime: 60_000,
    refetchInterval: 60_000,
  });
}
