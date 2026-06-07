import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/lib/api";
import type { FundamentalsResponse } from "@/types/fundamentals";

export function useFundamentals(symbol: string) {
  return useQuery<FundamentalsResponse>({
    queryKey: ["stock", "fundamentals", symbol],
    queryFn: () =>
      apiGet<FundamentalsResponse>(
        `/stock/${encodeURIComponent(symbol)}/fundamentals`
      ),
    enabled: !!symbol,
    staleTime: 3_600_000,
  });
}
