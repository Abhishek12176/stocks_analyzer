import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/lib/api";
import type { FullAnalysisResponse } from "@/types/analysis";

export function useFullAnalysis(symbol: string) {
  return useQuery<FullAnalysisResponse>({
    queryKey: ["stock", "full", symbol],
    queryFn: () =>
      apiGet<FullAnalysisResponse>(
        `/stock/${encodeURIComponent(symbol)}`
      ),
    enabled: !!symbol,
    staleTime: 180_000,
    retry: 1,
    retryDelay: 1000,
  });
}
