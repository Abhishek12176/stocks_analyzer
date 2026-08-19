import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/lib/api";
import type { ShareholdingResponse } from "@/types/shareholding";

export function useShareholding(symbol: string) {
  return useQuery<ShareholdingResponse>({
    queryKey: ["stock", "shareholding", symbol],
    queryFn: () =>
      apiGet<ShareholdingResponse>(
        `/stock/${encodeURIComponent(symbol)}/shareholding`
      ),
    enabled: !!symbol,
    staleTime: 86_400_000,
    retry: false,
  });
}
