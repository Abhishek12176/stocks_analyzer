import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/lib/api";
import type { ForecastResponse } from "@/types/forecast";

interface UseForecastOptions {
  enabled?: boolean;
}

export function useForecast(symbol: string, options: UseForecastOptions = {}) {
  return useQuery<ForecastResponse>({
    queryKey: ["stock", "forecast", symbol],
    queryFn: () =>
      apiGet<ForecastResponse>(
        `/stock/${encodeURIComponent(symbol)}/forecast`,
        { fast: "true" }
      ),
    enabled: !!symbol && options.enabled !== false,
    staleTime: 180_000,
    retry: 1,
    retryDelay: 1500,
  });
}