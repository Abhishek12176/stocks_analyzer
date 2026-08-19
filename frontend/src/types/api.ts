export interface ApiError {
  error: string;
  code: string;
  details?: string;
}

export interface ApiResponse<T> {
  data: T | null;
  error: ApiError | null;
  loading: boolean;
}

export interface StockSymbol {
  symbol: string;
  name: string;
  exchange: string;
}

export interface SearchResponse {
  results: StockSymbol[];
}

export interface ScreenerCriteria {
  minROE?: number;
  maxROE?: number;
  minDE?: number;
  maxDE?: number;
  minRevenueGrowth?: number;
  maxRevenueGrowth?: number;
  minProfitGrowth?: number;
  maxProfitGrowth?: number;
  minOPM?: number;
  maxPE?: number;
  sector?: string;
}

export interface ScreenerResult {
  symbol: string;
  name: string;
  price: number;
  pe: number;
  roe: number;
  de: number;
  revenueGrowth: number;
  profitGrowth: number;
  score: number;
}

export interface CompareRequest {
  symbols: string[];
}

export interface CompareResponse {
  stocks: {
    symbol: string;
    name: string;
    metrics: Record<string, number | null>;
  }[];
}
