export type BasketCategory =
  | "ai-growth"
  | "momentum"
  | "strong-buy"
  | "banking"
  | "it"
  | "auto"
  | "pharma"
  | "energy";

export type Sentiment = "bullish" | "bearish" | "neutral";

export interface BasketStock {
  symbol: string;
  name: string;
  currentPrice: number;
  change: number;
  changePercent: number;
  signal: "BUY" | "SELL" | "HOLD" | "NEUTRAL";
  rsi: number;
  macd: number;
  trend: string;
  confidence: number;
}

export interface BasketDefinition {
  id: BasketCategory;
  name: string;
  description: string;
  sector: string;
  symbols: string[];
}

export interface BasketData {
  id: BasketCategory;
  name: string;
  description: string;
  sector: string;
  stocks: BasketStock[];
  sentiment: Sentiment;
  averageReturn: number;
  totalStocks: number;
}

export interface BasketSummary {
  totalBaskets: number;
  totalStocks: number;
  bullishBaskets: number;
  bearishBaskets: number;
}
