export type SignalCategoryId =
  | "strong-buy"
  | "buy"
  | "hold"
  | "sell"
  | "strong-sell"
  | "rsi-oversold"
  | "rsi-overbought"
  | "macd-bullish"
  | "macd-bearish"
  | "bullish-trend"
  | "bearish-trend";

export interface SignalStock {
  symbol: string;
  name: string;
  currentPrice: number;
  change: number;
  changePercent: number;
  signal: "BUY" | "SELL" | "HOLD" | "NEUTRAL";
  confidence: number;
  rsi: number;
  macd: number;
  macdSignal: number;
  sma20: number;
  sma50: number;
  trend: string;
  sparkline: number[];
}

export interface SignalCategory {
  id: SignalCategoryId;
  label: string;
  stocks: SignalStock[];
  count: number;
}
