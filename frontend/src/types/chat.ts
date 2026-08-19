export type ChatSignal = "BUY" | "SELL" | "HOLD" | "NEUTRAL";

export interface ChatStock {
  symbol: string;
  name: string;
  currentPrice: number;
  change: number;
  changePercent: number;
  signal: ChatSignal;
  confidence: number;
  trend: string;
  prediction20d: string;
  predictionLabel: string;
  rsi: number | null;
  macd: number | null;
  macdSignal: number | null;
  sma20: number | null;
  sma50: number | null;
  sparkline: number[];
}

export interface ChatIntent {
  maxPrice: number | null;
  minPrice: number | null;
  action: string | null;
  top: number;
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

export interface ChatResponse {
  reply: string;
  stocks: ChatStock[];
  totalFound: number;
  intent: ChatIntent;
  source: "llm" | "existing-model";
  generatedAt: string;
}
