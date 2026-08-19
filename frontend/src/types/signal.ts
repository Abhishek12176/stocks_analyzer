export type SignalAction = "BUY" | "SELL" | "HOLD" | "NEUTRAL";
export type SignalDirection = "bullish" | "bearish" | "neutral";

export interface SignalReason {
  factor: string;
  impact: "bullish" | "bearish" | "neutral";
  detail: string;
}

export interface TradeSignal {
  action: SignalAction;
  direction: SignalDirection;
  confidence: number;
  reasons: SignalReason[];
  generatedAt: string;
}

export interface SignalResponse {
  signal: TradeSignal;
  quote: {
    price: number;
    change: number;
    changePercent: number;
  };
}
