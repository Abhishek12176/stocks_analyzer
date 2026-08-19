import type { StockQuote, TechnicalIndicators } from "./stock";
import type { Fundamentals, FundamentalScore } from "./fundamentals";
import type { TradeSignal } from "./signal";

export interface FullAnalysisResponse {
  quote: StockQuote;
  indicators: TechnicalIndicators;
  fundamentals: Fundamentals;
  score: FundamentalScore;
  signal: TradeSignal;
}
