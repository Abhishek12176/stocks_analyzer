export interface StockPrice {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface TechnicalIndicators {
  rsi: number;
  macd: number;
  signal: number;
  sma20: number;
  sma50: number;
}

export interface StockQuote {
  symbol: string;
  companyName: string;
  exchange: string;
  currentPrice: number;
  change: number;
  changePercent: number;
  lastUpdated: string;
}

export interface PriceResponse {
  quote: StockQuote;
  history: StockPrice[];
  indicators: TechnicalIndicators;
}

export interface IntradayPrice {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface IntradayResponse {
  symbol: string;
  company_name: string;
  exchange: string;
  current_price: number;
  change: number;
  change_percent: number;
  last_updated: string;
  interval: string;
  history: IntradayPrice[];
}
