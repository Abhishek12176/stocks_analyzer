export interface Fundamentals {
  marketCap: number | null;
  peRatio: number | null;
  eps: number | null;
  bookValue: number | null;
  dividendYield: number | null;

  roe: number | null;
  roce: number | null;
  debtEquity: number | null;
  deCategory: string;
  opm: number | null;
  npm: number | null;
  grossMargin: number | null;

  currentRatio: number | null;
  interestCoverage: number | null;
  altmanZScore: number | null;

  revenueGrowth: number | null;
  profitGrowth: number | null;
  epsGrowth: number | null;
  fcfGrowth: number | null;

  assetTurnover: number | null;
  inventoryTurnover: number | null;
  receivablesDays: number | null;

  peVsSector: number | null;
  pbRatio: number | null;
  evEbitda: number | null;

  piotroskiFScore: number | null;
  promoterHolding: number | null;
  fiiHolding: number | null;
  diiHolding: number | null;
  publicHolding: number | null;

  sharesOutstanding: number | null;
  sector: string;
}

export interface CategoryScore {
  name: string;
  score: number;
  weight: number;
  metrics: { name: string; value: number | null; maxScore: number }[];
}

export interface FundamentalScore {
  total: number;
  rating: string;
  ratingColor: string;
  categories: CategoryScore[];
}

export interface FundamentalsResponse {
  fundamentals: Fundamentals;
  score: FundamentalScore;
}
