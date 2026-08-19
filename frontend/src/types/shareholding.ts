export interface QuarterlyHolding {
  quarter: string;
  promoter: string;
  fii: string;
  dii: string;
  government: string;
  public: string;
}

export interface MajorShareholder {
  rank: number;
  name: string;
  holdingPercent: number;
}

export interface ShareholdingResponse {
  quarterlyData: QuarterlyHolding[];
  latest: {
    promoter: string;
    fii: string;
    dii: string;
    government: string;
    public: string;
  };
  majorShareholders: MajorShareholder[];
  source: string;
}
