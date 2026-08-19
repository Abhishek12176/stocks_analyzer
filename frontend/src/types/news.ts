export interface SentimentScore {
  label: "positive" | "negative" | "neutral";
  score: number;
}

export interface NewsArticle {
  title: string;
  summary: string;
  link: string;
  source: string;
  published: string;
  publishedDt: string | null;
  sentiment?: SentimentScore | null;
}

export interface NewsResponse {
  articles: NewsArticle[];
  total: number;
  source: string;
}
