export interface NewsArticle {
  title: string;
  summary: string;
  link: string;
  source: string;
  published: string;
  publishedDt: string | null;
}

export interface NewsResponse {
  articles: NewsArticle[];
  total: number;
  source: string;
}
