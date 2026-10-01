import type { MetadataRoute } from "next";

const getBaseUrl = (): string => {
  if (process.env.NEXT_PUBLIC_SITE_URL) {
    return process.env.NEXT_PUBLIC_SITE_URL.replace(/\/$/, "");
  }
  if (process.env.VERCEL_PROJECT_PRODUCTION_URL) {
    return `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}`;
  }
  if (process.env.VERCEL_URL) {
    return `https://${process.env.VERCEL_URL}`;
  }
  return "https://equitylens.vercel.app";
};

export default function sitemap(): MetadataRoute.Sitemap {
  const baseUrl = getBaseUrl();
  const currentDate = new Date();

  // Core static pages
  const routes: MetadataRoute.Sitemap = [
    {
      url: `${baseUrl}`,
      lastModified: currentDate,
      changeFrequency: "daily",
      priority: 1.0,
    },
    {
      url: `${baseUrl}/markets/stocks`,
      lastModified: currentDate,
      changeFrequency: "daily",
      priority: 0.8,
    },
    {
      url: `${baseUrl}/signals`,
      lastModified: currentDate,
      changeFrequency: "daily",
      priority: 0.8,
    },
    {
      url: `${baseUrl}/compare`,
      lastModified: currentDate,
      changeFrequency: "weekly",
      priority: 0.7,
    },
    {
      url: `${baseUrl}/watchlist`,
      lastModified: currentDate,
      changeFrequency: "weekly",
      priority: 0.6,
    },
    {
      url: `${baseUrl}/basket`,
      lastModified: currentDate,
      changeFrequency: "weekly",
      priority: 0.6,
    },
    {
      url: `${baseUrl}/history`,
      lastModified: currentDate,
      changeFrequency: "weekly",
      priority: 0.5,
    },
  ];

  // Top popular Indian stocks for indexing
  const popularSymbols = [
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "INFY",
    "ICICIBANK",
    "TATAMOTORS",
    "SBIN",
    "BHARTIARTL",
    "ITC",
    "LT",
    "HINDUNILVR",
    "KOTAKBANK",
    "WIPRO",
    "BAJFINANCE",
    "MARUTI",
  ];

  const stockRoutes: MetadataRoute.Sitemap = popularSymbols.map((symbol) => ({
    url: `${baseUrl}/stock/${symbol}`,
    lastModified: currentDate,
    changeFrequency: "daily",
    priority: 0.7,
  }));

  return [...routes, ...stockRoutes];
}
