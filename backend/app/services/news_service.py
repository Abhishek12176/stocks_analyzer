import logging

import requests
from datetime import datetime, timezone
from app.config import settings
from app.services.cache_service import cache_service
from app.services.sentiment_service import analyze_articles

logger = logging.getLogger(__name__)


def clean_symbol(symbol: str) -> str:
    return (
        symbol.upper()
        .replace(".NS", "")
        .replace(".NSE", "")
        .replace(".BO", "")
        .replace(".BSE", "")
        .strip()
    )


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        try:
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return None


def normalize_article(item: dict) -> dict:
    published_dt = parse_date(item.get("pubDate"))
    return {
        "title": item.get("title") or "No title",
        "summary": item.get("description") or "",
        "link": item.get("link") or "",
        "source": item.get("source_name") or "NewsData.io",
        "published_dt": published_dt.isoformat() if published_dt else None,
        "published": (
            published_dt.strftime("%d %b %Y, %I:%M %p") if published_dt else "Latest"
        ),
    }


def fetch_stock_news(symbol: str, limit: int = 30) -> list[dict]:
    """Fetch news articles for a stock symbol from NewsData.io."""
    cache_key = f"news_{symbol}_{limit}"
    cached = cache_service.get(cache_service.news_cache, cache_key)
    if cached:
        return cached

    if not settings.newsdata_api_key:
        return []

    clean_name = clean_symbol(symbol)
    search_query = f"{clean_name} stock"

    all_news = []
    seen_links = set()
    next_page = None

    while len(all_news) < limit:
        params = {
            "apikey": settings.newsdata_api_key,
            "q": search_query,
            "language": "en",
            "country": "in",
            "size": 10,
        }
        if next_page:
            params["page"] = next_page

        try:
            response = requests.get(
                "https://newsdata.io/api/1/news",
                params=params,
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.warning("NewsData.io API error for %s: %s", symbol, exc)
            break

        if data.get("status") != "success":
            break

        articles = data.get("results", [])
        if not articles:
            break

        for item in articles:
            article = normalize_article(item)

            if not article["link"] or article["link"] in seen_links:
                continue

            seen_links.add(article["link"])
            all_news.append(article)

            if len(all_news) >= limit:
                break

        next_page = data.get("nextPage")
        if not next_page:
            break

    all_news.sort(
        key=lambda a: a["published_dt"] or datetime.min.replace(tzinfo=timezone.utc).isoformat(),
        reverse=True,
    )

    all_news = analyze_articles(all_news)

    return all_news[:limit]
