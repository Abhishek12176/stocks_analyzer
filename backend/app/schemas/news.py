from typing import Optional
from app.schemas.base import BaseSchema


class SentimentScore(BaseSchema):
    label: str  # positive, negative, neutral
    score: float


class NewsArticle(BaseSchema):
    title: str
    summary: str
    link: str
    source: str
    published: str
    published_dt: Optional[str] = None
    sentiment: Optional[SentimentScore] = None


class NewsResponse(BaseSchema):
    articles: list[NewsArticle]
    total: int
    source: str
