from typing import Optional
from app.schemas.base import BaseSchema


class NewsArticle(BaseSchema):
    title: str
    summary: str
    link: str
    source: str
    published: str
    published_dt: Optional[str] = None


class NewsResponse(BaseSchema):
    articles: list[NewsArticle]
    total: int
    source: str
