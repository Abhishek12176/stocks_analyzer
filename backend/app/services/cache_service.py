from cachetools import TTLCache
from app.config import settings


class CacheService:
    def __init__(self):
        self.price_cache = TTLCache(
            maxsize=500, ttl=settings.cache_ttl_price
        )
        self.fundamentals_cache = TTLCache(
            maxsize=200, ttl=settings.cache_ttl_fundamentals
        )
        self.shareholding_cache = TTLCache(
            maxsize=200, ttl=settings.cache_ttl_shareholding
        )
        self.news_cache = TTLCache(
            maxsize=100, ttl=settings.cache_ttl_news
        )

    def get(self, cache: TTLCache, key: str):
        return cache.get(key)

    def set(self, cache: TTLCache, key: str, value):
        cache[key] = value

    def clear_all(self):
        self.price_cache.clear()
        self.fundamentals_cache.clear()
        self.shareholding_cache.clear()
        self.news_cache.clear()


cache_service = CacheService()
