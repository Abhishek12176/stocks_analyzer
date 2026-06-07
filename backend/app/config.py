from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    app_name: str = "EquityLens API"
    debug: bool = False
    port: int = 8000

    # CORS - comma-separated list or JSON array
    cors_origins: List[str] = ["http://localhost:3000"]

    # API Key for backend-to-frontend auth (optional)
    api_key: str | None = None

    # External API Keys
    newsdata_api_key: str = ""

    # Rate limiting
    rate_limit_per_minute: int = 60

    # Cache TTLs (seconds)
    cache_ttl_price: int = 60
    cache_ttl_fundamentals: int = 3600
    cache_ttl_shareholding: int = 86400
    cache_ttl_news: int = 900

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


settings = Settings()
