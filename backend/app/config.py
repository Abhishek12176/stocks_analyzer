import json

from pydantic import field_validator
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    app_name: str = "EquityLens API"
    debug: bool = False
    port: int = 8000

    # CORS origins: accepts JSON array string or comma-separated string
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

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | List[str]) -> List[str]:
        if isinstance(v, list):
            return v
        if not v:
            return ["http://localhost:3000"]
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        return [o.strip() for o in v.split(",") if o.strip()]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
        "validate_default": True,
    }


settings = Settings()
