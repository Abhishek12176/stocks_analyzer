from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "EquityLens API"
    debug: bool = False
    cors_origins: list[str] = ["http://localhost:3000"]
    api_key: str | None = None

    # API Keys for external services
    newsdata_api_key: str = ""

    # Rate limiting
    rate_limit_per_minute: int = 60

    # Cache TTLs (seconds)
    cache_ttl_price: int = 60
    cache_ttl_fundamentals: int = 3600
    cache_ttl_shareholding: int = 86400
    cache_ttl_news: int = 900

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
