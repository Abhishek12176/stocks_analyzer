from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "EquityLens API"
    debug: bool = False
    port: int = 8000

    cors_origins_str: str = '["http://localhost:3000"]'

    api_key: str | None = None
    newsdata_api_key: str = ""
    rate_limit_per_minute: int = 60
    cache_ttl_price: int = 180
    cache_ttl_fundamentals: int = 3600
    cache_ttl_shareholding: int = 86400
    cache_ttl_news: int = 900

    @property
    def cors_origins(self) -> list[str]:
        import json
        try:
            return json.loads(self.cors_origins_str)
        except Exception:
            return [o.strip() for o in self.cors_origins_str.split(",") if o.strip()]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
        "validate_default": True,
    }


settings = Settings()
