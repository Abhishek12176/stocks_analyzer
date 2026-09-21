from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "EquityLens API"
    debug: bool = False
    port: int = 8000

    cors_origins_str: str = '["http://localhost:3000"]'

    api_key: str | None = None
    newsdata_api_key: str = ""

    # LLM (OpenAI-compatible) settings for the AI Chat Assistant
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_max_tokens: int = 1000
    openai_temperature: float = 0.4
    chat_fallback_enabled: bool = True

    # Chat enrichment: max symbols deep-dived per message + news headlines per symbol.
    chat_max_enrich_symbols: int = 2
    chat_news_headlines: int = 3

    rate_limit_per_minute: int = 60
    cache_ttl_price: int = 180
    cache_ttl_fundamentals: int = 3600
    cache_ttl_shareholding: int = 86400
    cache_ttl_news: int = 900
    cache_ttl_market: int = 1800

    # ML / forecasting system settings
    ml_feature_version: str = "v1"
    ml_model_version: str = "baseline-v1"
    ml_horizon: int = 20
    ml_training_period: str = "PRE_TRAINING"
    ml_forecast_feature_start: str = "2016-01-01"

    # Realistic cost model for backtesting (per side).
    ml_cost_brokerage_inr: float = 20.0
    ml_cost_fee_percent: float = 0.01
    ml_cost_slippage_bps: float = 10.0
    ml_max_position_price: float = 0.0  # 0 = no per-trade price cap

    # F&O / options layer (Task 7).
    ml_risk_free_rate: float = 0.06  # risk-free rate used in Black-Scholes greeks
    ml_options_enabled: bool = True  # master toggle; off = graceful disable

    # Signal policy + ensemble (Task 9).
    ml_threshold_buy: float = 0.6   # P(up) >= buy  -> BUY
    ml_threshold_sell: float = 0.4  # P(up) <= sell -> SELL (short if allowed)
    # Signal-quality gate: below this full-OOF ensemble AUC, no confident signal.
    ml_min_auc_for_signal: float = 0.55
    ml_ensemble_version: str = "ensemble-v1"
    # Task 31 (19-Sep-2026, ADOPTED): per-symbol high-conviction BUY thresholds
    # tuned on the older 80% OOF-fit rows, evaluated on the untouched newest-20%
    # holdout of a same-day control10y snapshot (as_of 2026-09-18) then confirmed
    # on an independently re-fetched 10y frame. Median holdout BUY hit-rate
    # 50.9% -> 57.1%; formal-ledger median cumRet -20.06% -> -4.97% (loss ~75%
    # cut) at unchanged AUC 0.6166. Symbols absent here keep ml_threshold_buy.
    # Residual (documented): confirmation on a genuinely NEW trading window is
    # pending the next session (market closed at adoption). Revert: {} .
    ml_per_symbol_buy_thresholds: dict[str, float] = {
        "TCS": 0.66, "RELIANCE": 0.72, "HDFCBANK": 0.64, "INFY": 0.72,
    }

    # Monitoring / drift (Task 9).
    ml_psi_threshold: float = 0.25      # population-stability-index alarm
    ml_freshness_max_days: int = 7      # calendar days before data is "stale"
    ml_brier_drift_threshold: float = 0.05  # Brier deterioration before alarm

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
