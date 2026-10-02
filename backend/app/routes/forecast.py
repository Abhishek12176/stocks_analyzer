"""Forecast route — probabilistic 20-day forecast + explanation + backtest (Task 10)."""

import asyncio
import logging

from fastapi import APIRouter, Query

from app.ml.pipeline import forecast_symbol, _graceful
from app.schemas.forecast import ForecastResponse
from app.utils.validators import clean_symbol, validate_symbol
from app.utils.exceptions import InvalidSymbolError

logger = logging.getLogger("equitylens.forecast")
# A cold 10y forecast = 10y network pull + walk-forward ensemble training.
# Measured: 75s compute on a 2476-row frame (fast=True) plus ~60-150s of data
# fetch, so a 4-minute cap silently killed most first-ever runs ("timed out").
_TIMEOUT = 600

router = APIRouter(prefix="/stock", tags=["forecast"])


@router.get("/{symbol}/forecast", response_model=ForecastResponse)
async def get_stock_forecast(
    symbol: str,
    fast: bool = Query(True, description="lighter walk-forward for API latency"),
    enrich: bool = Query(False, description="include sentiment/macro/options enrichments"),
):
    """Probabilistic 20-day forecast for a stock.

    Builds the full PIT feature frame (no lookahead), walks forward out of
    sample, calibrates the ensemble P(up), applies the BUY/HOLD/SELL policy and
    returns explanation + backtest performance. Unavailable data/model paths
    return `is_available: false` — never a fabricated forecast.
    """
    clean = clean_symbol(symbol)
    if not validate_symbol(clean):
        raise InvalidSymbolError(symbol)

    period = "5y" if fast else "10y"
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                forecast_symbol, clean, period=period, fast=bool(fast), enrich=bool(enrich)
            ),
            timeout=_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("forecast timed out for %s", clean)
        result = _graceful(clean, "forecast computation timed out")
    return ForecastResponse(**result)


DEFAULT_WARM_SYMBOLS = [
    "TCS", "RELIANCE", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "ITC", "BHARTIARTL"
]


@router.post("/warm-cache")
async def warm_forecast_cache(symbols: list[str] | None = None):
    """Daily cron endpoint to pre-calculate and warm forecast cache for major stocks."""
    target_symbols = [clean_symbol(s) for s in (symbols or DEFAULT_WARM_SYMBOLS)]
    results = {}
    for s in target_symbols:
        if not validate_symbol(s):
            continue
        try:
            res = await asyncio.wait_for(
                asyncio.to_thread(forecast_symbol, s, period="5y", fast=True, enrich=False),
                timeout=120,
            )
            results[s] = {
                "ok": res.get("is_available", False),
                "signal": res.get("latest", {}).get("signal"),
                "as_of": res.get("latest", {}).get("as_of"),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("cron warm-cache failed for %s: %s", s, exc)
            results[s] = {"ok": False, "error": str(exc)}

    return {
        "status": "completed",
        "count": len(results),
        "results": results,
    }