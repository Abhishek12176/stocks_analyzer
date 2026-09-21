"""Forecast route — probabilistic 20-day forecast + explanation + backtest (Task 10)."""

import asyncio
import logging

from fastapi import APIRouter, Query

from app.ml.pipeline import forecast_symbol, _graceful
from app.schemas.forecast import ForecastResponse
from app.utils.validators import clean_symbol, validate_symbol
from app.utils.exceptions import InvalidSymbolError

logger = logging.getLogger("equitylens.forecast")
_TIMEOUT = 240

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

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                forecast_symbol, clean, period="10y", fast=bool(fast), enrich=bool(enrich)
            ),
            timeout=_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("forecast timed out for %s", clean)
        result = _graceful(clean, "forecast computation timed out")
    except Exception as exc:  # noqa: BLE001
        logger.exception("forecast failed for %s", clean)
        result = _graceful(clean, f"{type(exc).__name__}: {exc}")

    return ForecastResponse(**result)