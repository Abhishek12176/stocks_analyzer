import asyncio
import logging

from fastapi import APIRouter, Query
from app.services.yfinance_service import fetch_price_data, fetch_intraday_data, search_symbols
from app.services.fundamental_service import fundamentals_service
from app.services.signal_service import generate_trade_signal
from app.services.news_service import fetch_stock_news
from app.services.shareholding_service import fetch_shareholding_data
from app.utils.validators import clean_symbol, validate_symbol
from app.utils.exceptions import InvalidSymbolError
from app.schemas.stock import PriceResponse, SearchResponse, SymbolSearchResult, IntradayResponse
from app.schemas.fundamentals import FundamentalsResponse
from app.schemas.signal import SignalResponse
from app.schemas.news import NewsResponse, NewsArticle
from app.schemas.shareholding import ShareholdingResponse

logger = logging.getLogger("equitylens.stock")
_TIMEOUT = 30

router = APIRouter(prefix="/stock", tags=["stock"])


@router.get("/search", response_model=SearchResponse)
async def search_stocks(q: str = Query(..., min_length=1, max_length=100)):
    """Search for stock symbols by name or ticker."""
    results = search_symbols(q)
    return SearchResponse(
        results=[SymbolSearchResult(**r) for r in results]
    )


@router.get("/{symbol}/price", response_model=PriceResponse)
async def get_stock_price(symbol: str, period: str = Query("1y", pattern=r"^(1mo|3mo|6mo|1y|2y|5y|max)$")):
    """Get price history, current quote, and technical indicators."""
    clean = clean_symbol(symbol)
    if not validate_symbol(clean):
        raise InvalidSymbolError(symbol)
    return fetch_price_data(clean, period)


@router.get("/{symbol}/intraday", response_model=IntradayResponse)
async def get_stock_intraday(symbol: str, interval: str = Query("5m", pattern=r"^(1m|5m|10m|20m|30m|1h)$")):
    """Get intraday price data at the specified interval."""
    clean = clean_symbol(symbol)
    if not validate_symbol(clean):
        raise InvalidSymbolError(symbol)
    return fetch_intraday_data(clean, interval)


@router.get("/{symbol}/fundamentals", response_model=FundamentalsResponse)
async def get_stock_fundamentals(symbol: str):
    """Get fundamental data and multi-factor scoring."""
    clean = clean_symbol(symbol)
    if not validate_symbol(clean):
        raise InvalidSymbolError(symbol)
    f = fundamentals_service.get_fundamentals(clean, "NSE")

    def pct(val):
        return round(val * 100, 2) if val is not None else None

    fundamentals = {
        "market_cap": f.get("market_cap"),
        "pe_ratio": f.get("pe_ratio"),
        "eps": f.get("eps"),
        "book_value": None,
        "dividend_yield": None,
        "roe": pct(f.get("roe")),
        "roce": pct(f.get("roce")),
        "debt_equity": f.get("debt_to_equity"),
        "de_category": f.get("de_category", ""),
        "opm": pct(f.get("operating_margin")),
        "npm": None,
        "gross_margin": None,
        "current_ratio": None,
        "interest_coverage": None,
        "altman_z_score": None,
        "revenue_growth": None,
        "profit_growth": None,
        "eps_growth": None,
        "fcf_growth": None,
        "asset_turnover": None,
        "inventory_turnover": None,
        "receivables_days": None,
        "pe_vs_sector": None,
        "pb_ratio": None,
        "ev_ebitda": None,
        "piotroski_f_score": None,
        "promoter_holding": None,
        "fii_holding": None,
        "dii_holding": None,
        "public_holding": None,
        "shares_outstanding": None,
        "sector": f.get("sector", ""),
    }

    score_val = f.get("fundamental_score", 0) or 0
    rating = f.get("rating", "N/A") or "N/A"
    score_color = (
        "green" if score_val >= 65 else "gold" if score_val >= 45 else "red"
    )

    score_obj = {
        "total": score_val,
        "rating": rating,
        "rating_color": score_color,
        "categories": [],
    }

    return FundamentalsResponse(fundamentals=fundamentals, score=score_obj)


@router.get("/{symbol}/signal", response_model=SignalResponse)
async def get_stock_signal(symbol: str):
    """Get trade signal based on technical indicators."""
    clean = clean_symbol(symbol)
    if not validate_symbol(clean):
        raise InvalidSymbolError(symbol)
    price_data = fetch_price_data(clean)
    indicators = price_data["indicators"]
    signal = generate_trade_signal(
        price=price_data["quote"]["current_price"],
        rsi=indicators.get("rsi"),
        macd=indicators.get("macd"),
        signal=indicators.get("signal"),
        sma20=indicators.get("sma20"),
        sma50=indicators.get("sma50"),
    )
    signal["quote"] = {
        "price": price_data["quote"]["current_price"],
        "change": price_data["quote"]["change"],
        "change_percent": price_data["quote"]["change_percent"],
    }
    return signal


@router.get("/{symbol}/news", response_model=NewsResponse)
async def get_stock_news(
    symbol: str,
    limit: int = Query(30, ge=1, le=100),
):
    """Get latest news for a stock."""
    clean = clean_symbol(symbol)
    if not validate_symbol(clean):
        raise InvalidSymbolError(symbol)
    articles = fetch_stock_news(clean, limit)
    return NewsResponse(
        articles=[NewsArticle(**a) for a in articles],
        total=len(articles),
        source="NewsData.io",
    )


@router.get("/{symbol}/shareholding", response_model=ShareholdingResponse)
async def get_stock_shareholding(symbol: str):
    """Get shareholding pattern data."""
    clean = clean_symbol(symbol)
    if not validate_symbol(clean):
        raise InvalidSymbolError(symbol)
    data = fetch_shareholding_data(clean)
    if not data:
        raise InvalidSymbolError(symbol)
    return data


@router.get("/{symbol}", response_model=dict)
async def get_full_analysis(symbol: str):
    """Get a full analysis bundle for a stock — fetches price + fundamentals in parallel."""
    clean = clean_symbol(symbol)
    if not validate_symbol(clean):
        raise InvalidSymbolError(symbol)

    async def fetch_price():
        return await asyncio.to_thread(fetch_price_data, clean)

    async def fetch_fundamentals():
        return await asyncio.to_thread(fundamentals_service.get_fundamentals, clean, "NSE")

    price_data = None
    fundamentals_raw = None

    try:
        price_data, fundamentals_raw = await asyncio.wait_for(
            asyncio.gather(fetch_price(), fetch_fundamentals(), return_exceptions=True),
            timeout=_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("Full analysis timed out for %s", clean)

    if isinstance(price_data, Exception):
        logger.error("Price fetch failed for %s: %s", clean, price_data)
        price_data = None
    if isinstance(fundamentals_raw, Exception):
        logger.error("Fundamentals fetch failed for %s: %s", clean, fundamentals_raw)
        fundamentals_raw = None

    f = fundamentals_raw or {}

    def pct(val):
        return round(val * 100, 2) if val is not None else None

    fundamentals = {
        "market_cap": f.get("market_cap"),
        "pe_ratio": f.get("pe_ratio"),
        "eps": f.get("eps"),
        "book_value": None,
        "dividend_yield": None,
        "roe": pct(f.get("roe")),
        "roce": pct(f.get("roce")),
        "debt_equity": f.get("debt_to_equity"),
        "de_category": f.get("de_category", ""),
        "opm": pct(f.get("operating_margin")),
        "npm": None,
        "gross_margin": None,
        "current_ratio": None,
        "interest_coverage": None,
        "altman_z_score": None,
        "revenue_growth": None,
        "profit_growth": None,
        "eps_growth": None,
        "fcf_growth": None,
        "asset_turnover": None,
        "inventory_turnover": None,
        "receivables_days": None,
        "pe_vs_sector": None,
        "pb_ratio": None,
        "ev_ebitda": None,
        "piotroski_f_score": None,
        "promoter_holding": None,
        "fii_holding": None,
        "dii_holding": None,
        "public_holding": None,
        "shares_outstanding": None,
        "sector": f.get("sector", ""),
    }

    score_val = f.get("fundamental_score", 0) or 0
    rating = f.get("rating", "N/A") or "N/A"
    score_color = (
        "green" if score_val >= 65 else "gold" if score_val >= 45 else "red"
    )

    score = {
        "total": score_val,
        "rating": rating,
        "rating_color": score_color,
        "categories": [],
    }

    quote_data = (price_data or {}).get("quote", {})
    indicators_data = (price_data or {}).get("indicators", {})

    signal = generate_trade_signal(
        price=quote_data.get("current_price", 0),
        rsi=indicators_data.get("rsi"),
        macd=indicators_data.get("macd"),
        signal=indicators_data.get("signal"),
        sma20=indicators_data.get("sma20"),
        sma50=indicators_data.get("sma50"),
    )
    if quote_data:
        signal["quote"] = {
            "price": quote_data.get("current_price", 0),
            "change": quote_data.get("change", 0),
            "change_percent": quote_data.get("change_percent", 0),
        }

    return {
        "quote": quote_data,
        "indicators": indicators_data,
        "fundamentals": fundamentals,
        "score": score,
        "signal": signal["signal"],
    }
