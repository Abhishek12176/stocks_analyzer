from typing import Optional
from app.schemas.base import BaseSchema


class StockPrice(BaseSchema):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class TechnicalIndicators(BaseSchema):
    rsi: Optional[float] = None
    macd: Optional[float] = None
    signal: Optional[float] = None
    sma20: Optional[float] = None
    sma50: Optional[float] = None


class StockQuote(BaseSchema):
    symbol: str
    company_name: str
    exchange: str
    current_price: float
    change: float
    change_percent: float
    last_updated: str


class PriceResponse(BaseSchema):
    quote: StockQuote
    history: list[StockPrice]
    indicators: TechnicalIndicators


class SymbolSearchResult(BaseSchema):
    symbol: str
    name: str
    exchange: str


class SearchResponse(BaseSchema):
    results: list[SymbolSearchResult]


class IntradayPrice(BaseSchema):
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class IntradayResponse(BaseSchema):
    symbol: str
    company_name: str
    exchange: str
    current_price: float
    change: float
    change_percent: float
    last_updated: str
    interval: str
    history: list[IntradayPrice]
