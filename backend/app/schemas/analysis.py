from typing import Optional
from app.schemas.base import BaseSchema
from app.schemas.stock import StockQuote, TechnicalIndicators
from app.schemas.fundamentals import Fundamentals, FundamentalScore
from app.schemas.signal import TradeSignal
from app.schemas.forecast import ForecastResponse


class FullAnalysisResponse(BaseSchema):
    quote: StockQuote
    indicators: TechnicalIndicators
    fundamentals: Fundamentals
    score: FundamentalScore
    signal: TradeSignal
    forecast: Optional[ForecastResponse] = None
