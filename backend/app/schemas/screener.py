from typing import Optional
from app.schemas.base import BaseSchema


class ScreenerCriteria(BaseSchema):
    min_roe: Optional[float] = None
    max_roe: Optional[float] = None
    min_de: Optional[float] = None
    max_de: Optional[float] = None
    min_revenue_growth: Optional[float] = None
    max_revenue_growth: Optional[float] = None
    min_profit_growth: Optional[float] = None
    max_profit_growth: Optional[float] = None
    min_opm: Optional[float] = None
    max_pe: Optional[float] = None
    sector: Optional[str] = None


class ScreenerResult(BaseSchema):
    symbol: str
    name: str
    price: float
    pe: Optional[float]
    roe: Optional[float]
    de: Optional[float]
    revenue_growth: Optional[float]
    profit_growth: Optional[float]
    score: float


class ScreenerResponse(BaseSchema):
    results: list[ScreenerResult]
    total: int
