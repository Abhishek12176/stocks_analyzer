from typing import Optional
from app.schemas.base import BaseSchema


class Fundamentals(BaseSchema):
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    eps: Optional[float] = None
    book_value: Optional[float] = None
    dividend_yield: Optional[float] = None
    roe: Optional[float] = None
    roce: Optional[float] = None
    debt_equity: Optional[float] = None
    de_category: str = ""
    opm: Optional[float] = None
    npm: Optional[float] = None
    gross_margin: Optional[float] = None
    current_ratio: Optional[float] = None
    interest_coverage: Optional[float] = None
    altman_z_score: Optional[float] = None
    revenue_growth: Optional[float] = None
    profit_growth: Optional[float] = None
    eps_growth: Optional[float] = None
    fcf_growth: Optional[float] = None
    asset_turnover: Optional[float] = None
    inventory_turnover: Optional[float] = None
    receivables_days: Optional[float] = None
    pe_vs_sector: Optional[float] = None
    pb_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    piotroski_f_score: Optional[int] = None
    promoter_holding: Optional[float] = None
    fii_holding: Optional[float] = None
    dii_holding: Optional[float] = None
    public_holding: Optional[float] = None
    shares_outstanding: Optional[int] = None
    sector: str = ""


class CategoryMetric(BaseSchema):
    name: str
    value: Optional[float] = None
    max_score: float


class CategoryScore(BaseSchema):
    name: str
    score: float
    weight: float
    metrics: list[CategoryMetric]


class FundamentalScore(BaseSchema):
    total: float
    rating: str
    rating_color: str
    categories: list[CategoryScore]


class FundamentalsResponse(BaseSchema):
    fundamentals: Fundamentals
    score: FundamentalScore
