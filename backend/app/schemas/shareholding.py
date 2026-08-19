from typing import Optional
from app.schemas.base import BaseSchema


class QuarterlyHolding(BaseSchema):
    quarter: str
    promoter: str
    fii: str
    dii: str
    government: str
    public: str


class LatestHolding(BaseSchema):
    promoter: str
    fii: str
    dii: str
    government: str
    public: str


class MajorShareholder(BaseSchema):
    rank: int
    name: str
    holding_percent: float


class ShareholdingResponse(BaseSchema):
    quarterly_data: list[QuarterlyHolding]
    latest: LatestHolding
    major_shareholders: list[MajorShareholder]
    source: str
