from typing import Optional
from app.schemas.base import BaseSchema


class SignalReason(BaseSchema):
    factor: str
    impact: str  # bullish, bearish, neutral
    detail: str


class TradeSignal(BaseSchema):
    action: str  # BUY, SELL, HOLD, NEUTRAL
    direction: str  # bullish, bearish, neutral
    confidence: float
    reasons: list[SignalReason]
    generated_at: str


class QuoteSummary(BaseSchema):
    price: float
    change: float
    change_percent: float


class SignalResponse(BaseSchema):
    signal: TradeSignal
    quote: QuoteSummary
