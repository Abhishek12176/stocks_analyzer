from fastapi import APIRouter
from app.services.market_service import get_market_overview

router = APIRouter(prefix="/market", tags=["market"])


@router.get("/overview")
async def market_overview():
    """Get top 10 bullish and top 10 bearish stocks from the watchlist."""
    return get_market_overview()
