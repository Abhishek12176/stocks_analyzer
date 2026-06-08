from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.yfinance_service import fetch_price_data
from app.services.fundamental_service import fundamentals_service
from app.utils.validators import clean_symbol, validate_symbol
from app.utils.exceptions import InvalidSymbolError

router = APIRouter(prefix="/compare", tags=["compare"])


class CompareRequest(BaseModel):
    symbols: list[str]


@router.post("/")
async def compare_stocks(req: CompareRequest):
    """Compare multiple stocks side-by-side."""
    if len(req.symbols) < 2:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "At least 2 symbols required",
                "code": "INVALID_INPUT",
            },
        )
    if len(req.symbols) > 5:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Maximum 5 symbols allowed",
                "code": "INVALID_INPUT",
            },
        )

    results = []
    for symbol in req.symbols:
        clean = clean_symbol(symbol)
        if not validate_symbol(clean):
            continue

        try:
            price_data = fetch_price_data(clean)
            f = fundamentals_service.get_fundamentals(clean, "NSE")

            results.append(
                {
                    "symbol": clean,
                    "name": price_data["quote"]["company_name"],
                    "metrics": {
                        "price": price_data["quote"]["current_price"],
                        "pe": f.get("pe_ratio"),
                        "roe": f.get("roe"),
                        "de": f.get("debt_to_equity"),
                        "opm": f.get("operating_margin"),
                        "revenue_growth": round(f.get("revenue_growth") * 100, 2) if f.get("revenue_growth") is not None else None,
                        "profit_growth": round(f.get("profit_growth") * 100, 2) if f.get("profit_growth") is not None else None,
                        "score": f.get("fundamental_score", 0) or 0,
                    },
                }
            )
        except Exception:
            continue

    return {"stocks": results}
