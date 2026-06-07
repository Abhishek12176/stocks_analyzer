from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.services.yfinance_service import fetch_price_data
from app.services.signal_service import generate_trade_signal
from app.services.cache_service import cache_service


STOCK_WATCH = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "ITC", "SBIN",
    "BHARTIARTL", "KOTAKBANK", "BAJFINANCE", "LT", "WIPRO", "AXISBANK",
    "TITAN", "MARUTI", "SUNPHARMA", "HCLTECH", "ASIANPAINT", "DMART",
    "NTPC", "POWERGRID", "ULTRACEMCO", "TRENT", "ADANIENT", "ADANIPORTS",
    "BAJAJFINSV", "HINDUNILVR", "NESTLEIND", "ONGC", "COALINDIA",
    "JSWSTEEL", "TATASTEEL", "M&M", "EICHERMOT", "BRITANNIA",
    "HINDALCO", "DIVISLAB", "DRREDDY", "CIPLA", "TATACONSUM",
    "BAJAJ-AUTO", "HEROMOTOCO", "GRASIM", "TECHM", "APOLLOHOSP",
    "BEL", "HAL", "VEDL", "BPCL", "IOC",
    "PIDILITIND", "COLPAL", "MARICO", "DABUR", "HAVELLS",
    "SIEMENS", "BOSCHLTD", "ABB", "AMBER", "VOLTAS",
    "INDIGO", "ZOMATO", "PAYTM", "NAUKRI", "INFOLLION",
    "POLYCAB", "TORNTPOWER", "SUPREMEIND", "ASTRAZEN", "PFIZER",
    "LUPIN", "GLAND", "BIOCON", "LAURUSLABS", "AUROPHARMA",
    "TVSMOTOR", "ASHOKLEY", "BALKRISIND", "APOLLOTYRE", "MRF",
    "SRTRANSFIN", "PEL", "MUTHOOTFIN", "CHOLAFIN", "TV18BRDCST",
    "ZEEL", "PVRINOX", "INOXLEISUR", "DIXON", "BLUESTARCO",
    "ANGELONE", "ICICIPRULI", "HDFCLIFE", "SBILIFE", "LICI",
    "IRCTC", "INDIAMART", "JUBLFOOD", "RESTAURANT", "VBL",
]


def compute_signal_score(signal: dict) -> float:
    direction = signal.get("direction", "neutral")
    confidence = signal.get("confidence", 50)
    if direction == "bullish":
        return confidence
    elif direction == "bearish":
        return -confidence
    return 0


def fetch_stock_with_signal(symbol: str) -> dict | None:
    try:
        price_data = fetch_price_data(symbol, period="3mo")
        quote = price_data["quote"]
        indicators = price_data["indicators"]
        history = price_data["history"]

        signal = generate_trade_signal(
            price=quote["current_price"],
            rsi=indicators.get("rsi"),
            macd=indicators.get("macd"),
            signal=indicators.get("signal"),
            sma20=indicators.get("sma20"),
            sma50=indicators.get("sma50"),
        )

        sparkline_prices = [round(h["close"], 2) for h in history[-30:]]

        return {
            "symbol": quote["symbol"],
            "companyName": quote["company_name"],
            "exchange": quote["exchange"],
            "currentPrice": quote["current_price"],
            "change": quote["change"],
            "changePercent": quote["change_percent"],
            "signal": signal["signal"],
            "sparkline": sparkline_prices,
            "score": compute_signal_score(signal["signal"]),
        }
    except Exception:
        return None


def get_market_overview() -> dict:
    cache_key = "market_overview"
    cached = cache_service.get(cache_service.price_cache, cache_key)
    if cached:
        return cached

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_stock_with_signal, sym): sym for sym in STOCK_WATCH}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    bullish = sorted(
        [r for r in results if r["score"] > 0],
        key=lambda x: x["score"],
        reverse=True,
    )[:10]

    bearish = sorted(
        [r for r in results if r["score"] < 0],
        key=lambda x: x["score"],
    )[:10]

    response = {
        "bullish": bullish,
        "bearish": bearish,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "totalScanned": len(results),
    }

    cache_service.set(cache_service.price_cache, cache_key, response)
    return response
