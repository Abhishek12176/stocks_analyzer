import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
import pandas_ta as ta
import yfinance as yf

from app.services.signal_service import generate_trade_signal
from app.services.cache_service import cache_service
from app.utils.validators import add_exchange_suffix

logger = logging.getLogger("equitylens.signals")

ALL_SYMBOLS = [
    "INFY", "TCS", "HCLTECH", "WIPRO", "TECHM", "LTTS", "PERSISTENT", "COFORGE", "MINDTREE", "MPHASIS",
    "RELIANCE", "BHARTIARTL", "LT", "TITAN", "HAL", "ADANIENT", "DLF", "SIEMENS", "BEL",
    "TATASTEEL", "JSWSTEEL", "HINDALCO", "ADANIPORTS", "NTPC",
    "HDFCBANK", "ICICIBANK", "KOTAKBANK", "ITC", "HINDUNILVR", "NESTLEIND", "ASIANPAINT",
    "ULTRACEMCO", "BAJFINANCE", "MARUTI", "EICHERMOT", "DRREDDY", "SUNPHARMA",
    "TATACONSUM", "BRITANNIA",
    "SBIN", "AXISBANK", "INDUSINDBK", "BANKBARODA", "PNB", "CANBK", "FEDERALBNK", "IDFCFIRSTB", "RBLBANK",
    "LTIM",
    "TATAMOTORS", "HEROMOTOCO", "ASHOKLEY", "BALKRISIND", "TVSMOTOR", "MOTHERSUMI",
    "CIPLA", "DIVISLAB", "AUROPHARMA", "LUPIN", "BIOCON", "TORNTPHARM", "ALKEM", "PFIZER", "CADILAHC",
    "ONGC", "IOC", "BPCL", "GAIL", "POWERGRID", "ADANIGREEN", "TATAPOWER", "COALINDIA", "HINDPETRO",
]

CATEGORY_ORDER = [
    "strong-buy", "buy", "hold", "sell", "strong-sell",
    "rsi-oversold", "rsi-overbought",
    "macd-bullish", "macd-bearish",
    "bullish-trend", "bearish-trend",
]

CATEGORY_LABELS = {
    "strong-buy": "Strong Buy",
    "buy": "Buy",
    "hold": "Hold",
    "sell": "Sell",
    "strong-sell": "Strong Sell",
    "rsi-oversold": "RSI Oversold",
    "rsi-overbought": "RSI Overbought",
    "macd-bullish": "MACD Bullish Crossover",
    "macd-bearish": "MACD Bearish Crossover",
    "bullish-trend": "Bullish Trend",
    "bearish-trend": "Bearish Trend",
}


def _fetch_one(symbol: str) -> dict | None:
    try:
        ticker = add_exchange_suffix(symbol)
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")

        if hist.empty:
            return None

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if isinstance(hist[col], pd.DataFrame):
                hist[col] = hist[col].iloc[:, 0]
            hist[col] = pd.to_numeric(hist[col], errors="coerce")

        hist.dropna(inplace=True)
        if hist.empty:
            return None

        latest = hist.iloc[-1]
        current_price = float(latest["Close"])
        prev_close = float(hist.iloc[-2]["Close"]) if len(hist) > 1 else current_price
        change = current_price - prev_close
        change_percent = (change / prev_close) * 100

        rsi_val = float(ta.rsi(hist["Close"], length=14).iloc[-1]) if len(hist) >= 15 else None
        macd_df = ta.macd(hist["Close"], fast=12, slow=26, signal=9) if len(hist) >= 35 else None
        macd_val = float(macd_df["MACD_12_26_9"].iloc[-1]) if macd_df is not None else None
        sig_val = float(macd_df["MACDs_12_26_9"].iloc[-1]) if macd_df is not None else None
        sma20 = float(ta.sma(hist["Close"], length=20).iloc[-1]) if len(hist) >= 20 else None
        sma50 = float(ta.sma(hist["Close"], length=50).iloc[-1]) if len(hist) >= 50 else None

        sig = generate_trade_signal(
            price=current_price,
            rsi=rsi_val,
            macd=macd_val,
            signal=sig_val,
            sma20=sma20,
            sma50=sma50,
        )

        sparkline = [round(h, 2) for h in hist["Close"].tail(30).tolist()]
        sig_obj = sig["signal"]
        is_strong_trend = (sma20 or 0) > (sma50 or 0)

        return {
            "symbol": symbol.upper(),
            "name": symbol.upper(),
            "currentPrice": round(current_price, 2),
            "change": round(change, 2),
            "changePercent": round(change_percent, 2),
            "signal": sig_obj["action"],
            "confidence": sig_obj["confidence"],
            "rsi": rsi_val,
            "macd": macd_val,
            "macdSignal": sig_val,
            "sma20": sma20,
            "sma50": sma50,
            "trend": "Strong Trend" if is_strong_trend else "Weak Trend",
            "sparkline": sparkline,
        }
    except Exception as exc:
        logger.debug("Signal scan failed for %s: %s", symbol, exc)
        return None


def _categorize(stock: dict) -> list[str]:
    cats = []
    is_strong_trend = stock["trend"] == "Strong Trend"

    if stock["signal"] == "BUY" and stock["confidence"] >= 85 and is_strong_trend:
        cats.append("strong-buy")
    if stock["signal"] == "BUY" and stock["confidence"] >= 70:
        cats.append("buy")
    if stock["signal"] == "HOLD":
        cats.append("hold")
    if stock["signal"] == "SELL":
        cats.append("sell")
    if stock["signal"] == "SELL" and stock["confidence"] >= 85:
        cats.append("strong-sell")
    if stock["rsi"] is not None and stock["rsi"] < 30:
        cats.append("rsi-oversold")
    if stock["rsi"] is not None and stock["rsi"] > 70:
        cats.append("rsi-overbought")
    if stock["macd"] is not None and stock["macdSignal"] is not None:
        if stock["macd"] > stock["macdSignal"]:
            cats.append("macd-bullish")
        if stock["macd"] < stock["macdSignal"]:
            cats.append("macd-bearish")
    if stock["sma20"] is not None and stock["sma50"] is not None:
        if stock["sma20"] > stock["sma50"]:
            cats.append("bullish-trend")
        if stock["sma20"] < stock["sma50"]:
            cats.append("bearish-trend")
    return cats


SIGNALS_CACHE_TTL = 600  # 10 minutes — expensive to regenerate


def get_all_signals() -> dict:
    cache_key = "signals_all"
    cached = cache_service.get(cache_service.price_cache, cache_key)
    if cached:
        return cached

    t0 = time.time()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(_fetch_one, sym): sym for sym in ALL_SYMBOLS}
        for future in as_completed(futures):
            try:
                result = future.result(timeout=20)
                if result:
                    results.append(result)
            except Exception as exc:
                logger.debug("Signal scan timeout: %s", exc)

    categories = []
    for cat_id in CATEGORY_ORDER:
        stocks = [s for s in results if cat_id in _categorize(s)]
        stocks.sort(key=lambda x: x["confidence"], reverse=True)
        categories.append({
            "id": cat_id,
            "label": CATEGORY_LABELS[cat_id],
            "stocks": stocks,
            "count": len(stocks),
        })

    response = {
        "categories": categories,
        "totalStocks": len(results),
        "generatedAt": datetime.now(timezone.utc).isoformat(),
    }

    try:
        cache_service.set(cache_service.price_cache, cache_key, response)
    except Exception:
        pass
    logger.info("Signals scan complete: %d stocks processed in %.1fs", len(results), time.time() - t0)
    return response
