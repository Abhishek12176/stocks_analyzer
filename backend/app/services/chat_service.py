"""AI Chat Assistant service.

Pipeline:
  1. Route the message (smalltalk / specific-stock deep dive / screening).
  2. extract_symbols()  -> known NSE symbols/companies named in the message.
     parse_intent()     -> structured filters (price range, action, top-N).
  3. load predictions   -> ONLY from the existing technical-indicator model via
                           signals_service.get_all_signals() (never new forecasts).
  4. enrich mentioned symbols with REAL data (quote, indicators, trade signal,
     fundamentals, news headlines, ML forecast) so answers are dense & accurate.
  5. answer             -> if OPENAI_API_KEY is set, an OpenAI-compatible LLM
                           analyses the full DATA payload (strictly forbidden from
                           inventing). Otherwise a deterministic template builds
                           a structured deep-dive / ranked-list reply.
"""

import asyncio
import functools
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

import httpx

from app.config import settings
from app.services.signal_service import generate_trade_signal
from app.services.signals_service import ALL_SYMBOLS, get_all_signals
from app.utils.validators import validate_symbol

logger = logging.getLogger("equitylens.chat")

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
HINGLISH_WORDS = re.compile(
    r"\b(rupaye|rupee|rs|inr|kam|zyada|upar|neeche|niche|stock|predictions?)\b",
    re.IGNORECASE,
)

STOCK_KEYWORDS = re.compile(
    r"\b(stocks?|share|shares?|predict|prediction|price|rate|buy|sell|hold|signal|market|nifty|bse|symbol|nse|fundamental|technical)\b",
    re.IGNORECASE,
)

# Words that indicate the user wants ACTUAL predictions/signals/screening
# (vs. just asking to learn about a concept like "what is RSI").
STOCK_QUERY_KEYWORDS = re.compile(
    r"\b(predict|prediction|signal|buy|sell|hold|overbought|oversold|bullish|bearish|screening|screen|gainers?|losers?|trend|recommend|recommendation|suggest)\b",
    re.IGNORECASE,
)

SMALLTALK_PATTERNS = [
    r"\b(hi+|hii+|hello+|hlo|hallo|hey+|namaste|namaskar|namaskaram|good\s*(morning|afternoon|evening|night)|gd\s*(morning|afternoon|evening|night))\b",
    r"\b(thanks|thank\s*you|thanku|thankz|thx|shukriya|dhanyavad|dhanyavaad)\b",
    r"\b(bye|goodbye|gn|good\s*night)\b",
    r"\b(kaise\s+ho|kaisi\s+ho|how\s+are\s+you|howz\s+u|whats?\s*up|kya\s+haal)\b",
    r"\b(tum\s+kaun|aap\s+kaun|who\s+are\s+you|what\s+can\s+you\s+do|tum\s+kya\s+kar\s+sakte|help)\b",
    r"\b(ok|okay|k|fine|theek\s+hai|acha|achha|thik)\b",
    r"\b(gm|gd\s*mrng|gud\s*mrng|gd\s*noon|gud\s*noon|gud\s*night|gdn)\b",
    r"\b(welcome|wc|you'?re\s*welcome|your\s*welcome|my\s*pleasure|swagat|swagatam)\b",
]
SMALLTALK_RE = re.compile("|".join(SMALLTALK_PATTERNS), re.IGNORECASE)

# Financial disclaimer appended to every NSE/stock-related answer.
_FINANCIAL_DISCLAIMER_HI = (
    "\n\n⚠️ **Disclaimer:** Yeh AI-generated analysis hai — yeh **financial advice nahi hai**. "
    "Invest karne se pehle apni research karein ya SEBI-registered advisor se salah lein.\n"
    "🙏 Dhanyawad AVORA Chatbot use karne ke liye!"
)
_FINANCIAL_DISCLAIMER_EN = (
    "\n\n⚠️ **Disclaimer:** This is AI-generated analysis — it is **not financial advice**. "
    "Do your own research or consult a SEBI-registered advisor before investing.\n"
    "🙏 Thank you for using the AVORA Chatbot!"
)


def _financial_disclaimer(hinglish: bool = False) -> str:
    return _FINANCIAL_DISCLAIMER_HI if hinglish else _FINANCIAL_DISCLAIMER_EN

SYMBOL_SET = set(ALL_SYMBOLS)

# Common NSE company-name aliases -> symbol (unambiguous only; bare ambiguous
# names like "adani"/"tata"/"hdfc" are intentionally NOT included).
COMPANY_NAMES: dict[str, str] = {
    "reliance industries": "RELIANCE",
    "reliance ind": "RELIANCE",
    "reliance": "RELIANCE",
    "tata consultancy services": "TCS",
    "tata consultancy": "TCS",
    "tcs": "TCS",
    "infosys": "INFY",
    "wipro": "WIPRO",
    "hcl technologies": "HCLTECH",
    "hcltech": "HCLTECH",
    "hcl": "HCLTECH",
    "tech mahindra": "TECHM",
    "techm": "TECHM",
    "larsen toubro": "LT",
    "larsen": "LT",
    "tata motors": "TATAMOTORS",
    "tata steel": "TATASTEEL",
    "tata consumer": "TATACONSUM",
    "tata power": "TATAPOWER",
    "maruti suzuki": "MARUTI",
    "maruti": "MARUTI",
    "hero motocorp": "HEROMOTOCO",
    "eicher motors": "EICHERMOT",
    "eicher": "EICHERMOT",
    "bajaj finance": "BAJFINANCE",
    "bharti airtel": "BHARTIARTL",
    "airtel": "BHARTIARTL",
    "bharti": "BHARTIARTL",
    "titan": "TITAN",
    "hal": "HAL",
    "adani enterprises": "ADANIENT",
    "adanient": "ADANIENT",
    "adani green": "ADANIGREEN",
    "adani ports": "ADANIPORTS",
    "dlf": "DLF",
    "siemens": "SIEMENS",
    "bel": "BEL",
    "jsw steel": "JSWSTEEL",
    "jsw": "JSWSTEEL",
    "hindalco": "HINDALCO",
    "ntpc": "NTPC",
    "ongc": "ONGC",
    "indian oil": "IOC",
    "ioc": "IOC",
    "bpcl": "BPCL",
    "gail": "GAIL",
    "power grid": "POWERGRID",
    "powergrid": "POWERGRID",
    "coal india": "COALINDIA",
    "coalindia": "COALINDIA",
    "hindustan petroleum": "HINDPETRO",
    "hpcl": "HINDPETRO",
    "hdfc bank": "HDFCBANK",
    "hdfcbank": "HDFCBANK",
    "icici bank": "ICICIBANK",
    "icici": "ICICIBANK",
    "kotak mahindra": "KOTAKBANK",
    "kotak bank": "KOTAKBANK",
    "kotak": "KOTAKBANK",
    "sbi": "SBIN",
    "state bank of india": "SBIN",
    "state bank": "SBIN",
    "axis bank": "AXISBANK",
    "indusind bank": "INDUSINDBK",
    "indusind": "INDUSINDBK",
    "bank of baroda": "BANKBARODA",
    "baroda": "BANKBARODA",
    "punjab national bank": "PNB",
    "pnb": "PNB",
    "canara bank": "CANBK",
    "canara": "CANBK",
    "federal bank": "FEDERALBNK",
    "federalbnk": "FEDERALBNK",
    "idfc first": "IDFCFIRSTB",
    "idfc": "IDFCFIRSTB",
    "rbl bank": "RBLBANK",
    "rblbank": "RBLBANK",
    "itc": "ITC",
    "hindustan unilever": "HINDUNILVR",
    "hul": "HINDUNILVR",
    "nestle india": "NESTLEIND",
    "nestle": "NESTLEIND",
    "asian paints": "ASIANPAINT",
    "ultratech cement": "ULTRACEMCO",
    "ultratech": "ULTRACEMCO",
    "dr reddy": "DRREDDY",
    "drreddy": "DRREDDY",
    "sun pharma": "SUNPHARMA",
    "sunpharma": "SUNPHARMA",
    "cipla": "CIPLA",
    "divis labs": "DIVISLAB",
    "divislab": "DIVISLAB",
    "auropharma": "AUROPHARMA",
    "lupin": "LUPIN",
    "biocon": "BIOCON",
    "torrent pharma": "TORNTPHARM",
    "torntpharm": "TORNTPHARM",
    "alkem": "ALKEM",
    "pfizer": "PFIZER",
    "cadila healthcare": "CADILAHC",
    "cadilahc": "CADILAHC",
    "cadila": "CADILAHC",
    "motherson": "MOTHERSUMI",
    "mothersumi": "MOTHERSUMI",
    "ashok leyland": "ASHOKLEY",
    "ashokley": "ASHOKLEY",
    "balkrishna industries": "BALKRISIND",
    "balkrishna": "BALKRISIND",
    "tvs motor": "TVSMOTOR",
    "tvsmotor": "TVSMOTOR",
    "l&t infotech": "LTIM",
    "ltim": "LTIM",
    "ltimindtree": "LTIM",
    "mindtree": "MINDTREE",
    "mphasis": "MPHASIS",
    "persistent": "PERSISTENT",
    "coforge": "COFORGE",
    "ltts": "LTTS",
    # Popular NSE names not tracked in the screening list — resolved via live fetch.
    "zomato": "ZOMATO",
    "irctc": "IRCTC",
    "yes bank": "YESBANK",
    "yesbank": "YESBANK",
    "jio financial services": "JIOFIN",
    "jio financial": "JIOFIN",
    "jiofin": "JIOFIN",
    "jio": "JIOFIN",
    "bajaj auto": "BAJAJ-AUTO",
    "bajaj finserv": "BAJAJFINSV",
    "vedanta": "VEDL",
    "hdfc life": "HDFCLIFE",
    "hdfc life insurance": "HDFCLIFE",
    "sbi cards": "SBICARD",
    "sbi card": "SBICARD",
    "sbicard": "SBICARD",
    "tata chemicals": "TATACHEM",
    "tatachem": "TATACHEM",
    "tata elxsi": "TATAELXSI",
    "tataelxsi": "TATAELXSI",
    "trent": "TRENT",
    "dmart": "DMART",
    "avenue supermarts": "DMART",
    "pnt400": "PNB",
    "jubilant foodworks": "JUBLFOOD",
    "jubilant foods": "JUBLFOOD",
    "jublfood": "JUBLFOOD",
    "godrej consumer": "GODREJCP",
    "marico": "MARICO",
    "britannia": "BRITANNIA",
    "adani power": "ADANIPOWER",
    "adani wilmar": "AWL",
    "kolte patil": "KOLTEPATIL",
    "crompton": "CROMPTON",
    "honeywell automation": "HONAUT",
    "suzlon energy": "SUZLON",
    "suzlon": "SUZLON",
    "yes bank": "YESBANK",
    "idbi bank": "IDBI",
    "idbi": "IDBI",
    "vodafone idea": "VI",
    "vi": "VI",
    "nhpc": "NHPC",
    "jindal power": "JPPOWER",
    "steel authority of india": "SAIL",
    "sail": "SAIL",
    "national mineral development": "NMDC",
    "nmdc": "NMDC",
    "irfc": "IRFC",
    "rvnl": "RVNL",
    "nbcc": "NBCC",
}


def _normalize_name(phrase: str) -> str:
    return re.sub(r"\s+", " ", phrase.strip().lower())


def extract_symbols(message: str) -> list[str]:
    """Known NSE symbols (or companies) named in the user's message.

    Matches bare tickers (case-insensitive, with/without .NS/.NSE suffix) and
    common company names (e.g. "reliance", "hdfc bank", "infosys").
    """
    low = (message or "").lower()
    found: list[str] = []

    for tok in re.findall(r"[a-z0-9]+(?:\.[a-z]+)?", low):
        base = tok.split(".")[0].upper()
        if base in SYMBOL_SET and base not in found:
            found.append(base)

    for phrase in sorted(COMPANY_NAMES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", low):
            sym = COMPANY_NAMES[phrase]
            if sym not in found:
                found.append(sym)

    return found


# All-caps words that look like tickers but are almost never NSE symbols.
_TICKER_STOPWORDS = {
    "STOCKS", "SHARE", "SHARES", "PRICE", "PRICES", "RUPEES", "RUPEE", "INR",
    "NIFTY", "SENSEX", "NSE", "BSE", "BUY", "SELL", "HOLD", "SIGNAL", "SIGNALS",
    "TOP", "BEST", "LIST", "UNDER", "BELLOW", "BELOW", "ABOVE", "THAN", "MORE",
    "TODAY", "TOMORROW", "WEEK", "MONTH", "YEAR", "DAYS", "STOCK", "MARKET",
    "RSI", "MACD", "SMA", "PE", "ROE", "PB", "EPS", "IPO", "NEWS", "TREND",
    "ALL", "THESE", "THOSE", "THEM", "THE", "AND", "FOR", "WITH", "FROM",
    "HIGH", "LOW", "GAINERS", "LOSERS", "VOLUME", "BETWEEN", "ABOVE",
}


def _ticker_candidates(message: str) -> list[str]:
    """Find all-caps tokens that could be NSE tickers the user typed by hand,
    even if we don't track them in the screening list."""
    if not message:
        return []
    found: list[str] = []
    for tok in re.findall(r"\b[A-Z][A-Z0-9]{2,9}\b", message):
        base = tok.split(".")[0].upper()
        if base in _TICKER_STOPWORDS or not base.isalpha() or base.isdigit():
            continue
        if not validate_symbol(base):
            continue
        if base not in found:
            found.append(base)
    return found[:3]


def _resolve_symbols(message: str) -> list[str]:
    """Known symbols/companies + any user-typed all-caps ticker candidates."""
    found = extract_symbols(message)
    for cand in _ticker_candidates(message):
        if cand not in found:
            found.append(cand)
    return found[:3]


def is_smalltalk(message: str) -> bool:
    """True when the message is a greeting / thanks / chit-chat (not a stock query)."""
    text = message.strip().lower()
    if not text:
        return False
    if STOCK_KEYWORDS.search(text):
        return False
    if _resolve_symbols(text):
        return False
    intent = parse_intent(text)
    if (
        intent.get("maxPrice") is not None
        or intent.get("minPrice") is not None
        or intent.get("action") is not None
        or intent.get("bullishDays") is not None
    ):
        return False
    if len(text.split()) > 12:
        return False
    return bool(SMALLTALK_RE.search(text))


def is_stock_query(message: str) -> bool:
    """True when the user asks about stock predictions/signals/screening, or names
    a specific NSE stock/company (those get real-data deep dives)."""
    text = message.strip().lower()
    if not text:
        return False
    intent = parse_intent(text)
    if (
        intent.get("maxPrice") is not None
        or intent.get("minPrice") is not None
        or intent.get("action") is not None
        or intent.get("bullishDays") is not None
    ):
        return True
    if _resolve_symbols(text):
        return True
    return bool(STOCK_QUERY_KEYWORDS.search(text))


NUMBER = r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"

# Flexible price core: "100", "₹100", "100/-", "100 rs", "100 rupaye", "100 ke" …
_CORE_PRICE = (
    rf"[₹\s]*{NUMBER}\s*(?:[-/]*\s*)?"
    r"(?:rupees?|rupay\w*|rp|rps|rs\.?|inr)?\s*"
    r"(?:ke|ka|ki|wale)?\s*"
)

MAX_PRICE_PATTERNS = [
    re.compile(
        rf"(?:under|below|less(?:er)?\s+than|cheaper\s+than|upto|up\s+to|max(?:imum)?|at\s+most|matlab)\s*[₹\s]*{NUMBER}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"{_CORE_PRICE}(?:se\s+)?(?:kam|neeche|niche|below|under|less\s+than)\b",
        re.IGNORECASE,
    ),
    re.compile(rf"<\s*{NUMBER}"),
]

MIN_PRICE_PATTERNS = [
    re.compile(
        rf"(?:above|over|more\s+than|greater\s+than|min(?:imum)?|at\s+least)\s*[₹\s]*{NUMBER}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"{_CORE_PRICE}(?:se\s+)?(?:zyada|upar|ooper|bada|badaa|above|over|more\s+than)\b",
        re.IGNORECASE,
    ),
    re.compile(rf">\s*{NUMBER}"),
]

ACTION_PATTERNS = {
    "strong_buy": re.compile(r"\b(strong\s*buy|strong\s*bullish|strongly\s+bullish|very\s+bullish|super\s+bullish|bahut\s+bullish)\b", re.IGNORECASE),
    "buy": re.compile(r"\b(buy|bullish|bull|bullish-keeda|positive|kharid|khareed|kharido|le\s+sak)\b", re.IGNORECASE),
    "strong_sell": re.compile(r"\b(strong\s*sell|strong\s*bearish|strongly\s+bearish|very\s+bearish|super\s+bearish|bahut\s+bearish)\b", re.IGNORECASE),
    "sell": re.compile(r"\b(sell|bearish|bear|bearish-keeda|negative|becho|bech)\b", re.IGNORECASE),
    "hold": re.compile(r"\b(hold|neutral|sideways)\b", re.IGNORECASE),
}

TOP_PATTERN = re.compile(
    rf"(?:top|best|pehle|sabse|list)\s*[of]?\s*{NUMBER}",
    re.IGNORECASE,
)

# Numbers that are clearly NOT prices (durations, percentages, counts).
_NON_PRICE_UNIT = re.compile(
    rf"{NUMBER}\s*(?:(?:din|dino|dinon|dives|days?|saals?|salo|mohine|months?|years?|weeks?|percent)\b|%)",
    re.IGNORECASE,
)


def _parse_num(raw: str) -> float:
    return float(raw.replace(",", ""))


def parse_intent(message: str) -> dict:
    """Extract structured filters from a free-text user message."""
    intent: dict[str, Any] = {
        "maxPrice": None,
        "minPrice": None,
        "action": None,
        "top": 10,
        "bullishDays": None,
    }

    for pat in MAX_PRICE_PATTERNS:
        m = pat.search(message)
        if m:
            intent["maxPrice"] = round(_parse_num(m.group(1)), 2)
            break

    for pat in MIN_PRICE_PATTERNS:
        m = pat.search(message)
        if m:
            intent["minPrice"] = round(_parse_num(m.group(1)), 2)
            break

    for action, pat in ACTION_PATTERNS.items():
        if pat.search(message):
            intent["action"] = action
            break

    # "N din se bullish/upar/badh raha" — duration-based bullish filter
    dur_m = re.search(
        r"(\d{1,2})\s*(?:din|dino|dinon|days?|dives)\s*(?:se|ka|ke)?\s*"
        r"(?:bullish|upar|ooper|badh\s*raha|up\s*trend|chadh)",
        message, re.IGNORECASE,
    )
    if dur_m:
        intent["bullishDays"] = min(int(dur_m.group(1)), 60)

    top_m = TOP_PATTERN.search(message)
    if top_m:
        try:
            n = int(_parse_num(next(g for g in top_m.groups() if g)))
            intent["top"] = min(max(n, 1), 50)
        except (StopIteration, ValueError):
            pass

    # Bare number (no keyword) defaults to "price under X" — but never when the
    # number is a duration/percentage like "20 din" or "30 day".
    if intent["maxPrice"] is None and intent["minPrice"] is None:
        if not _NON_PRICE_UNIT.search(message):
            bare = re.search(rf"[₹\s]*{NUMBER}\s*(?:wale|ke|ka|ki|price|stocks)?\b", message)
            if bare and len(message) < 200:
                intent["maxPrice"] = round(_parse_num(bare.group(1)), 2)

    # "penny stocks" / "penni stocks" = low-priced stocks (default underwater cap).
    if intent["maxPrice"] is None and re.search(r"\bpenn(y|i)?\b", message, re.IGNORECASE):
        intent["maxPrice"] = 100.0

    return intent


def _describe_prediction(s: dict) -> str:
    """Build a short human-readable reason from the model's indicator data."""
    parts: list[str] = []

    if s.get("trend") == "Strong Trend" or (
        s.get("sma20") is not None and s.get("sma50") is not None and s["sma20"] > s["sma50"]
    ):
        parts.append("SMA20 above SMA50 (uptrend)")
    elif s.get("sma20") is not None and s.get("sma50") is not None and s["sma20"] < s["sma50"]:
        parts.append("SMA20 below SMA50 (downtrend)")

    rsi = s.get("rsi")
    if rsi is not None:
        if rsi < 30:
            parts.append(f"RSI {rsi:.1f} oversold")
        elif rsi > 70:
            parts.append(f"RSI {rsi:.1f} overbought")
        else:
            parts.append(f"RSI {rsi:.1f}")

    macd, macd_signal = s.get("macd"), s.get("macdSignal")
    if macd is not None and macd_signal is not None:
        parts.append("MACD bullish" if macd > macd_signal else "MACD bearish")

    return "; ".join(parts) if parts else "No indicator data available"


def load_predictions() -> list[dict]:
    """Flatten & dedupe every stock prediction from the existing model."""
    try:
        data = get_all_signals()
    except Exception as exc:
        logger.error("Failed to load signals for chat: %s", exc)
        return []

    seen: dict[str, dict] = {}
    for category in data.get("categories", []):
        cat_id = category.get("id")
        for s in category.get("stocks", []):
            symbol = s.get("symbol")
            if not symbol:
                continue
            enriched = seen.get(symbol)
            if enriched is None:
                enriched = dict(s)
                enriched["categories"] = []
                enriched["prediction20d"] = _describe_prediction(s)
                enriched["predictionLabel"] = f"{s.get('signal', 'HOLD')} ({s.get('confidence', 0):.0f}%)"
                seen[symbol] = enriched
            if cat_id and cat_id not in enriched["categories"]:
                enriched["categories"].append(cat_id)

    return list(seen.values())


def filter_predictions(stocks: list[dict], intent: dict) -> list[dict]:
    """Apply the parsed intent filters to the model's predictions."""
    result: list[dict] = []
    for s in stocks:
        price = s.get("currentPrice")
        if price is None:
            continue
        if intent.get("maxPrice") is not None and price > intent["maxPrice"]:
            continue
        if intent.get("minPrice") is not None and price < intent["minPrice"]:
            continue

        action = intent.get("action")
        cats = set(s.get("categories") or [])
        if action == "strong_buy" and "strong-buy" not in cats:
            continue
        if action == "strong_sell" and "strong-sell" not in cats:
            continue
        if action == "buy" and s.get("signal") != "BUY":
            continue
        if action == "sell" and s.get("signal") != "SELL":
            continue
        if action == "hold" and s.get("signal") != "HOLD":
            continue

        bullish_days = intent.get("bullishDays")
        if bullish_days is not None:
            ret = s.get(f"ret{bullish_days}d")
            if ret is not None:
                if ret <= 0:
                    continue
            else:
                # exact N precomputed nahi — consecutive up-days pe fallback
                up = s.get("upDaysConsecutive")
                if up is None or up < bullish_days:
                    continue

        result.append(s)

    order = {"BUY": 0, "HOLD": 1, "SELL": 2, "NEUTRAL": 3}
    result.sort(
        key=lambda x: (order.get(x.get("signal", "NEUTRAL"), 3), -(x.get("confidence") or 0))
    )
    return result


# ---------------------------------------------------------------------------
# Symbol deep-dive enrichment (real data, never fabricated)
# ---------------------------------------------------------------------------

def _compact_fundamentals(f: dict | None) -> dict | None:
    if not f:
        return None

    def _pct(v):
        return round(v * 100, 2) if isinstance(v, (int, float)) else None

    def _num(v):
        return round(v, 2) if isinstance(v, (int, float)) else None

    return {
        "sector": f.get("sector"),
        "marketCap": f.get("market_cap"),
        "pe": _num(f.get("pe_ratio")),
        "eps": _num(f.get("eps")),
        "roe": _pct(f.get("roe")),
        "roce": _pct(f.get("roce")),
        "debtEquity": _num(f.get("debt_to_equity")),
        "opm": _pct(f.get("operating_margin")),
        "revenueGrowth": _pct(f.get("revenue_growth")),
        "profitGrowth": _pct(f.get("profit_growth")),
        "score": f.get("fundamental_score"),
        "rating": f.get("rating"),
    }


async def _fetch_source(partial_fn, timeout: float):
    try:
        return await asyncio.wait_for(asyncio.to_thread(partial_fn), timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        logger.debug("chat enrich source failed: %s", exc)
        return None


async def _enrich_symbol(symbol: str) -> dict | None:
    """Real-time context for one NSE symbol (quote/indicators/signal/
    fundamentals/news). Returns None when nothing could be fetched."""
    from app.services.fundamental_service import fundamentals_service
    from app.services.news_service import fetch_stock_news
    from app.services.sentiment_service import average_sentiment_score
    from app.services.yfinance_service import fetch_price_data

    price_data, fund_raw, articles = await asyncio.gather(
        _fetch_source(functools.partial(fetch_price_data, symbol, "6mo"), 20.0),
        _fetch_source(functools.partial(fundamentals_service.get_fundamentals, symbol, "NSE"), 20.0),
        _fetch_source(functools.partial(fetch_stock_news, symbol, settings.chat_news_headlines), 15.0),
    )

    quote = (price_data or {}).get("quote") or {}
    indicators = (price_data or {}).get("indicators") or {}

    try:
        sentiment = average_sentiment_score(articles or [])
    except Exception:  # noqa: BLE001
        sentiment = None

    signal = None
    if quote.get("current_price") is not None:
        try:
            signal = generate_trade_signal(
                price=quote.get("current_price", 0),
                rsi=indicators.get("rsi"),
                macd=indicators.get("macd"),
                signal=indicators.get("signal"),
                sma20=indicators.get("sma20"),
                sma50=indicators.get("sma50"),
                sentiment_score=sentiment,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("chat signal failed for %s: %s", symbol, exc)

    news = [
        {
            "title": a.get("title"),
            "source": a.get("source"),
            "sentiment": (a.get("sentiment") or {}).get("label"),
        }
        for a in (articles or [])[: settings.chat_news_headlines]
    ]

    ctx = {
        "symbol": symbol,
        "quote": {
            "price": quote.get("current_price"),
            "change": quote.get("change"),
            "changePercent": quote.get("change_percent"),
        },
        "indicators": {
            "rsi": indicators.get("rsi"),
            "macd": indicators.get("macd"),
            "macdSignal": indicators.get("signal"),
            "sma20": indicators.get("sma20"),
            "sma50": indicators.get("sma50"),
        },
        "signal": (signal or {}).get("signal"),
        "sentiment": sentiment,
        "news": news,
        "fundamentals": _compact_fundamentals(fund_raw),
    }

    if not ctx["quote"].get("price") and not ctx["fundamentals"] and not news:
        return None
    return ctx


async def _maybe_forecast(symbol: str) -> dict | None:
    """Best-effort structured forecast for one symbol (never fabricates)."""
    try:
        from app.ml.pipeline import forecast_symbol
        result = await asyncio.wait_for(
            asyncio.to_thread(forecast_symbol, symbol, period="10y", fast=True),
            timeout=600,
        )
    except asyncio.TimeoutError:
        logger.warning("chat forecast timed out for %s", symbol)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("chat forecast failed for %s: %s", symbol, exc)
        return None
    if not result.get("is_available"):
        return None
    return result


def _forecast_payload(forecast: dict) -> dict:
    """Compact LLM-safe summary of a structured forecast (no raw snapshots)."""
    latest = forecast.get("latest") or {}
    cal = forecast.get("calibration") or {}
    monitor = forecast.get("monitor") or {}
    return {
        "symbol": forecast.get("symbol"),
        "horizonDays": forecast.get("horizon", 20),
        "probability20dUp": latest.get("probability"),
        "rawProbability": latest.get("raw_probability"),
        "signal": latest.get("signal"),
        "confidence": latest.get("confidence"),
        "thresholdBuy": latest.get("threshold_buy"),
        "thresholdSell": latest.get("threshold_sell"),
        "reason": latest.get("reason"),
        "calibrationMethod": cal.get("method"),
        "brierCalibrated": cal.get("brier_calibrated"),
        "topFactors": forecast.get("explanation", {}).get("factors", [])[:5],
        "monitorMode": monitor.get("mode"),
        "monitorAlarms": monitor.get("alarms", []),
    }


# ---------------------------------------------------------------------------
# Analyst reasoning engine (deterministic, data-driven "thinking")
# ---------------------------------------------------------------------------

def _analyst_report(ctx: dict, forecast: dict | None) -> dict[str, Any]:
    """Composite analyst verdict for one stock from real data.

    Weighs technicals + fundamentals + news sentiment + ML forecast probability
    into a 0-100 score with human-readable reasons. Pure data, no fabrication.
    """
    score = 50
    reasons: list[str] = []

    ind = ctx.get("indicators") or {}
    quote = ctx.get("quote") or {}
    fund = ctx.get("fundamentals") or {}
    price, sma20, sma50 = quote.get("price"), ind.get("sma20"), ind.get("sma50")

    # Technicals
    if sma20 is not None and sma50 is not None:
        if sma20 > sma50:
            score += 20
            reasons.append("SMA20 above SMA50 (uptrend)")
        else:
            score -= 20
            reasons.append("SMA20 below SMA50 (downtrend)")
    if price is not None and sma20 is not None:
        if price > sma20:
            score += 10
            reasons.append(f"Price {_fmt_price(price)} is above SMA20 {_fmt_price(sma20)}")
        else:
            score -= 10
            reasons.append(f"Price {_fmt_price(price)} is below SMA20 {_fmt_price(sma20)}")

    rsi = ind.get("rsi")
    if rsi is not None:
        if rsi < 30:
            reasons.append(f"RSI {rsi:.1f} oversold (bounce zone)")
        elif rsi > 70:
            score -= 10
            reasons.append(f"RSI {rsi:.1f} overbought (pullback risk)")
        elif 40 <= rsi <= 60:
            score += 5
            reasons.append(f"RSI {rsi:.1f} in healthy neutral zone")
        else:
            reasons.append(f"RSI {rsi:.1f}")

    macd, macds = ind.get("macd"), ind.get("macdSignal")
    if macd is not None and macds is not None:
        if macd > macds:
            score += 10
            reasons.append("MACD bullish crossover")
        else:
            score -= 10
            reasons.append("MACD bearish crossover")

    # Fundamentals
    roe = fund.get("roe")
    if roe is not None:
        if roe >= 15:
            score += 10
            reasons.append(f"ROE {roe}% (strong return on equity)")
        elif roe >= 10:
            score += 5
            reasons.append(f"ROE {roe}% (decent return on equity)")
        elif roe < 0:
            score -= 5
            reasons.append(f"ROE {roe}% negative")

    if fund.get("opm") is not None:
        if fund["opm"] >= 20:
            score += 5
            reasons.append(f"Operating margin {fund['opm']}% (healthy)")
        elif fund["opm"] < 5:
            score -= 5
            reasons.append(f"Operating margin {fund['opm']}% thin")

    rev_g = fund.get("revenueGrowth")
    if rev_g is not None:
        score += 5 if rev_g > 0 else -5
        reasons.append(f"Revenue growth {rev_g:+.1f}%")

    prof_g = fund.get("profitGrowth")
    if prof_g is not None:
        score += 5 if prof_g > 0 else -5
        reasons.append(f"Profit growth {prof_g:+.1f}%")

    de = fund.get("debtEquity")
    if de is not None:
        if de > 2:
            score -= 5
            reasons.append(f"High leverage (debt/equity {de})")
        elif de < 1 and de >= 0:
            score += 5
            reasons.append(f"Low leverage (debt/equity {de})")

    pe = fund.get("pe")
    if pe is not None and pe > 50:
        reasons.append(f"Rich valuation (PE {pe})")

    # News sentiment (-1..1)
    sent = ctx.get("sentiment")
    if sent is not None:
        if sent > 0.2:
            score += 10
            reasons.append("Positive news sentiment")
        elif sent < -0.2:
            score -= 10
            reasons.append("Negative news sentiment")

    # ML ensemble forecast probability (calibrated -> gets a strong say in the verdict)
    if forecast is not None:
        latest = forecast.get("latest") or {}
        prob = latest.get("probability")
        sig = latest.get("signal")
        if prob is not None:
            reasons.append(f"ML ensemble model: {sig} with P(up) {prob:.2f}")
            score = int(0.4 * score + 0.6 * (prob * 100))
            reasons.append(f"Verdict weighted toward calibrated ML forecast (P(up) {prob:.2f})")

    score = max(0, min(100, score))
    if score >= 75:
        action = "STRONG BUY"
    elif score >= 60:
        action = "BUY"
    elif score >= 45:
        action = "HOLD"
    elif score >= 30:
        action = "SELL"
    else:
        action = "STRONG SELL"

    return {"score": score, "action": action, "reasons": reasons[:10]}


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are AVORA AI — a best-in-class NSE (Indian stock market) analyst assistant.

The user's DATA message contains REAL data pulled live from AVORA's systems:
- "stocks": the existing technical-indicator model's 20-day predictions (SMA trend, RSI, MACD) for screened stocks.
- "contexts": live deep-dive data for the specific stocks the user asked about (quote, indicators, trade signal, fundamentals, news sentiment).
- "synthesis": AVORA's deterministic analyst verdict (score 0-100 + action + reasons) computed from that data — use it as a reference, but explain in your own words.
- "forecasts": the calibrated ML ensemble forecast (P(up), BUY/HOLD/SELL, explanation factors) for those stocks.

THINK LIKE AN ANALYST:
- Weigh technicals (trend, RSI, MACD) together with fundamentals (PE, ROE, margins, growth) and news sentiment before concluding.
- Give a clear overall view (bullish / bearish / neutral) with the exact numbers from DATA as evidence.
- For specific-stock questions answer in structured markdown sections: Price & Signal, Technicals, Fundamentals, News, ML Forecast.
- For screening questions ("stocks under X", "top buy signals") give a short ranked list — no deep analysis per stock.

STRICT RULES:
1. Use ONLY the data given in the DATA message. Never invent stocks, prices, signals, percentages, or news.
2. Never hallucinate a symbol that is not in the DATA. Use the exact symbol as shown.
3. When a FORECASTS/contexts block has the calibrated probability, that number wins over any rough signal — never modify it.
4. If DATA is empty, say you could not find matching stocks.
5. Answer in the same language the user used (Hindi/Hinglish or English), friendly and concise.
6. Keep the reply under ~320 words. Plain text with simple markdown (### headings, **bold** for numbers/stocks).
7. If a FORECASTS block exists for the stock the user asked about, ALWAYS include the calibrated P(up) and signal in your answer.
8. End by telling the user they can click a stock name to open the full chart + analysis page."""


def _build_llm_messages(message: str, stocks: list[dict], history: list | None,
                        contexts: dict[str, dict] | None = None,
                        forecasts: dict[str, dict] | None = None) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in (history or [])[-8:]:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": message})

    data_payload: dict[str, Any] = {
        "stocks": [
            {
                "symbol": s.get("symbol"),
                "currentPrice": s.get("currentPrice"),
                "changePercent": s.get("changePercent"),
                "signal": s.get("signal"),
                "confidence": s.get("confidence"),
                "trend": s.get("trend"),
                "prediction20d": s.get("prediction20d"),
                "rsi": s.get("rsi"),
                "macd": s.get("macd"),
            }
            for s in stocks
        ]
    }
    if contexts:
        data_payload["contexts"] = contexts
        data_payload["synthesis"] = {
            sym: _analyst_report(ctx, (forecasts or {}).get(sym))
            for sym, ctx in contexts.items()
        }
    if forecasts:
        data_payload["forecasts"] = {
            sym: _forecast_payload(fc) for sym, fc in forecasts.items()
        }
    messages.append({"role": "user", "content": "DATA:\n" + json.dumps(data_payload, ensure_ascii=False)})
    return messages


async def _llm_reply(message: str, stocks: list[dict], history: list | None,
                     contexts: dict[str, dict] | None = None,
                     forecasts: dict[str, dict] | None = None) -> str:
    url = settings.openai_base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.openai_model,
        "temperature": settings.openai_temperature,
        "max_tokens": settings.openai_max_tokens,
        "messages": _build_llm_messages(message, stocks, history, contexts, forecasts),
    }

    timeout = httpx.Timeout(60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        body = resp.json()

    return body["choices"][0]["message"]["content"].strip()


def _hinglish_requested(message: str) -> bool:
    return bool(DEVANAGARI.search(message) or HINGLISH_WORDS.search(message))


def _fmt_price(v) -> str:
    return f"₹{float(v):,.2f}" if v is not None else "N/A"


def _reason_of(ctx: dict) -> str:
    parts: list[str] = []
    ind = ctx.get("indicators") or {}
    sma20, sma50 = ind.get("sma20"), ind.get("sma50")
    if sma20 is not None and sma50 is not None:
        parts.append("SMA20 above SMA50 (uptrend)" if sma20 > sma50 else "SMA20 below SMA50 (downtrend)")
    rsi = ind.get("rsi")
    if rsi is not None:
        parts.append(f"RSI {rsi:.1f} {'oversold' if rsi < 30 else 'overbought' if rsi > 70 else ''}".strip())
    if ind.get("macd") is not None and ind.get("macdSignal") is not None:
        parts.append("MACD bullish" if ind["macd"] > ind["macdSignal"] else "MACD bearish")
    return "; ".join(parts) if parts else "No indicator data"


def _forecast_line(symbol: str, forecast: dict | None, hinglish: bool) -> str:
    if not forecast:
        return ""
    latest = forecast.get("latest") or {}
    prob = latest.get("probability")
    sig = latest.get("signal")
    if prob is None:
        return f"- **{symbol}** 20-day ML forecast: **{sig}**"
    pct = f"{prob * 100:.0f}%"
    if hinglish:
        return f"- **{symbol}** 20-din ML forecast: **{sig}** (calibrated p(up) {prob:.2f}, {pct})"
    return f"- **{symbol}** 20-day ML forecast: **{sig}** (calibrated p(up) = {prob:.2f}, {pct})"


def _fallback_deep_dive(symbol: str, ctx: dict, forecast: dict | None, hinglish: bool) -> str:
    """Deterministic, fully data-driven analyst deep dive for one named stock."""
    quote = ctx.get("quote") or {}
    ind = ctx.get("indicators") or {}
    fund = ctx.get("fundamentals") or {}

    # If Yahoo gave us nothing usable, be honest instead of printing a fake report.
    fund_bits_src = [f for f in (fund.get("sector"), fund.get("pe"), fund.get("roe")) if f is not None]
    has_data = (
        quote.get("price") is not None
        or bool(fund_bits_src)
        or bool(ctx.get("news"))
        or forecast is not None
    )
    if not has_data:
        if hinglish:
            return (
                f"Abhi **{symbol}** ke live data Nahi mil paaye (Yahoo Finance se baar-baar error aaraha). "
                "Kripya thodi der baad dobara try karein, ya kisi aur stock ka naam puchhein. 🙂"
            )
        return (
            f"I couldn't fetch **{symbol}** live data right now (Yahoo Finance is failing "
            "for this symbol repeatedly). Please try again shortly, or ask about another stock. 🙂"
        )

    report = _analyst_report(ctx, forecast)
    signal = (ctx.get("signal") or {}).get("action")

    title = f"**{symbol} — Avora Analysis**"
    verdict = f"**Verdict: {report['action']}** (confidence score {report['score']}/100)"
    price_line = (f"Price: **{_fmt_price(quote.get('price'))}**"
                  + (f" (change {quote.get('changePercent'):+.2f}%)" if quote.get("changePercent") is not None else ""))
    tech_line = (_reason_of(ctx) or "No indicator data available")
    tech_section = f"Technicals: **{signal or 'N/A'}** — {tech_line}"

    fund_bits = []
    if fund.get("sector"):
        fund_bits.append(f"Sector: {fund['sector']}")
    if fund.get("pe") is not None:
        fund_bits.append(f"PE: {fund['pe']}")
    if fund.get("eps") is not None:
        fund_bits.append(f"EPS: {fund['eps']}")
    if fund.get("roe") is not None:
        fund_bits.append(f"ROE: {fund['roe']}%")
    if fund.get("opm") is not None:
        fund_bits.append(f"OPM: {fund['opm']}%")
    if fund.get("revenueGrowth") is not None:
        fund_bits.append(f"Rev Growth: {fund['revenueGrowth']}%")
    if fund.get("profitGrowth") is not None:
        fund_bits.append(f"Profit Growth: {fund['profitGrowth']}%")
    if fund.get("debtEquity") is not None:
        fund_bits.append(f"D/E: {fund['debtEquity']}")
    if fund.get("score") is not None:
        fund_bits.append(f"Avora Score: {fund['score']} ({fund.get('rating') or 'N/A'})")
    fund_section = f"Fundamentals: " + (" · ".join(fund_bits) if fund_bits else "no data")

    news = ctx.get("news") or []
    news_lines = []
    if news:
        news_lines.append("News:")
        for a in news:
            title_t = a.get("title") or ""
            src = a.get("source") or ""
            sent = a.get("sentiment") or ""
            suffix = f" ({sent})" if sent else ""
            news_lines.append(f"- {title_t}{suffix} — {src}" if src else f"- {title_t}{suffix}")

    why_head = "Kyun? (reasons)" if hinglish else "Why?"
    reason_lines = [f"{why_head}"]
    reason_lines += [f"- {r}" for r in report["reasons"]] if report["reasons"] else ["- no data"]

    fc_line = _forecast_line(symbol, forecast, hinglish)
    cta = (
        "Kisi bhi stock ke naam par click karein aur pura chart + analysis dekhein. 📈"
        if hinglish else
        "Click any stock name to open its full chart and analysis. 📈"
    )

    sections = [title, "", verdict, price_line, tech_section, fund_section,
                *news_lines, "", *reason_lines]
    if fc_line:
        sections += ["", fc_line]
    sections += ["", cta]
    return "\n".join(sections)


def _fallback_reply(stocks: list[dict], intent: dict, hinglish: bool,
                    forecasts: dict[str, dict] | None = None) -> str:
    total = len(stocks)
    max_price = intent.get("maxPrice")
    min_price = intent.get("minPrice")
    action = intent.get("action")

    def _forecast_lines(parts: list[str]) -> list[str]:
        if not forecasts:
            return parts
        for sym, fc in forecasts.items():
            line = _forecast_line(sym, fc, hinglish)
            if line:
                parts.append(line)
        return parts

    if hinglish:
        parts: list[str] = []
        condition = ""
        if max_price is not None and min_price is not None:
            condition = f" jo ₹{min_price:g} se ₹{max_price:g} ke beech me hain"
        elif max_price is not None:
            condition = f" jo ₹{max_price:g} se kam price ke hain"
        elif min_price is not None:
            condition = f" jo ₹{min_price:g} se zyada price ke hain"
        if intent.get("bullishDays"):
            condition += f" jo {intent['bullishDays']} din se bullish hain"

        action_text = {
            "buy": " 'BUY' signal ke saath",
            "sell": " 'SELL' signal ke saath",
            "hold": " 'HOLD' signal ke saath",
            "strong_buy": " strong-bullish (STRONG BUY) signal ke saath",
            "strong_sell": " strong-bearish (STRONG SELL) signal ke saath",
        }.get(action, "")

        if total == 0:
            return (
                f"Maaf kijiye, aapki filters ({condition.strip() or 'selected criteria'}) ke hisaab se "
                "koi stock nahi mila. Koi aur price range try karein. 🙂"
            )

        parts.append(
            f"Yahan {total} NSE stocks hain{condition}{action_text} jinme 20-din ka prediction include hai:"
        )
        for s in stocks:
            parts.append(
                f"- **{s.get('symbol')}**: ₹{s.get('currentPrice', 0):,.2f} | "
                f"{s.get('predictionLabel', 'HOLD')} | {s.get('prediction20d', '')}"
            )
        parts = _forecast_lines(parts)
        parts.append("")
        parts.append("Kisi bhi stock ke naam par click karein aur uska pura chart + analysis dekhein. 📈")
        return "\n".join(parts)

    parts = []
    condition = ""
    if max_price is not None and min_price is not None:
        condition = f" priced between ₹{min_price:g} and ₹{max_price:g}"
    elif max_price is not None:
        condition = f" priced below ₹{max_price:g}"
    elif min_price is not None:
        condition = f" priced above ₹{min_price:g}"
    if intent.get("bullishDays"):
        condition += f" that have been bullish for the last {intent['bullishDays']} days"

    action_text = {
            "buy": " with a BUY signal",
            "sell": " with a SELL signal",
            "hold": " with a HOLD signal",
            "strong_buy": " with a strong-bullish (STRONG BUY) signal",
            "strong_sell": " with a strong-bearish (STRONG SELL) signal",
        }.get(action, "")

    if total == 0:
        return (
            f"Sorry, no stocks matched your filters{condition}. Try a different price range."
        )

    parts.append(
        f"Here are {total} NSE stocks{condition}{action_text} with their 20-day predictions:"
    )
    for s in stocks:
        parts.append(
            f"- **{s.get('symbol')}**: ₹{s.get('currentPrice', 0):,.2f} | "
            f"{s.get('predictionLabel', 'HOLD')} | {s.get('prediction20d', '')}"
        )
    parts = _forecast_lines(parts)
    parts.append("")
    parts.append("Click any stock name to open its full chart and analysis. 📈")
    return "\n".join(parts)


async def process_chat(message: str, history: list | None = None) -> dict:
    """Main entrypoint: route -> detect symbols -> parse intent -> answer."""
    message = (message or "").strip()

    if is_smalltalk(message):
        return {
            "reply": await _smalltalk_reply(message),
            "stocks": [],
            "totalFound": 0,
            "intent": {"maxPrice": None, "minPrice": None, "action": None, "top": 10},
            "source": "llm",
            "generatedAt": datetime.now(timezone.utc).isoformat(),
        }

    if not is_stock_query(message):
        return {
            "reply": await _general_reply(message, history),
            "stocks": [],
            "totalFound": 0,
            "intent": {"maxPrice": None, "minPrice": None, "action": None, "top": 10},
            "source": "llm" if settings.openai_api_key else "existing-model",
            "generatedAt": datetime.now(timezone.utc).isoformat(),
        }

    intent = parse_intent(message)
    symbols = _resolve_symbols(message)
    all_stocks = load_predictions()
    filtered = filter_predictions(all_stocks, intent)
    displayed = filtered[: intent.get("top", 10)]
    by_symbol = {s.get("symbol"): s for s in displayed}

    # Deep-dive context for the named symbols (real data, soft-failing).
    contexts: dict[str, dict] = {}
    for symbol in symbols[: settings.chat_max_enrich_symbols]:
        ctx = await _enrich_symbol(symbol)
        if ctx is not None:
            contexts[symbol] = ctx

    # ML forecast for the first named symbol (calibrated, never fabricated).
    forecasts: dict[str, dict] = {}
    if symbols:
        matched = await _maybe_forecast(symbols[0])
        if matched is not None:
            forecasts[symbols[0]] = matched

    reply: str | None = None
    used_llm = False

    if settings.openai_api_key:
        try:
            reply = await _llm_reply(
                message, displayed, history, contexts or None, forecasts or None
            )
            used_llm = True
        except Exception as exc:
            logger.warning("LLM chat failed, falling back to template: %s", exc)
            reply = None

    if not reply:
        if contexts:
            reply = "\n\n".join(
                _fallback_deep_dive(sym, ctx, forecasts.get(sym), _hinglish_requested(message))
                for sym, ctx in contexts.items()
            )
        else:
            reply = _fallback_reply(displayed, intent, _hinglish_requested(message), forecasts or None)

    reply = reply.rstrip() + _financial_disclaimer(_hinglish_requested(message))

    return {
        "reply": reply,
        "stocks": displayed,
        "totalFound": len(filtered),
        "intent": intent,
        "forecasts": forecasts,
        "contexts": contexts,
        "source": "llm" if used_llm else "existing-model",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Smalltalk / general chat
# ---------------------------------------------------------------------------

SMALLTALK_SYSTEM_PROMPT = """You are AVORA's friendly AI assistant for an Indian stock analysis platform.
The user is making small talk (greeting, thanks, how are you, saying goodbye, etc.).
Respond briefly, warmly, and in the same language the user used (Hindi/Hinglish or English).
Keep it under 40 words. Then invite them to ask about NSE stock predictions, for example:
- "500 se kam rate wale stock predictions do"
- "top buy signals"
- "stocks under Rs 500"."""


async def _llm_smalltalk(message: str) -> str:
    url = settings.openai_base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.openai_model,
        "temperature": settings.openai_temperature,
        "max_tokens": 150,
        "messages": [
            {"role": "system", "content": SMALLTALK_SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
    }

    timeout = httpx.Timeout(30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        body = resp.json()

    return body["choices"][0]["message"]["content"].strip()


def _fallback_smalltalk(message: str, hinglish: bool) -> str:
    text = message.lower()

    if re.search(r"\b(gm|gd\s*mrng|gud\s*mrng|good\s*morning)\b", text):
        if hinglish:
            return (
                "Good morning! ☀️ AVORA par aane ka shukriya. "
                "Aaj ka **20-din ka stock prediction** dekhna chahenge? "
                "Jaise: **top buy signals** ya **500 se kam rate wale stocks**."
            )
        return (
            "Good morning! ☀️ Thanks for checking in with AVORA. "
            "Want today's **20-day stock predictions**? Try **top buy signals** "
            "or **stocks under ₹500**."
        )

    if re.search(r"\b(gd\s*noon|gud\s*noon|good\s*afternoon)\b", text):
        if hinglish:
            return (
                "Good afternoon! 🌤️ Kaise chal raha hai aaj ka din? "
                "**NSE predictions** dekhne ke liye puchhiye, jaise: "
                "**stocks under ₹500** ya kisi stock ka naam."
            )
        return (
            "Good afternoon! 🌤️ How's your day going? "
            "Ask for **NSE predictions**, like **stocks under ₹500** "
            "or name any stock."
        )

    if re.search(r"\b(bye|goodbye|gn|gdn|good\s*night|gud\s*night)\b", text):
        if hinglish:
            return (
                "Good night! 🌙 Aapke saath baat karke achha laga. "
                "Kisi bhi waqt **stock predictions** ke liye wapas aa sakte hain. "
                "Shubh ratri!"
            )
        return (
            "Good night! 🌙 It was great chatting with you on AVORA. "
            "Come back anytime for **stock predictions**. Sleep well!"
        )

    if re.search(r"\b(thanks|thank\s*you|thanku|thankz|thx|shukriya|dhanyavad|dhanyavaad)\b", text):
        if hinglish:
            return (
                "Koi baat nahi! 😊 Agar kisi aur stock ke baare me jaanna ho "
                "to bas puchhiye — jaise **RELIANCE prediction** ya **top gainers**."
            )
        return (
            "You're welcome! 😊 If you'd like predictions for any stock, "
            "just ask — e.g. **RELIANCE prediction** or **top gainers**."
        )

    if re.search(r"\b(welcome|wc|you'?re\s*welcome|your\s*welcome|my\s*pleasure|swagat|swagatam)\b", text):
        if hinglish:
            return (
                "Swagat hai! 🙏 Main AVORA hoon — aapka stock analysis assistant. "
                "**20-din ke predictions** ke liye puchhiye: **top buy signals** "
                "ya **stocks under ₹500**."
            )
        return (
            "Welcome! 🙏 I'm AVORA, your stock analysis assistant. "
            "Ask for **20-day predictions** like **top buy signals** or "
            "**stocks under ₹500**."
        )

    if hinglish:
        return (
            "Namaste! 🙏 Main AVORA ka AI assistant hoon. "
            "Mujhse NSE stocks ke **20-din ke predictions** puchhiye, "
            "jaise: **500 se kam rate wale stock predictions do** ya **top buy signals**."
        )
    return (
        "Hello! 👋 I'm AVORA's AI assistant. Ask me about NSE stock predictions, "
        "like **stocks under ₹500** or **top buy signals**."
    )


async def _smalltalk_reply(message: str) -> str:
    if settings.openai_api_key:
        try:
            return await _llm_smalltalk(message)
        except Exception as exc:
            logger.warning("LLM smalltalk failed, using template: %s", exc)
    return _fallback_smalltalk(message, _hinglish_requested(message))


CONCEPT_KB: list[tuple[re.Pattern, str, str]] = [
    (
        re.compile(r"\b(rsi|relative strength index)"),
        "RSI (Relative Strength Index) is a momentum indicator on a 0-100 scale.\n- RSI > 70 → overbought (pullback risk).\n- RSI < 30 → oversold (possible bounce zone).\n- 40-60 → neutral/healthy.\nAVORA computes 14-day RSI from daily closes.",
        "RSI (Relative Strength Index) ek momentum indicator hai, 0-100 scale par.\n- RSI > 70 → overbought (girne ka risk).\n- RSI < 30 → oversold (aane wala bounce).\n- 40-60 → neutral/healthy.\nAVORA 14-din ke RSI se calculate karta hai.",
    ),
    (
        re.compile(r"\b(macd|moving average convergence)"),
        "MACD (Moving Average Convergence Divergence) shows trend momentum using two EMAs (12 & 26) minus a signal line (9).\n- MACD above signal → bullish momentum.\n- MACD below signal → bearish momentum.\n- Values are in price units; the crossover matters more than the level.",
        "MACD (Moving Average Convergence Divergence) do EMAs (12 & 26) aur signal line (9) se trend momentum dikhata hai.\n- MACD, signal se upar → bullish momentum.\n- MACD, signal se neeche → bearish.\n- Crossover zyada matter karta hai, level se nahi.",
    ),
    (
        re.compile(r"\b(sma|simple moving average|moving average)\b"),
        "SMA (Simple Moving Average) is the average closing price over N days — a trend filter.\n- Price above SMA20 + SMA20 above SMA50 → short-term uptrend.\n- AVORA uses SMA20 vs SMA50 to classify Strong/Weak trend.",
        "SMA (Simple Moving Average) N din ka average closing price hai — trend filter.\n- Price, SMA20 se upar + SMA20, SMA50 se upar → uptrend.\n- AVORA trend check ke liye SMA20 vs SMA50 compare karta hai.",
    ),
    (
        re.compile(r"\b(p/?e ratio|price.?to.?earnings|pe)\b"),
        "PE Ratio = Price ÷ Earnings Per Share. It tells how much investors pay per ₹1 of profit.\n- Low PE (vs sector) → can be undervalued/value.\n- High PE → growth expectations, but also richer valuation/risk.\n- Compare it with the sector, not in isolation.",
        "PE Ratio = Price ÷ EPS. Bataata hai investors ₹1 profit ke liye kitna PAY karte hain.\n- Low PE (sector ke hisaab se) → sasta/value ho sakta hai.\n- High PE → growth ka dam, par valuation rich.\n- Hamesha sector se compare karein, akela nahi.",
    ),
    (
        re.compile(r"\b(pb ratio|price.?to.?book|book value)\b"),
        "PB Ratio = Price ÷ Book Value per share. Book value ≈ net assets per share.\n- Very high PB → market has high growth/long-term expectations for the business.\n- Useful for banks/financials (assets matter).",
        "PB Ratio = Price ÷ Book Value per share. Book value ≈ net assets per share.\n- Bahut high PB → market ko business se high growth ki ummeed.\n- Banks/financial stocks ke liye zyada useful.",
    ),
    (
        re.compile(r"\b(roe|return on equity)\b"),
        "ROE (Return on Equity) = Net Profit ÷ Shareholders' Equity. Shows how efficiently a company uses shareholders' money.\n- >15% is generally strong; >20% is excellent.\n- High ROE + low debt is the ideal combo.",
        "ROE (Return on Equity) = Net Profit ÷ Shareholders' Equity. Dikhata hai company shareholders ka paisa kitni achhi tarah use karti hai.\n- >15% strong, >20% excellent.\n- High ROE + kam debt = best combo.",
    ),
    (
        re.compile(r"\b(dividend yield)\b"),
        "Dividend Yield = Annual dividend ÷ Price × 100. Income you earn per ₹100 invested.\n- TS/utilities-like stable businesses give higher, steady yields.\n- Note: yield rises when price falls — check why the price fell.",
        "Dividend Yield = Annual dividend ÷ Price × 100. Har ₹100 invest par milegi kitni dividend.\n- Stable businesses (utilities/FMCG) steady yield dete hain.\n- Note: price gire to yield badhta hai — check karo ki price kyun gira.",
    ),
    (
        re.compile(r"\b(nifty|sensex)\b"),
        "Nifty 50 is India's benchmark index of the 50 largest NSE-listed companies. Sensex is the 30-stock BSE index.\n- They act as the market pulse — if the index is up, most large-caps usually follow.\n- Stock-specific news/sectors can still move a stock against the index.",
        "Nifty 50 India ka benchmark index hai — NSE ki sabse badi 50 companies. Sensex BSE ka 30-stock index hai.\n- Ye market ka pulse hain — index upar to zyada large-caps usually follow karte hain.\n- Phir bhi stock-specific news/sector market ke against move kar sakta hai.",
    ),
    (
        re.compile(r"\b(nse|bse)\b"),
        "NSE (National Stock Exchange) and BSE (Bombay Stock Exchange) are India's two main exchanges.\n- Trading happens on both; each stock has a code (e.g. RELIANCE.NS on NSE, RELIANCE.BO on BSE).\n- AVORA uses NSE data (Nifty universe).",
        "NSE (National Stock Exchange) aur BSE (Bombay Stock Exchange) India ke do main exchanges hain.\n- Dono par trading hoti hai — har stock ka code hota hai (jaise RELIANCE.NS, RELIANCE.BO).\n- AVORA NSE data use karta hai (Nifty universe).",
    ),
    (
        re.compile(r"\b(ipo)\b"),
        "IPO (Initial Public Offering) is when a private company sells its shares to the public for the first time to raise money.\n- Research the business & valuation before applying.\n- Listings can 'pop' or fall below the issue price — gains are not guaranteed.",
        "IPO (Initial Public Offering) = private company pehli baar public me shares bechti hai paise jama karne ke liye.\n- Apply karne se pehle business aur valuation zaroor padhein.\n- Listing upar bhi aa sakti hai, neeche bhi — guaranteed gain nahi hai.",
    ),
    (
        re.compile(r"\b(bull market|bear market|bullish|bearish)\b"),
        "Bull market = prices rising over time with optimism; bear market = sustained fall with pessimism.\n- 'Bullish' = expectation of rise, 'bearish' = expectation of fall.\n- AVORA signals use SMA/RSI/MACD to decide if a stock's short-term bias is bullish or bearish.",
        "Bull market = prices lambi period me badhte hain (optimism); bear market = sustained gullet (pessimism).\n- 'Bullish' = upar jane ki ummeed, 'bearish' = girne ki ummeed.\n- AVORA ke signals SMA/RSI/MACD se short-term bias batate hain.",
    ),
    (
        re.compile(r"\b(how\s+to\s+(start|begin|invest)|invest\s+karna|investing\s+kaise|mutual\s+fund|sip)\b"),
        "How to start investing in India (simple path):\n1. Open a Demat + trading account (Zerodha/Groww/Upstox etc.).\n2. Start with a diversified investment — index mutual funds / SIPs are the safest start.\n3. Learn the basics (PE, ROE, debt, growth) before picking individual stocks.\n4. Invest for the long term; AVORA's 20-day signals help with timing, not a get-rich plan.\nNo tip is a guarantee — always evaluate risk yourself.",
        "India me investing kaise shuru karein (simple raasta):\n1. Demat + trading account kholen (Zerodha/Groww/Upstox etc.).\n2. Shuruaat diversified se karein — index mutual funds / SIP sabse safe hain.\n3. Single stock lene se pehle basics seekhein (PE, ROE, debt, growth).\n4. Long-term ke liye invest karein; AVORA ke 20-din signals timing ke liye hain, get-rich plan nahi.\nKoi tip guarantee nahi — risk khud evaluate karein.",
    ),
]


def _concept_kb_reply(message: str) -> str | None:
    """Deterministic educational answers for common NSE/stock concepts."""
    text = message.lower()
    hinglish = _hinglish_requested(message)
    for pattern, en, hi in CONCEPT_KB:
        if pattern.search(text):
            return hi if hinglish else en
    return None


_WHO_PATTERN = re.compile(
    r"\b(who\s+(made|created|built|developed)\s+you|"
    r"who\s+is\s+your\s+(maker|creator|developer)|"
    r"(kisne|kaun|koun)\s+(banaya|banayi|banai)|"
    r"(tumko|tumhe|aapko|aapne)\s+(kisne|kaun|koun)\s+banaya|"
    r"(tumhara|tumhari|aapka|aapke)\s+(maker|creator|developer))\b",
    re.IGNORECASE,
)


def _who_made_you(message: str) -> str | None:
    """Deterministic reply for "who made/created/built AVORA?" questions."""
    if not _WHO_PATTERN.search(message):
        return None
    if _hinglish_requested(message):
        return (
            "Mujhe **AVORA team** ne banaya hai 😊 — NSE predictions, deep-dive "
            "analysis aur Hinglish/English chat sab ek hi jagah.\n\n"
            "Jo bhi answer deta hoon wo **real data** (indicators, fundamentals, news) "
            "se aata hai — fabricated numbers kabhi nahi."
        )
    return (
        "I was built by the **AVORA team** 😊 — NSE predictions, deep-dive analysis "
        "and Hinglish/English chat all in one place.\n\n"
        "Everything I answer comes from **real data** (indicators, fundamentals, news) — "
        "I never fabricate numbers."
    )


# Out-of-scope topics — AVORA only helps with NSE stocks & the dashboard.
OUT_OF_SCOPE_PATTERNS = [
    r"\b(weather|mausam|temperature|rain|barish)\b",
    r"\b(cricket|football|ipl|world\s*cup|hockey|tennis|match|score)\b",
    r"\b(movie|film|bollywood|hollywood|actor|actress|celebrity|song|singer)\b",
    r"\b(recipe|khana|food\s*recipe|cooking|biryani)\b",
    r"\b(girlfriend|boyfriend|relationship|love\s*life|shadi|marriage|wife|husband)\b",
    r"\b(politics|election|prime\s*minister|president|government|modi|rahul)\b",
    r"\b(world\s*news|international\s*news|breaking\s*news)\b",
    r"\b(translate|essay|homework|assignment|school\s*project)\b",
    r"\b(math|physics|chemistry)\b",
    r"\bsolve\b|\bequation\b",
    r"\d+\s*[+\-*/]\s*\d+\s*=\s*",
    r"\b(joke|jokes|chutkula|funny)\b",
    r"\b(friends|party|hangout|plan\s*ban|outing)\b",
    r"\b(health|doctor|medicine|bimari|illness|fever)\b",
]
OUT_OF_SCOPE_RE = re.compile("|".join(OUT_OF_SCOPE_PATTERNS), re.IGNORECASE)


def _out_of_scope_reply(message: str) -> str | None:
    if not OUT_OF_SCOPE_RE.search(message or ""):
        return None
    if _hinglish_requested(message):
        return (
            "Maaf kijiye 🙏, main **AVORA ka AI assistant** hoon aur sirf **NSE stock "
            "analysis aur AVORA dashboard** ke liye hoon. Mere paas general world data "
            "ya stock market ke bahar ke topics ki jankari nahi hai.\n\n"
            "Main aapki madad kar sakta hoon: stock predictions, prices, buy/sell "
            "signals, fundamentals, news aur dashboard features me.\n\n"
            "Kripya kuch **stocks ya AVORA platform** se juda sawaal puchiye. 🙂"
        )
    return (
        "Sorry 🙏, I'm **AVORA's AI assistant**, specialised in **NSE stock analysis "
        "and the AVORA dashboard**. I don't have access to general world data or topics "
        "outside the stock market.\n\n"
        "I can help you with: stock predictions, prices, buy/sell signals, fundamentals, "
        "news, and dashboard features.\n\n"
        "Please ask me something related to **stocks or the AVORA platform**. 🙂"
    )


GENERAL_SYSTEM_PROMPT = """You are AVORA AI — a friendly assistant for AVORA, an Indian stock (NSE) analysis platform.

Your scope is LIMITED to NSE stocks and the AVORA dashboard:
- Stock predictions, prices, buy/sell/hold signals, fundamentals, news, and concepts (RSI, MACD, PE, ROE, etc.).
- Dashboard features and how to use the platform.

If the user asks about anything OUTSIDE this scope (world news, weather, sports, movies, politics, general knowledge, math, jokes, personal topics, etc.), politely decline and redirect:
"I'm AVORA's AI assistant, specialised in NSE stock analysis and the AVORA dashboard. I can't help with topics outside the stock market. Please ask me something about stocks or the AVORA platform."

Rules:
- Answer in the same language the user uses (Hindi/Hinglish or English).
- Be friendly, concise, and professional. Keep answers under 150 words.
- Never pretend to have live stock data — for actual numbers, point the user to ask for predictions (e.g. "500 se kam rate wale stocks")."""


async def _llm_general(message: str, history: list | None) -> str:
    url = settings.openai_base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }

    messages: list[dict] = [{"role": "system", "content": GENERAL_SYSTEM_PROMPT}]
    for turn in (history or [])[-8:]:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message})

    payload = {
        "model": settings.openai_model,
        "temperature": settings.openai_temperature,
        "max_tokens": settings.openai_max_tokens,
        "messages": messages,
    }

    timeout = httpx.Timeout(45.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        body = resp.json()

    return body["choices"][0]["message"]["content"].strip()


def _fallback_general(message: str, hinglish: bool) -> str:
    if hinglish:
        return (
            "Mujhe aapka sawaal samajh aaya! 😊 Main AVORA AI hoon — main aapke saath "
            "general baat kar sakta hoon (jokes, concepts, sawaal) aur NSE stocks ke "
            "**predictions** bhi de sakta hoon.\n\n"
            "Koi bhi cheez puchiye, jaise: **RSI kya hai?**, **aaj ka joke sunao**, ya "
            f"**\"{message[:60]}\"** ke baare me batao."
        )
    return (
        "Got it! 😊 I'm AVORA AI — happy to chat about anything (concepts, jokes, general "
        "questions) and also give NSE stock **predictions**.\n\n"
        f"Ask me anything, like: **What is RSI?**, **tell me a joke**, or tell me more about "
        f"**\"{message[:60]}\"**."
    )


async def _general_reply(message: str, history: list | None) -> str:
    hinglish = _hinglish_requested(message)
    kb = _concept_kb_reply(message)
    if kb:
        return kb + _financial_disclaimer(hinglish)
    who = _who_made_you(message)
    if who:
        return who
    oos = _out_of_scope_reply(message)
    if oos:
        return oos
    if settings.openai_api_key:
        try:
            return (await _llm_general(message, history)) + _financial_disclaimer(hinglish)
        except Exception as exc:
            logger.warning("LLM general chat failed, using template: %s", exc)
    return _fallback_general(message, hinglish) + _financial_disclaimer(hinglish)