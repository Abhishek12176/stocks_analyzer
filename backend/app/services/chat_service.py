"""AI Chat Assistant service.

Pipeline:
  1. parse_intent()  -> extract filters (max/min price, action, top-N) from the
                        user's message (English + Hinglish friendly).
  2. load predictions -> ONLY from the existing technical-indicator model via
                        signals_service.get_all_signals() (never new forecasts).
  3. filter & rank   -> apply intent filters to the model's predictions.
  4. answer          -> if OPENAI_API_KEY is set, an OpenAI-compatible LLM
                        rephrases the data (strictly forbidden from inventing).
                        Otherwise a deterministic template builds the reply.
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

import httpx

from app.config import settings
from app.services.signal_service import generate_trade_signal
from app.services.signals_service import get_all_signals

logger = logging.getLogger("equitylens.chat")

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
HINGLISH_WORDS = re.compile(
    r"\b(rupaye|rupee|rs|inr|kam|zyada|upar|neeche|niche|stock|predictions?)\b",
    re.IGNORECASE,
)

STOCK_KEYWORDS = re.compile(
    r"\b(stock|share|predict|prediction|price|rate|buy|sell|hold|signal|market|nifty|bse|symbol|nse|fundamental|technical)\b",
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
]
SMALLTALK_RE = re.compile("|".join(SMALLTALK_PATTERNS), re.IGNORECASE)


def is_smalltalk(message: str) -> bool:
    """True when the message is a greeting / thanks / chit-chat (not a stock query)."""
    text = message.strip().lower()
    if not text:
        return False
    if STOCK_KEYWORDS.search(text):
        return False
    if len(text.split()) > 12:
        return False
    return bool(SMALLTALK_RE.search(text))


def is_stock_query(message: str) -> bool:
    """True when the user is asking for stock predictions/signals/screening
    (answers use the existing prediction model). Educational questions like
    "what is RSI" or "stock market kaise kaam karta hai" return False."""
    text = message.strip().lower()
    if not text:
        return False
    intent = parse_intent(text)
    if (
        intent.get("maxPrice") is not None
        or intent.get("minPrice") is not None
        or intent.get("action") is not None
    ):
        return True
    return bool(STOCK_QUERY_KEYWORDS.search(text))

NUMBER = r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"

MAX_PRICE_PATTERNS = [
    re.compile(
        rf"(?:under|below|less(?:er)?\s+than|cheaper\s+than|upto|up\s+to|max(?:imum)?|at\s+most|matlab)\s*[₹\s]*{NUMBER}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"[₹]?\s*{NUMBER}\s*(?:se\s+kam|se\s+neeche|se\s+niche|se\s+chota|se\s+chhota|kam\s+rate|below|under|less\s+than)\b",
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
        rf"[₹]?\s*{NUMBER}\s*(?:se\s+zyada|se\s+upar|se\s+ooper|se\s+bada|se\s+badaa|above|over|more\s+than)\b",
        re.IGNORECASE,
    ),
    re.compile(rf">\s*{NUMBER}"),
]

ACTION_PATTERNS = {
    "buy": re.compile(r"\b(buy|strong\s+buy|kharid|khareed|kharido|le\s+sak|sacrifice)\b", re.IGNORECASE),
    "sell": re.compile(r"\b(sell|strong\s+sell|becho|bech)\b", re.IGNORECASE),
    "hold": re.compile(r"\b(hold|neutral)\b", re.IGNORECASE),
}

TOP_PATTERN = re.compile(
    rf"(?:top|best|pehle|sabse|list)\s*[of]?\s*{NUMBER}",
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

    top_m = TOP_PATTERN.search(message)
    if top_m:
        try:
            n = int(_parse_num(next(g for g in top_m.groups() if g)))
            intent["top"] = min(max(n, 1), 50)
        except (StopIteration, ValueError):
            pass

    # Bare number (no keyword) defaults to "price under X"
    if intent["maxPrice"] is None and intent["minPrice"] is None:
        bare = re.search(rf"[₹\s]*{NUMBER}\s*(?:wale|ke|ka|ki|price|stocks)?\b", message)
        if bare and len(message) < 200:
            intent["maxPrice"] = round(_parse_num(bare.group(1)), 2)

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
        for s in category.get("stocks", []):
            symbol = s.get("symbol")
            if not symbol or symbol in seen:
                continue
            enriched = dict(s)
            enriched["prediction20d"] = _describe_prediction(s)
            enriched["predictionLabel"] = f"{s.get('signal', 'HOLD')} ({s.get('confidence', 0):.0f}%)"
            seen[symbol] = enriched

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
        if action == "buy" and s.get("signal") != "BUY":
            continue
        if action == "sell" and s.get("signal") != "SELL":
            continue
        if action == "hold" and s.get("signal") != "HOLD":
            continue

        result.append(s)

    order = {"BUY": 0, "HOLD": 1, "SELL": 2, "NEUTRAL": 3}
    result.sort(
        key=lambda x: (order.get(x.get("signal", "NEUTRAL"), 3), -(x.get("confidence") or 0))
    )
    return result


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the AI assistant for AVORA, an Indian stock analysis platform.
You answer questions about NSE stock predictions.

STRICT RULES:
1. Use ONLY the prediction data that is given to you in the DATA message. Never invent stocks, prices, signals, or percentages.
2. Never hallucinate a stock symbol that is not present in the DATA.
3. The predictions come from AVORA's technical-indicator model (SMA trend, RSI, MACD) and represent a 20-day outlook.
4. If the DATA list is empty, say you could not find any stocks matching the filters.
5. Answer in the same language the user used (Hindi/Hinglish or English). Keep it friendly and concise.
6. Format the reply as plain text with simple markdown. Keep it under 220 words.
7. If a user asks about something unrelated to stocks, politely say you only help with NSE stock predictions.
8. Always end by telling the user they can click any stock name to open its full chart and analysis page."""


def _build_llm_messages(message: str, stocks: list[dict], history: list | None) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in (history or [])[-8:]:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": message})

    data_payload = {
        "stocks": [
            {
                "symbol": s.get("symbol"),
                "currentPrice": s.get("currentPrice"),
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
    messages.append({"role": "user", "content": "DATA:\n" + json.dumps(data_payload, ensure_ascii=False)})
    return messages


async def _llm_reply(message: str, stocks: list[dict], history: list | None) -> str:
    url = settings.openai_base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.openai_model,
        "temperature": settings.openai_temperature,
        "max_tokens": settings.openai_max_tokens,
        "messages": _build_llm_messages(message, stocks, history),
    }

    timeout = httpx.Timeout(45.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        body = resp.json()

    return body["choices"][0]["message"]["content"].strip()


def _hinglish_requested(message: str) -> bool:
    return bool(DEVANAGARI.search(message) or HINGLISH_WORDS.search(message))


def _fallback_reply(stocks: list[dict], intent: dict, hinglish: bool) -> str:
    total = len(stocks)
    max_price = intent.get("maxPrice")
    min_price = intent.get("minPrice")
    action = intent.get("action")

    if hinglish:
        parts: list[str] = []
        condition = ""
        if max_price is not None and min_price is not None:
            condition = f" jo ₹{min_price:g} se ₹{max_price:g} ke beech me hain"
        elif max_price is not None:
            condition = f" jo ₹{max_price:g} se kam price ke hain"
        elif min_price is not None:
            condition = f" jo ₹{min_price:g} se zyada price ke hain"

        action_text = {"buy": " 'BUY' signal ke saath", "sell": " 'SELL' signal ke saath", "hold": " 'HOLD' signal ke saath"}.get(action, "")

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

    action_text = {"buy": " with a BUY signal", "sell": " with a SELL signal", "hold": " with a HOLD signal"}.get(action, "")

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
    parts.append("")
    parts.append("Click any stock name to open its full chart and analysis. 📈")
    return "\n".join(parts)


async def process_chat(message: str, history: list | None = None) -> dict:
    """Main entrypoint: parse intent -> load model predictions -> answer."""
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
    all_stocks = load_predictions()
    filtered = filter_predictions(all_stocks, intent)
    displayed = filtered[: intent.get("top", 10)]

    reply: str | None = None
    used_llm = False

    if settings.openai_api_key:
        try:
            reply = await _llm_reply(message, displayed, history)
            used_llm = True
        except Exception as exc:
            logger.warning("LLM chat failed, falling back to template: %s", exc)
            reply = None

    if not reply:
        reply = _fallback_reply(displayed, intent, _hinglish_requested(message))

    return {
        "reply": reply,
        "stocks": displayed,
        "totalFound": len(filtered),
        "intent": intent,
        "source": "llm" if used_llm else "existing-model",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
    }


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


def _fallback_smalltalk(hinglish: bool) -> str:
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
    return _fallback_smalltalk(_hinglish_requested(message))


GENERAL_SYSTEM_PROMPT = """You are AVORA AI — a friendly, knowledgeable general-purpose assistant for AVORA, an Indian stock analysis platform.

You can chat about anything, just like ChatGPT:
- Casual conversation, greetings, jokes, small talk.
- Explain concepts (what is RSI, how does the stock market work, what is NSE, etc.).
- Answer general knowledge and math/science questions.
- Give helpful, concise advice.

When the user asks about specific NSE stock predictions, prices, buy/sell signals or screening (e.g. "stocks under Rs 500"), guide them — those queries get answered by AVORA's prediction engine. You can invite them to ask such queries.

Rules:
- Answer in the same language the user uses (Hindi/Hinglish or English).
- Be friendly and concise. Keep answers under 150 words unless the user asks for detail.
- Never pretend to have live stock data or prices — for actual stock numbers, point the user to ask for predictions (e.g. "500 se kam rate wale stocks")."""


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
    if settings.openai_api_key:
        try:
            return await _llm_general(message, history)
        except Exception as exc:
            logger.warning("LLM general chat failed, using template: %s", exc)
    return _fallback_general(message, _hinglish_requested(message))
