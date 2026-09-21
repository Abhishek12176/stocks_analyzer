"""GDELT-based historical news sentiment (RESEARCH ONLY — Task 24).

Problem this module solves:
  - The production NewsData path (``news_service``/``sentiment_service``) only
    returns *today's* articles on the free tier, so ``sent_*`` features carry
    data on exactly one row of a 5-year frame and are dropped by
    ``forecast_frame`` as (near-)constant. There is no way to measure whether
    news sentiment helps the locked OOF benchmark with that source.
  - GDELT (globaldatabase.events.gdeltproject.org) is a free, full-text news
    index back to 2017 that exposes a per-day **AvgTone** via its DOC 2.0 API
    (``mode=timelinetone``). That gives a real 5-year daily sentiment series
    per symbol, so the locked A/B (baseline full-OOF AUC 0.5703 vs +``sent_*``)
    becomes measurable.

Design (STAYS OUT OF PRODUCTION):
  - New module only. ``sentiment_features.py``, ``build_feature_frame`` and
    ``pipeline.py`` are NOT touched.
  - Each day with news becomes ONE causal event
    ``{"published_dt": <ISO>, "sentiment": {"score": <scaled tones>}}`` and the
    existing, PIT-safe ``sentiment_features.add_sentiment_features`` is reused
    to attach ``sent_*`` columns (article effective from T+1 only, no lookahead).
  - GDELT AvgTone is [-100, 100]; scaled by 0.01 to [-1, 1] to match the
    FinBERT/VADER compound scale used by the production sentiment path.

Fetch strategy (GDELT rate limit = 1 request / 5 s, hard IP throttling):
  1. Try the full span as ONE ``timelinetone`` call.
  2. If that is rejected or comes back too coarse (fewer distinct days than
     expected), fall back to per-quarter calls and concatenate.
  429s are retried with a fixed backoff for a bounded number of attempts.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from app.ml.sentiment_features import add_sentiment_features

logger = logging.getLogger("gdelt.sentiment")

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_RATE_SEC = 6.0  # stay comfortably above GDELT's 1-request-per-5s limit
GDELT_MAX_RETRIES = 2
GDELT_TONE_SCALE = 0.01  # AvgTone [-100,100] -> [-1,1]
# If a single full-span call returns fewer than this many distinct days, treat
# it as GDELT having coarsened/rejected the window and fall back to quarters.
MIN_DISTINCT_DAYS = 120

_SAFE_QUERIES: dict[str, str] = {
    "TCS": '"TATA CONSULTANCY SERVICES"',
    "RELIANCE": '"RELIANCE INDUSTRIES"',
    "HDFCBANK": '"HDFC BANK"',
    "INFY": '"INFOSYS"',
}


def _probe_gdelt(retries: int = 3, wait: float = 45.0) -> bool:
    """Check GDELT reachability; returns True if a light query answers 200.

    GDELT aggressively temp-blocks IPs that burst. Probe first so long
    multi-quarter loops don't burn time against a live 429 wall.
    """
    for attempt in range(retries):
        try:
            r = requests.get(
                GDELT_DOC_URL,
                params={"query": "INFOSYS", "mode": "artlist", "format": "json",
                        "timespan": "1d", "maxrecords": "3"},
                timeout=45,
            )
        except requests.RequestException:  # noqa: BLE001
            time.sleep(wait)
            continue
        if r.status_code == 200:
            return True
        if attempt < retries - 1:
            logger.warning("GDELT probe HTTP %s (attempt %d) -> waiting %.0fs",
                           r.status_code, attempt + 1, wait)
            time.sleep(wait)
    return False


def _gdelt_call(
    query: str,
    start: datetime,
    end: datetime,
    *,
    retries: int = GDELT_MAX_RETRIES,
) -> list[dict]:
    """One DOC-API ``timelinetone`` call for ``[start, end]`` with 429 backoff."""
    params = {
        "query": query,
        "mode": "timelinetone",
        "format": "json",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
        "timelinebucket": "24h",
    }
    for attempt in range(retries + 1):
        try:
            resp = requests.get(GDELT_DOC_URL, params=params, timeout=90)
        except requests.RequestException as exc:  # noqa: BLE001
            logger.warning("GDELT request error: %s", exc)
            time.sleep(GDELT_RATE_SEC * (attempt + 2))
            continue
        if resp.status_code == 200:
            return (resp.json() or {}).get("timeline") or []
        if attempt < retries:
            backoff = GDELT_RATE_SEC * (attempt + 1)
            logger.warning(
                "GDELT HTTP %s for %s (attempt %d) -> backoff %.0fs",
                resp.status_code, query, attempt + 1, backoff,
            )
            time.sleep(backoff)
        else:
            logger.error("GDELT gave up on %s (HTTP %s)", query, resp.status_code)
            return []
    return []


def fetch_gdelt_timeline(
    query: str,
    start: datetime,
    end: datetime,
    *,
    quarter_fallback: bool = True,
    probe: bool = True,
) -> pd.Series:
    """Daily AvgTone series (indexed by date) for ``[start, end]``.

    Returns a float Series with daily GDELT AvgTone (unnormalised, [-100, 100]).
    Days with no news are simply absent (NaN after reindex).
    """
    if probe and not _probe_gdelt():
        logger.error("GDELT unreachable (temp IP block) -> returning empty")
        return pd.Series(dtype=float)

    points = _gdelt_call(query, start, end)
    if not points:
        # Hard-empty (blocked or no matches). Do NOT burn time in a quarterly
        # retry loop against a live 429 wall — the caller's cooldown retries.
        return _points_to_series(points)

    if quarter_fallback and len({p.get("date") for p in points}) < MIN_DISTINCT_DAYS:
        logger.info("full-span call coarse/empty for %s -> quarterly fallback", query)
        parts: list[pd.Series] = []
        cursor = start
        while cursor < end:
            q_end = min(cursor + timedelta(days=91), end)
            q = _gdelt_call(query, cursor, q_end)
            time.sleep(GDELT_RATE_SEC)
            if q:
                s = _points_to_series(q)
                if not s.empty:
                    parts.append(s)
            cursor = q_end
        if not parts:
            return pd.Series(dtype=float)
        series = pd.concat(parts)
        series = series[~series.index.duplicated(keep="last")].sort_index()
        return series

    return _points_to_series(points)


def _points_to_series(points: list[dict]) -> pd.Series:
    """Convert GDELT timeline JSON points into a date-indexed tone Series."""
    rows: dict[pd.Timestamp, float] = {}
    for p in points:
        d = p.get("date")
        v = p.get("value")
        if d is None or v is None:
            continue
        try:
            ts = pd.Timestamp(str(d))
            rows[ts] = float(v) * GDELT_TONE_SCALE
        except (ValueError, TypeError):
            continue
    return pd.Series(rows).sort_index()


def tone_to_articles(tone: pd.Series, symbol: str) -> list[dict]:
    """Wrap a daily tone series as one PIT article/event per news day.

    Each day with a tone becomes ``{"published_dt": ..., "sentiment":
    {"score": <scaled tone>}, "title": "<SYM> GDELT daily tone", ...}`` so the
    existing causal ``add_sentiment_features`` can consume it unchanged.
    """
    articles: list[dict] = []
    for ts, val in tone.dropna().items():
        articles.append({
            "title": f"{symbol} GDELT daily news tone",
            "summary": "",
            "link": "",
            "source": "GDELT",
            "published_dt": ts.normalize().isoformat() + "+00:00",
            "published": ts.strftime("%d %b %Y"),
            "sentiment": {"label": "pos" if val > 0 else "neg" if val < 0 else "neutral",
                          "score": round(float(val), 4)},
        })
    return articles


def add_gdelt_sentiment_features(
    frame: pd.DataFrame,
    tone: pd.Series,
    symbol: str,
    window: int = 5,
) -> pd.DataFrame:
    """Attach ``sent_*`` columns from a GDELT tone series to a feature frame.

    ``window`` defaults to 5 to match the production sentiment engine window.
    """
    articles = tone_to_articles(tone, symbol)
    return add_sentiment_features(frame, articles, window=window)


def _coverage_report(tone: pd.Series, trading: pd.DatetimeIndex) -> dict:
    present = tone.reindex(trading).dropna()
    return {
        "n_days_with_news": int(len(present)),
        "n_trading_days": int(len(trading)),
        "coverage_pct": round(float(len(present) / len(trading) * 100), 2) if len(trading) else 0.0,
        "first_news_day": str(present.index.min().date()) if len(present) else None,
        "last_news_day": str(present.index.max().date()) if len(present) else None,
        "n_distinct_tone_values": int(present.nunique()),
        "tone_min": round(float(present.min()), 4) if len(present) else None,
        "tone_max": round(float(present.max()), 4) if len(present) else None,
        "tone_mean": round(float(present.mean()), 4) if len(present) else None,
    }