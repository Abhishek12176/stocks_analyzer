"""Point-in-time sentiment features (Task 6).

News articles (from `news_service`/`sentiment_service`) are converted into
events via the `events` module, so they inherit the exact same no-lookahead
semantics: an article published *after* the NSE close of day T is only
effective from day T+1 and can never influence the day-T prediction row.

Article scores come from `article["sentiment"]["score"]` (already attached by
`sentiment_service.analyze_articles`) or a plain top-level `score`; an article
with neither is skipped (never fabricated as 0). Aggregations are causally
computed over a full daily calendar (so weekend/holiday news stays in the
trailing window) and only then sampled on the stock's trading dates.

Sentiment features (default window = 5 window-days):
- `sent_score_{w}d`     : mean score of articles effective in the window
- `sent_count_{w}d`     : count of such articles
- `sent_pos_ratio_{w}d` : fraction of positive-score articles in the window
- `sent_recent`         : mean score of articles effective this exact day
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import settings
from app.ml.events import _daily_counts, effective_date

SENTIMENT_WINDOW = 5


def _article_score(article: dict) -> float:
    sentiment = article.get("sentiment") or {}
    score = sentiment.get("score")
    if score is None:
        score = article.get("score")
    if score is None:
        return float("nan")
    return float(score)


def articles_to_events(articles: list[dict]) -> list[dict]:
    """Convert news articles into PIT events (published_dt -> event time)."""
    events = []
    for article in articles:
        published = article.get("published_dt")
        if published is None:
            continue
        event: dict = {"event_timestamp": published}
        if article.get("title"):
            event["title"] = article["title"]
        if article.get("category"):
            event["category"] = article["category"]
        score = _article_score(article)
        event["_score"] = None if pd.isna(score) else score
        events.append(event)
    return events


def _daily_score_frame(
    events: list[dict],
    trading_index: pd.DatetimeIndex,
    pad_days: int = 0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Causal per-calendar-day aggregation of article scores."""
    counts = _daily_counts(events, trading_index, pad_days=pad_days)
    cal = counts.index
    mean_scores = pd.Series(np.nan, index=cal)
    pos_counts = pd.Series(0.0, index=cal)

    score_map: dict[pd.Timestamp, list[float]] = {}
    for event in events:
        if event.get("_score") is None:
            continue
        day = effective_date(event)
        score_map.setdefault(day, []).append(event["_score"])

    for day, scores in score_map.items():
        if day in cal:
            mean_scores.loc[day] = float(sum(scores)) / len(scores)
            pos_counts.loc[day] = float(sum(1 for s in scores if s > 0))

    return mean_scores, counts.astype(float), pos_counts


def add_sentiment_features(
    df: pd.DataFrame,
    articles: list[dict],
    window: int = SENTIMENT_WINDOW,
) -> pd.DataFrame:
    """Attach causal news-sentiment columns to a copy of the frame."""
    out = df.copy()
    trading = pd.DatetimeIndex(out.index).normalize()
    events = articles_to_events(articles)

    mean_scores, daily_counts, pos_counts = _daily_score_frame(events, trading, pad_days=window)

    score_win = mean_scores.rolling(window, min_periods=1).mean()
    count_win = daily_counts.rolling(window).sum()
    pos_ratio_raw = pos_counts.rolling(window).sum() / count_win
    pos_ratio = pos_ratio_raw.replace([np.inf, -np.inf], np.nan)

    out[f"sent_score_{window}d"] = score_win.reindex(trading).values
    out[f"sent_count_{window}d"] = count_win.reindex(trading).values
    out[f"sent_pos_ratio_{window}d"] = pos_ratio.reindex(trading).values
    out["sent_recent"] = mean_scores.reindex(trading).values

    out.attrs["feature_version"] = settings.ml_feature_version
    return out


def list_sentiment_features(window: int = SENTIMENT_WINDOW) -> list[str]:
    return sorted(
        {
            f"sent_score_{window}d",
            f"sent_count_{window}d",
            f"sent_pos_ratio_{window}d",
            "sent_recent",
        }
    )