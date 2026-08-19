import logging
import re
from typing import Any

logger = logging.getLogger("equitylens.sentiment")

_finbert_pipeline = None
_vader = None


def _load_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is not None:
        return _finbert_pipeline
    try:
        from transformers import pipeline
        _finbert_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
        )
        logger.info("FinBERT loaded successfully")
    except Exception as exc:
        logger.warning("FinBERT not available (%s), using VADER", exc)
        _finbert_pipeline = False
    return _finbert_pipeline


def _load_vader():
    global _vader
    if _vader is not None:
        return _vader
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    return _vader


def _finbert_score(text: str) -> dict:
    pipe = _load_finbert()
    if not pipe:
        return {"label": "neutral", "score": 0.0}
    result = pipe(text[:512])[0]
    label = result["label"].lower()
    score = result["score"] if label == "positive" else -result["score"] if label == "negative" else 0.0
    return {"label": label, "score": round(score, 4)}


def _vader_score(text: str) -> dict:
    analyzer = _load_vader()
    scores = analyzer.polarity_scores(text[:512])
    compound = scores["compound"]
    if compound >= 0.35:
        label = "positive"
    elif compound <= -0.35:
        label = "negative"
    else:
        label = "neutral"
    return {"label": label, "score": round(compound, 4)}


def analyze_sentiment(text: str) -> dict:
    if not text or not text.strip():
        return {"label": "neutral", "score": 0.0}
    finbert = _load_finbert()
    if finbert:
        return _finbert_score(text)
    return _vader_score(text)


def analyze_articles(articles: list[dict]) -> list[dict]:
    for article in articles:
        try:
            combined = f"{article.get('title', '')} {article.get('summary', '')}"
            result = analyze_sentiment(combined)
            article["sentiment"] = result
        except Exception as exc:
            logger.debug("Sentiment failed for article: %s", exc)
            article["sentiment"] = {"label": "neutral", "score": 0.0}
    return articles


def average_sentiment_score(articles: list[dict]) -> float:
    scores = [
        a.get("sentiment", {}).get("score", 0.0)
        for a in articles
        if a.get("sentiment", {}).get("label") != "neutral"
    ]
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 4)
