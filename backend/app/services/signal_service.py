from datetime import datetime, timezone


def generate_trade_signal(
    price: float,
    rsi: float | None,
    macd: float | None,
    signal: float | None,
    sma20: float | None,
    sma50: float | None,
    prediction: str = "",
    sentiment_score: float | None = None,
) -> dict:
    """Generate a multi-factor trade signal based on technical indicators + news sentiment."""
    reasons = []
    bull_count = 0
    bear_count = 0

    # Trend — SMA crossover
    if sma20 is not None and sma50 is not None:
        if sma20 > sma50:
            bull_count += 1
            reasons.append(
                {
                    "factor": "Trend",
                    "impact": "bullish",
                    "detail": f"SMA20 ({sma20:.1f}) above SMA50 ({sma50:.1f}) — uptrend",
                }
            )
        else:
            bear_count += 1
            reasons.append(
                {
                    "factor": "Trend",
                    "impact": "bearish",
                    "detail": f"SMA20 ({sma20:.1f}) below SMA50 ({sma50:.1f}) — downtrend",
                }
            )
    else:
        reasons.append(
            {
                "factor": "Trend",
                "impact": "neutral",
                "detail": "SMA data not available",
            }
        )

    # RSI
    if rsi is not None:
        if rsi < 30:
            bull_count += 1
            reasons.append(
                {
                    "factor": "RSI",
                    "impact": "bullish",
                    "detail": f"RSI at {rsi:.1f} — oversold territory (< 30)",
                }
            )
        elif rsi > 70:
            bear_count += 1
            reasons.append(
                {
                    "factor": "RSI",
                    "impact": "bearish",
                    "detail": f"RSI at {rsi:.1f} — overbought territory (> 70)",
                }
            )
        else:
            reasons.append(
                {
                    "factor": "RSI",
                    "impact": "neutral",
                    "detail": f"RSI at {rsi:.1f} — neutral zone (30-70)",
                }
            )
    else:
        reasons.append(
            {
                "factor": "RSI",
                "impact": "neutral",
                "detail": "RSI data not available",
            }
        )

    # MACD
    if macd is not None and signal is not None:
        if macd > signal:
            bull_count += 1
            reasons.append(
                {
                    "factor": "MACD",
                    "impact": "bullish",
                    "detail": f"MACD ({macd:.2f}) above signal line ({signal:.2f})",
                }
            )
        else:
            bear_count += 1
            reasons.append(
                {
                    "factor": "MACD",
                    "impact": "bearish",
                    "detail": f"MACD ({macd:.2f}) below signal line ({signal:.2f})",
                }
            )
    else:
        reasons.append(
            {
                "factor": "MACD",
                "impact": "neutral",
                "detail": "MACD data not available",
            }
        )

    # Prediction
    if "Bullish" in prediction:
        bull_count += 1
        reasons.append(
            {
                "factor": "Forecast",
                "impact": "bullish",
                "detail": prediction,
            }
        )
    elif "Bearish" in prediction:
        bear_count += 1
        reasons.append(
            {
                "factor": "Forecast",
                "impact": "bearish",
                "detail": prediction,
            }
        )

    # Sentiment
    if sentiment_score is not None:
        if sentiment_score > 0.3:
            bull_count += 1
            reasons.append(
                {
                    "factor": "Sentiment",
                    "impact": "bullish",
                    "detail": f"News sentiment positive ({sentiment_score:+.2f}) — favorable coverage",
                }
            )
        elif sentiment_score < -0.3:
            bear_count += 1
            reasons.append(
                {
                    "factor": "Sentiment",
                    "impact": "bearish",
                    "detail": f"News sentiment negative ({sentiment_score:.2f}) — unfavorable coverage",
                }
            )
        else:
            reasons.append(
                {
                    "factor": "Sentiment",
                    "impact": "neutral",
                    "detail": f"News sentiment neutral ({sentiment_score:+.2f})",
                }
            )

    # Determine action
    total_factors = bull_count + bear_count
    if total_factors > 0:
        confidence = (
            max(bull_count, bear_count) / total_factors
        ) * 100
    else:
        confidence = 50

    if bull_count > bear_count:
        action = "BUY"
        direction = "bullish"
    elif bear_count > bull_count:
        action = "SELL"
        direction = "bearish"
    else:
        action = "HOLD"
        direction = "neutral"

    return {
        "signal": {
            "action": action,
            "direction": direction,
            "confidence": round(confidence, 1),
            "reasons": reasons,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "quote": {
            "price": round(price, 2),
            "change": 0,
            "change_percent": 0,
        },
    }
