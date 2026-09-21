"""Signal policy (Task 9).

Turns a calibrated P(up) into a daily trading decision using configured,
validated thresholds:
- P(up) >= `threshold_buy`  -> BUY  (direction +1)
- P(up) <= `threshold_sell` -> SELL (direction -1 when shorting is allowed,
                                  else 0 = flat)
- otherwise                 -> HOLD (direction 0)

Additionally a `confidence_floor` can force HOLD when the probability sits
too close to 0.5 to be worth acting on. The policy is a pure function of its
inputs (deterministic), so the same P(up) always maps to the same action.
"""

from __future__ import annotations

from typing import Any

from app.config import settings


def validate_thresholds(
    threshold_buy: float,
    threshold_sell: float,
    confidence_floor: float = 0.0,
) -> tuple[float, float, float]:
    """Sanity-checks and normalises the policy thresholds."""
    for name, v in (("threshold_buy", threshold_buy),
                    ("threshold_sell", threshold_sell),
                    ("confidence_floor", confidence_floor)):
        if not (isinstance(v, (int, float)) and 0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {v!r}")
    if threshold_sell >= threshold_buy:
        raise ValueError(
            "threshold_sell must be strictly below threshold_buy "
            f"({threshold_sell} >= {threshold_buy})")
    return float(threshold_buy), float(threshold_sell), float(confidence_floor)


def _confidence(prob: float) -> float:
    """How far the probability is from the 0.5 neutral point, in [0, 1]."""
    return float(abs(prob - 0.5) * 2.0)


def decide_signal(
    prob: float,
    threshold_buy: float | None = None,
    threshold_sell: float | None = None,
    confidence_floor: float = 0.08,  # raised from 0.0: prob in (0.46, 0.54) strictly HOLD
    allow_short: bool = False,
    regime_trend: float | None = None,
    regime_risk: float | None = None,
) -> dict[str, Any]:
    """Map one P(up) to a BUY / HOLD / SELL decision with regime defense.

    A `confidence_floor` forces HOLD when the probability sits too close to
    0.5 to be worth acting on. A `regime_trend` of -1 (market bear) or a
    `regime_risk` of -1 (risk-off) raises the effective buy threshold to
    0.62, so marginal BUY signals are blocked and the policy never chases
    longs into a falling market. Deterministic pure function of its inputs.
    """
    if not (isinstance(prob, (int, float)) and 0.0 <= prob <= 1.0):
        raise ValueError(f"prob must be in [0, 1], got {prob!r}")
    tb = settings.ml_threshold_buy if threshold_buy is None else threshold_buy
    ts = settings.ml_threshold_sell if threshold_sell is None else threshold_sell
    tb, ts, cf = validate_thresholds(tb, ts, confidence_floor)

    conf = _confidence(prob)

    # Defensive rule: NIFTY bear trend (-1) ya risk-off (-1) me marginal BUY
    # signals block karo (effective buy threshold 0.62), strictly HOLD/CASH.
    is_bear_regime = (regime_trend == -1) or (regime_risk == -1)
    effective_tb = max(tb, 0.62) if is_bear_regime else tb

    if conf < cf:
        signal, direction = "HOLD", 0
    elif prob >= effective_tb:
        signal, direction = "BUY", 1
    elif prob <= ts:
        signal, direction = "SELL", (-1 if allow_short else 0)
    else:
        signal, direction = "HOLD", 0

    return {
        "prob": float(prob),
        "signal": signal,
        "direction": direction,
        "confidence": conf,
        "threshold_buy": tb,
        "threshold_sell": ts,
        "effective_threshold_buy": effective_tb,
        "allow_short": bool(allow_short),
        "reason": _reason(prob, signal, effective_tb, ts),
    }


def _reason(prob: float, signal: str, tb: float, ts: float) -> str:
    if signal == "BUY":
        return f"P(up)={prob:.3f} >= BUY threshold {tb:.3f}"
    if signal == "SELL":
        return f"P(up)={prob:.3f} <= SELL threshold {ts:.3f}"
    return f"P(up)={prob:.3f} inside HOLD band ({ts:.3f} < p < {tb:.3f})"