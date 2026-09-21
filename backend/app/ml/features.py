"""Technical feature engine for the 20-day forecasting system (Task 3).

Extends `app/services/indicator_service.py` (SMA/EMA/RSI/MACD) with the full
set of technical features. All computations are pure pandas and strictly
causal (no lookahead): every feature at day T uses only data <= T.

Features produced:
- Bollinger Bands (upper/mid/lower/width/%B)
- ATR + ATR%  ·  ADX / +DI / -DI
- CCI  ·  OBV (+ segment slope)  ·  MFI
- ROC (1/5/10/20d)  ·  raw momentum  ·  multi-horizon returns (1/3/5/10/20/60/120d)
- price-distance from SMA20/50/200 and EMA50
- MA slopes (5d) and MA-crossovers (20x50, 20x200) + spread
- Heikin-Ashi body/direction
- Pivot support/resistance levels (past-only)
- Fibonacci retracement position (past-only)

Versioning: each produced DataFrame carries `df.attrs["feature_version"]`
matching `settings.ml_feature_version`; `list_technical_features()` returns the
deterministic list of feature column names for that version.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import settings
from app.services.indicator_service import ema, sma

# Tunables
RETURN_HORIZONS: tuple[int, ...] = (1, 3, 5, 10, 20, 60, 120)
BB_LENGTH = 20
BB_K = 2.0
ATR_LENGTH = 14
ADX_LENGTH = 14
CCI_LENGTH = 20
MFI_LENGTH = 14
PIVOT_WINDOW = 10
FIB_WINDOW = 60
DISTANCE_MA_PERIODS: tuple[int, ...] = (20, 50, 200)
EMA_DISTANCE_PERIOD = 50
CROSS_PAIRS: tuple[tuple[int, int], ...] = ((20, 50), (20, 200))
SLOPE_LOOKBACK = 5


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def bollinger(close: pd.Series, length: int = BB_LENGTH, k: float = BB_K) -> pd.DataFrame:
    mid = sma(close, length)
    std = close.rolling(window=length).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    span = (upper - lower).replace(0, np.nan)
    return pd.DataFrame(
        {
            "BB_upper": upper,
            "BB_mid": mid,
            "BB_lower": lower,
            "BB_width": (upper - lower) / mid,
            "BB_pctB": (close - lower) / span,
        }
    )


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    high_low = df["High"] - df["Low"]
    high_pc = (df["High"] - prev_close).abs()
    low_pc = (df["Low"] - prev_close).abs()
    return pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)


def atr(df: pd.DataFrame, length: int = ATR_LENGTH) -> pd.Series:
    return true_range(df).ewm(alpha=1 / length, adjust=False).mean()


def adx(df: pd.DataFrame, length: int = ADX_LENGTH) -> pd.DataFrame:
    high, low = df["High"], df["Low"]
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=df.index
    )
    atr_series = atr(df, length).replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr_series
    minus_di = 100 * minus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr_series
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame(
        {
            "ADX": dx.ewm(alpha=1 / length, adjust=False).mean(),
            "plus_DI": plus_di,
            "minus_DI": minus_di,
        }
    )


def cci(df: pd.DataFrame, length: int = CCI_LENGTH) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    ma = tp.rolling(window=length).mean()
    md = tp.rolling(window=length).apply(
        lambda x: float(np.mean(np.abs(x - np.mean(x)))), raw=True
    ).replace(0, np.nan)
    return (tp - ma) / (0.015 * md)


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


def mfi(df: pd.DataFrame, length: int = MFI_LENGTH) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    raw_money = tp * df["Volume"]
    pos = raw_money.where(tp > tp.shift(1), 0.0)
    neg = raw_money.where(tp < tp.shift(1), 0.0)
    pos_sum = pos.rolling(window=length).sum()
    neg_sum_raw = neg.rolling(window=length).sum()
    neg_sum = neg_sum_raw.replace(0, np.nan)
    mfr = pos_sum / neg_sum
    mfi_series = 100 - 100 / (1 + mfr)
    return mfi_series.where(neg_sum_raw > 0, 100.0)


def roc(close: pd.Series, n: int) -> pd.Series:
    return close.pct_change(n) * 100


def momentum(close: pd.Series, n: int) -> pd.Series:
    return close.diff(n)


def return_series(close: pd.Series, n: int) -> pd.Series:
    return close.pct_change(n)


def multi_horizon_returns(close: pd.Series, horizons: tuple[int, ...] = RETURN_HORIZONS) -> pd.DataFrame:
    out = {}
    for n in horizons:
        out[f"ret_{n}d"] = return_series(close, n)
    return pd.DataFrame(out, index=close.index)


def price_distance(close: pd.Series, ma: pd.Series) -> pd.Series:
    return close / ma - 1


def ma_slope(series: pd.Series, lookback: int = SLOPE_LOOKBACK) -> pd.Series:
    prev = series.shift(lookback).replace(0, np.nan)
    return (series - series.shift(lookback)) / prev


def ma_crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """+1 when fast crosses above slow, -1 when it crosses below, else 0."""
    above = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    below = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    return pd.Series(
        np.where(above, 1.0, np.where(below, -1.0, 0.0)), index=fast.index
    )


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha_close = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    ha_open = np.empty(len(df), dtype=float)
    if len(df) > 0:
        ha_open[0] = ha_close.iloc[0]
        for i in range(1, len(df)):
            ha_open[i] = 0.5 * (ha_open[i - 1] + ha_close.iloc[i - 1])
    ha_open = pd.Series(ha_open, index=df.index)
    ha_high = pd.concat([df["High"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df["Low"], ha_open, ha_close], axis=1).min(axis=1)
    body = ha_close - ha_open
    return pd.DataFrame(
        {
            "ha_close": ha_close,
            "ha_open": ha_open,
            "ha_high": ha_high,
            "ha_low": ha_low,
        }
    ).assign(
        ha_body_pct=lambda d: (d["ha_close"] - d["ha_open"]) / d["ha_open"].replace(0, np.nan),
        ha_direction=lambda d: np.sign(d["ha_close"] - d["ha_open"]),
    )


def pivot_levels(df: pd.DataFrame, window: int = PIVOT_WINDOW) -> pd.DataFrame:
    high = df["High"].rolling(window=window).max()
    low = df["Low"].rolling(window=window).min()
    close = df["Close"]
    return pd.DataFrame(
        {
            "pivot_high": high,
            "pivot_low": low,
            "pivot_dist_high_pct": close / high - 1,
            "pivot_dist_low_pct": close / low - 1,
        }
    )


def fibonacci_retracement(df: pd.DataFrame, window: int = FIB_WINDOW) -> pd.DataFrame:
    high = df["High"].rolling(window=window).max()
    low = df["Low"].rolling(window=window).min()
    span = (high - low).replace(0, np.nan)
    close = df["Close"]
    return pd.DataFrame(
        {
            "fib_level_pct": (close - low) / span,
            "fib_dist_382": (close - (high - 0.382 * span)).abs() / close,
            "fib_dist_618": (close - (high - 0.618 * span)).abs() / close,
        }
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def list_technical_features() -> list[str]:
    """Deterministic, versioned list of feature column names (sorted)."""
    cols = [
        "BB_upper", "BB_mid", "BB_lower", "BB_width", "BB_pctB",
        "ATR", "ATR_pct", "ADX", "plus_DI", "minus_DI",
        "CCI", "OBV", "OBV_slope", "MFI",
    ]
    for n in (1, 5, 10, 20):
        cols.append(f"roc_{n}d")
    for n in (10,):
        cols.append(f"mom_{n}d")
    for n in RETURN_HORIZONS:
        cols.append(f"ret_{n}d")
    for p in DISTANCE_MA_PERIODS:
        cols.append(f"dist_sma{p}")
    cols.append(f"dist_ema{EMA_DISTANCE_PERIOD}")
    for p in DISTANCE_MA_PERIODS:
        cols.append(f"ma_slope_sma{p}_{SLOPE_LOOKBACK}d")
    for fast, slow in CROSS_PAIRS:
        cols.append(f"ma_cross_{fast}_{slow}")
        cols.append(f"ma_spread_{fast}_{slow}")
    cols += [
        "ha_body_pct", "ha_direction",
        "pivot_high", "pivot_low",
        "pivot_dist_high_pct", "pivot_dist_low_pct",
        "fib_level_pct", "fib_dist_382", "fib_dist_618",
        "dist_52w_high", "vol_ratio_20_60",
    ]
    return sorted(set(cols))


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach all technical-feature columns to a copy of the OHLCV frame."""
    out = df.copy()
    close = out["Close"]

    bb = bollinger(close, BB_LENGTH, BB_K)
    atr_series = atr(out, ATR_LENGTH)
    adx_df = adx(out, ADX_LENGTH)
    obv_series = obv(out)

    features: dict[str, pd.Series] = {}
    features.update({c: bb[c] for c in bb.columns})
    features["ATR"] = atr_series
    features["ATR_pct"] = atr_series / close
    features.update({c: adx_df[c] for c in adx_df.columns})
    features["CCI"] = cci(out, CCI_LENGTH)
    features["OBV"] = obv_series
    features["OBV_slope"] = obv_series.pct_change(20)
    features["MFI"] = mfi(out, MFI_LENGTH)
    for n in (1, 5, 10, 20):
        features[f"roc_{n}d"] = roc(close, n)
    features["mom_10d"] = momentum(close, 10)
    features.update({c: multi_horizon_returns(close)[c] for c in multi_horizon_returns(close).columns})

    for period in DISTANCE_MA_PERIODS:
        features[f"dist_sma{period}"] = price_distance(close, sma(close, period))
    features[f"dist_ema{EMA_DISTANCE_PERIOD}"] = price_distance(close, ema(close, EMA_DISTANCE_PERIOD))
    for period in DISTANCE_MA_PERIODS:
        ma_series = sma(close, period)
        features[f"ma_slope_sma{period}_{SLOPE_LOOKBACK}d"] = ma_slope(ma_series, SLOPE_LOOKBACK)

    for fast, slow in CROSS_PAIRS:
        fast_sma = sma(close, fast)
        slow_sma = sma(close, slow)
        features[f"ma_cross_{fast}_{slow}"] = ma_crossover(fast_sma, slow_sma)
        features[f"ma_spread_{fast}_{slow}"] = (fast_sma - slow_sma) / slow_sma.replace(0, np.nan)

    features.update({c: heikin_ashi(out)[c] for c in ["ha_body_pct", "ha_direction"]})
    features.update({c: pivot_levels(out)[c] for c in ["pivot_high", "pivot_low", "pivot_dist_high_pct", "pivot_dist_low_pct"]})
    features.update({c: fibonacci_retracement(out)[c] for c in ["fib_level_pct", "fib_dist_382", "fib_dist_618"]})

    # 52-week high proximity (momentum) + volume trend
    features["dist_52w_high"] = (
        close / close.rolling(window=252, min_periods=1).max() - 1
    )
    features["vol_ratio_20_60"] = (
        out["Volume"].rolling(20).mean()
        / out["Volume"].rolling(60).mean().replace(0, np.nan)
    )

    for name, series in features.items():
        out[name] = series

    out.attrs["feature_version"] = settings.ml_feature_version
    return out