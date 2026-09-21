"""Momentum/Relative + Volatility/Risk feature engine for the 20-day forecast (Task 4).

Extends the technical engine (`app/ml/features.py`) with market-relative and
risk/volatility features. All computations are pure pandas/numpy, strictly
causal (no lookahead); market series are reindexed to the stock index and
left as NaN where missing (never fabricated/backfilled across gaps).

Relative features (stock vs market, default NIFTY 50):
- excess return 10/20/60d, geometric relative-strength ratio 20d
- rolling correlation 60d, stock vol vs market vol 20d
- cross-sectional relative-strength percentile rank (across a universe)

Volatility / risk features:
- rolling vol (20/60d, annualised 252), downside vol (20d)
- vol percentile (rank within trailing window), vol expansion/contraction (5d)
- mean high-low range (20d), gap frequency (open vs prev close >1%, 20d)
- rolling max drawdown (60d)

Versioned: each produced DataFrame carries `df.attrs["feature_version"]`;
`list_market_features()` returns the deterministic feature list.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import settings

TRADING_DAYS = 252
DEFAULT_MARKET_NAME = "nifty50"

# Convenient defaults (also used by the feature list)
VOL_WINDOWS: tuple[int, ...] = (20, 60)
DOWN_VOL_WINDOW = 20
VOL_PCTILE_VOL_WINDOW = 20
VOL_PCTILE_RANK_WINDOW = 252
VOL_EXP_WINDOW = 20
VOL_EXP_LOOKBACK = 5
HL_RANGE_WINDOW = 20
GAP_WINDOW = 20
GAP_THRESHOLD = 0.01
MAX_DRAWDOWN_WINDOW = 60
REL_RET_HORIZONS: tuple[int, ...] = (10, 20, 60)
RS_RATIO_HORIZON = 20
CORR_WINDOW = 60
VOL_RATIO_WINDOW = 20


# ---------------------------------------------------------------------------
# Volatility / risk
# ---------------------------------------------------------------------------

def daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change(fill_method=None)


def rolling_vol(close: pd.Series, window: int = VOL_WINDOWS[0], annualize: bool = True) -> pd.Series:
    vol = daily_returns(close).rolling(window=window).std(ddof=0)
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS)
    return vol


def downside_vol(close: pd.Series, window: int = DOWN_VOL_WINDOW, annualize: bool = True) -> pd.Series:
    """Downside deviation: RMS of negative returns only."""
    r = daily_returns(close)
    neg = r.where(r < 0, 0.0)
    dv = np.sqrt((neg ** 2).rolling(window=window).mean())
    if annualize:
        dv = dv * np.sqrt(TRADING_DAYS)
    return dv


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """Percentile rank (0..1) of the trailing-most value within each window."""
    return series.rolling(window=window).apply(
        lambda x: float(np.mean(x <= x[-1])), raw=True
    )


def vol_percentile(close: pd.Series, rank_window: int = VOL_PCTILE_RANK_WINDOW,
                   vol_window: int = VOL_PCTILE_VOL_WINDOW) -> pd.Series:
    return rolling_percentile(rolling_vol(close, vol_window), rank_window)


def vol_expansion(close: pd.Series, window: int = VOL_EXP_WINDOW,
                  lookback: int = VOL_EXP_LOOKBACK) -> pd.Series:
    """vol_t / vol_{t-lookback} — >1 expansion, <1 contraction."""
    vol = rolling_vol(close, window)
    return vol / vol.shift(lookback).replace(0, np.nan)


def high_low_range(df: pd.DataFrame, window: int = HL_RANGE_WINDOW) -> pd.Series:
    return ((df["High"] - df["Low"]) / df["Close"].replace(0, np.nan)).rolling(window=window).mean()


def gap_frequency(df: pd.DataFrame, window: int = GAP_WINDOW,
                  threshold: float = GAP_THRESHOLD) -> pd.Series:
    prev_close = df["Close"].shift(1)
    gap = (df["Open"] - prev_close).abs() / prev_close.replace(0, np.nan)
    return (gap > threshold).astype(float).rolling(window=window).mean()


def max_drawdown(close: pd.Series, window: int = MAX_DRAWDOWN_WINDOW) -> pd.Series:
    def _dd(x: np.ndarray) -> float:
        peak = np.maximum.accumulate(x)
        return float((x / peak - 1).min())

    return close.rolling(window=window).apply(_dd, raw=True)


# ---------------------------------------------------------------------------
# Relative (stock vs market)
# ---------------------------------------------------------------------------

def _common_pair(series: pd.Series, other: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Align two series on their common non-NaN dates only."""
    frame = pd.concat([series, other], axis=1, join="outer")
    frame = frame.dropna()
    return frame.iloc[:, 0], frame.iloc[:, 1]


def _reindex_like(feature: pd.Series, like: pd.Series) -> pd.Series:
    return feature.reindex(like.index)


def excess_return(stock: pd.Series, market: pd.Series, n: int) -> pd.Series:
    s, m = _common_pair(stock, market)
    out = s.pct_change(n, fill_method=None) - m.pct_change(n, fill_method=None)
    return _reindex_like(out, stock)


def relative_strength_ratio(stock: pd.Series, market: pd.Series, n: int) -> pd.Series:
    """(1+R_stock)/(1+R_market) - 1 — geometric relative strength over n days."""
    s, m = _common_pair(stock, market)
    denom = 1 + m.pct_change(n, fill_method=None)
    out = (1 + s.pct_change(n, fill_method=None)) / denom.replace(0, np.nan) - 1
    return _reindex_like(out, stock)


def rs_ratio_slope(stock, market, n=RS_RATIO_HORIZON, lookback=5):
    """Slope of the relative-strength ratio - accelerating relative strength."""
    rs = relative_strength_ratio(stock, market, n)
    return rs / rs.shift(lookback).replace(0, np.nan) - 1


def rolling_corr(stock: pd.Series, market: pd.Series, window: int = CORR_WINDOW) -> pd.Series:
    s, m = _common_pair(stock, market)
    out = daily_returns(s).rolling(window=window).corr(daily_returns(m))
    return _reindex_like(out, stock)


def vol_to_market_vol(stock: pd.Series, market: pd.Series, window: int = VOL_RATIO_WINDOW) -> pd.Series:
    s, m = _common_pair(stock, market)
    sv = rolling_vol(s, window)
    mv = rolling_vol(m, window)
    out = sv / mv.replace(0, np.nan)
    return _reindex_like(out, stock)


def cross_sectional_rank(returns_panel: pd.DataFrame, lookback: int = RS_RATIO_HORIZON) -> pd.DataFrame:
    """Percentile rank (0..1) of each symbol's trailing `lookback`-day cumulative
    return within the cross-section, per row (columns = symbols)."""
    cumulative = (1 + returns_panel).rolling(lookback).apply(np.prod, raw=True) - 1
    return cumulative.rank(axis=1, pct=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

FEATURE_PREFIX = "rel"

def list_market_features() -> list[str]:
    """Deterministic, versioned list of market/risk feature column names."""
    cols = [
        "vol_20d", "vol_60d",
        f"down_vol_{DOWN_VOL_WINDOW}d",
        f"vol_pctile_{VOL_PCTILE_RANK_WINDOW}d",
        f"vol_exp_{VOL_EXP_LOOKBACK}d",
        f"hl_range_{HL_RANGE_WINDOW}d",
        f"gap_freq_{GAP_WINDOW}d",
        f"max_dd_{MAX_DRAWDOWN_WINDOW}d",
    ]
    for n in REL_RET_HORIZONS:
        cols.append(f"{FEATURE_PREFIX}_ret_{DEFAULT_MARKET_NAME}_{n}d")
    cols.append(f"{FEATURE_PREFIX}_rs_ratio_{DEFAULT_MARKET_NAME}_{RS_RATIO_HORIZON}d")
    cols.append(f"{FEATURE_PREFIX}_rs_slope_{DEFAULT_MARKET_NAME}_{RS_RATIO_HORIZON}d")
    cols.append(f"{FEATURE_PREFIX}_corr_{DEFAULT_MARKET_NAME}_{CORR_WINDOW}d")
    cols.append(f"{FEATURE_PREFIX}_vol_ratio_{DEFAULT_MARKET_NAME}_{VOL_RATIO_WINDOW}d")
    return sorted(set(cols))


def _merge_market(df: pd.DataFrame, market_close: pd.Series) -> pd.Series:
    return market_close.reindex(df.index)


def add_market_features(df: pd.DataFrame, market_close: pd.Series) -> pd.DataFrame:
    """Attach market-relative + volatility/risk features to a copy of the frame.

    `market_close`: price Series of the reference (NIFTY 50). Reindexed to the
    stock's index; rows without market data stay NaN (no backfill).
    """
    out = df.copy()
    close = out["Close"]
    mkt = _merge_market(out, market_close)

    features: dict[str, pd.Series] = {
        "vol_20d": rolling_vol(close, 20),
        "vol_60d": rolling_vol(close, 60),
        f"down_vol_{DOWN_VOL_WINDOW}d": downside_vol(close, DOWN_VOL_WINDOW),
        f"vol_pctile_{VOL_PCTILE_RANK_WINDOW}d": vol_percentile(close),
        f"vol_exp_{VOL_EXP_LOOKBACK}d": vol_expansion(close),
        f"hl_range_{HL_RANGE_WINDOW}d": high_low_range(out),
        f"gap_freq_{GAP_WINDOW}d": gap_frequency(out),
        f"max_dd_{MAX_DRAWDOWN_WINDOW}d": max_drawdown(close),
    }
    for n in REL_RET_HORIZONS:
        features[f"{FEATURE_PREFIX}_ret_{DEFAULT_MARKET_NAME}_{n}d"] = excess_return(close, mkt, n)
    features[f"{FEATURE_PREFIX}_rs_ratio_{DEFAULT_MARKET_NAME}_{RS_RATIO_HORIZON}d"] = (
        relative_strength_ratio(close, mkt, RS_RATIO_HORIZON)
    )
    features[f"{FEATURE_PREFIX}_rs_slope_{DEFAULT_MARKET_NAME}_{RS_RATIO_HORIZON}d"] = (
        rs_ratio_slope(close, mkt, RS_RATIO_HORIZON)
    )
    features[f"{FEATURE_PREFIX}_corr_{DEFAULT_MARKET_NAME}_{CORR_WINDOW}d"] = rolling_corr(close, mkt)
    features[f"{FEATURE_PREFIX}_vol_ratio_{DEFAULT_MARKET_NAME}_{VOL_RATIO_WINDOW}d"] = vol_to_market_vol(close, mkt)

    for name, series in features.items():
        out[name] = series

    out.attrs["feature_version"] = settings.ml_feature_version
    return out


def add_sector_relative_features(
    df: pd.DataFrame,
    sector_close: pd.Series,
    sector_name: str,
) -> pd.DataFrame:
    """Attach stock-vs-SECTOR relative features (RESEARCH ONLY — Task 28 lever).

    Same relative-strength/volatility family as `add_market_features` but
    benchmarked against a sector index (e.g. NIFTY IT for TCS/INFY, NIFTY Bank
    for HDFCBANK) instead of NIFTY 50. Causal (reindexed to the stock index,
    NaN where the sector has no bar). Only invoked by the opt-in
    `replay_snapshot --sector` lever — production (NIFTY 50 only) is untouched.
    """
    out = df.copy()
    close = out["Close"]
    sec = _merge_market(out, sector_close)

    features: dict[str, pd.Series] = {}
    for n in REL_RET_HORIZONS:
        features[f"{FEATURE_PREFIX}_ret_{sector_name}_{n}d"] = excess_return(close, sec, n)
    features[f"{FEATURE_PREFIX}_rs_ratio_{sector_name}_{RS_RATIO_HORIZON}d"] = (
        relative_strength_ratio(close, sec, RS_RATIO_HORIZON)
    )
    features[f"{FEATURE_PREFIX}_rs_slope_{sector_name}_{RS_RATIO_HORIZON}d"] = (
        rs_ratio_slope(close, sec, RS_RATIO_HORIZON)
    )
    features[f"{FEATURE_PREFIX}_corr_{sector_name}_{CORR_WINDOW}d"] = rolling_corr(close, sec)
    features[f"{FEATURE_PREFIX}_vol_ratio_{sector_name}_{VOL_RATIO_WINDOW}d"] = (
        vol_to_market_vol(close, sec, VOL_RATIO_WINDOW)
    )

    for name, series in features.items():
        out[name] = series

    out.attrs["feature_version"] = settings.ml_feature_version
    return out