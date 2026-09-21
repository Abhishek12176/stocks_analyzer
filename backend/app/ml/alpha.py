"""Alpha factor engine (Task 5).

Constructs rank-normalised factor groups and combines them into composite
scores in [0, 1]. Every factor score is a *relative* measure: the value of a
component at day T is the percentile rank of its raw value within a trailing
window (or cross-section) up to T — so it is causal and comparable.

Factor groups:
- momentum      : trailing 20/60/120d rank-normalised returns, weighted composite
- mean_reversal : price vs 20d SMA, RSI, trailing 20d return (oversold -> high score)
- volatility    : inverse of trailing vol percentile (low vol -> high score)
- value         : low PE / low PB / high dividend yield (components optional;
                  any unavailable component is gracefully disabled)

Extensibility: `composite_score()` turns any named component series into a
weighted composite; `FACTOR_GROUPS` tracks each group's OOS-validation status
(validation itself happens in the backtest task — "no sounds-financial-implies-
predictive"). `mark_factor_validated(group)` flips the flag once validated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import settings
from app.ml.market_features import rolling_percentile, rolling_vol

FACTOR_VERSION = settings.ml_feature_version

MOMENTUM_HORIZONS: tuple[int, ...] = (20, 60, 120)
MEAN_REV_MA = 20
MEAN_REV_RSI = 14
MEAN_REV_HORIZON = 20
RANK_WINDOW = 252
VOL_FACTOR_WINDOW = 60


# ---------------------------------------------------------------------------
# Rank normalization
# ---------------------------------------------------------------------------

def rank_normalize(series: pd.Series, window: int = RANK_WINDOW) -> pd.Series:
    """Time-series percentile rank of the trailing-most value (0..1), causal."""
    return rolling_percentile(series, window)


# ---------------------------------------------------------------------------
# Composite helper
# ---------------------------------------------------------------------------

NEUTRAL = 0.5


def composite_score(factors: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    """Weighted average of named factor components; missing rows -> neutral 0.5
    (graceful disable, never fabricated). Returns a Series in [0, 1]."""
    names = [name for name in weights if name in factors]
    if not names:
        return pd.Series(dtype=float)
    idx = factors[next(iter(factors))].index
    cumulative = pd.Series(0.0, index=idx)
    total_weight = 0.0
    for name in names:
        s = factors[name].reindex(idx).fillna(NEUTRAL)
        cumulative = cumulative + weights[name] * s
        total_weight += weights[name]
    return cumulative / total_weight if total_weight > 0 else pd.Series(np.nan, index=idx)


# ---------------------------------------------------------------------------
# Factor groups
# ---------------------------------------------------------------------------

def momentum_factor(
    close: pd.Series,
    horizons: tuple[int, ...] = MOMENTUM_HORIZONS,
    window: int = RANK_WINDOW,
) -> pd.Series:
    components = {}
    weights = {}
    for h in horizons:
        name = f"mom_{h}d"
        if len(close) >= h + 1:
            components[name] = rank_normalize(close.pct_change(h, fill_method=None), window)
            weights[name] = 1.0
    return composite_score(components, weights)


def mean_reversion_factor(
    close: pd.Series,
    ma_period: int = MEAN_REV_MA,
    rsi_period: int = MEAN_REV_RSI,
    ret_horizon: int = MEAN_REV_HORIZON,
    window: int = RANK_WINDOW,
    rsi: pd.Series | None = None,
) -> pd.Series:
    """High score when oversold: price far below MA, low RSI, recent fall."""
    from app.services.indicator_service import sma

    components: dict[str, pd.Series] = {}
    weights: dict[str, float] = {}

    dist = close / sma(close, ma_period) - 1
    if dist.notna().any():
        components["dist_ma"] = 1 - rank_normalize(dist, window)
        weights["dist_ma"] = 1.0

    rsi_series = rsi if rsi is not None else None
    if rsi_series is None:
        from app.services.indicator_service import rsi as rsi_fn
        rsi_series = rsi_fn(close, rsi_period)
    if rsi_series.notna().any():
        components["rsi"] = 1 - rank_normalize(rsi_series, window)
        weights["rsi"] = 1.0

    if len(close) >= ret_horizon + 1:
        recent_ret = close.pct_change(ret_horizon, fill_method=None)
        components["recent_ret"] = 1 - rank_normalize(recent_ret, window)
        weights["recent_ret"] = 1.0

    return composite_score(components, weights)


def volatility_factor(
    close: pd.Series,
    window: int = VOL_FACTOR_WINDOW,
    rank_window: int = RANK_WINDOW,
) -> pd.Series:
    """Low trailing volatility -> high score (inverse vol percentile)."""
    vol = rolling_vol(close, window, annualize=False)
    return 1 - rolling_percentile(vol, rank_window)


def value_factor(
    pe: pd.Series | None = None,
    pb: pd.Series | None = None,
    dividend_yield: pd.Series | None = None,
    window: int = RANK_WINDOW,
) -> pd.Series | None:
    """Composite value score: low PE, low PB, high dividend yield.
    Any missing component is disabled (returns None only if all absent)."""
    components: dict[str, pd.Series] = {}
    weights: dict[str, float] = {}
    if pe is not None and pe.notna().any():
        components["pe"] = 1 - rank_normalize(pe, window)
        weights["pe"] = 1.0
    if pb is not None and pb.notna().any():
        components["pb"] = 1 - rank_normalize(pb, window)
        weights["pb"] = 1.0
    if dividend_yield is not None and dividend_yield.notna().any():
        components["div_yield"] = rank_normalize(dividend_yield, window)
        weights["div_yield"] = 1.0
    if not components:
        return None
    return composite_score(components, weights)


# ---------------------------------------------------------------------------
# Validation metadata + orchestrator
# ---------------------------------------------------------------------------

FACTOR_GROUPS: dict[str, dict] = {
    "momentum": {"components": ["mom_20d", "mom_60d", "mom_120d"], "oos_validated": False},
    "mean_reversion": {"components": ["dist_ma", "rsi", "recent_ret"], "oos_validated": False},
    "volatility": {"components": ["vol_inv_pctile"], "oos_validated": False},
    "value": {"components": ["pe", "pb", "div_yield"], "oos_validated": False},
}


def factor_metadata() -> dict[str, dict]:
    return {name: dict(meta) for name, meta in FACTOR_GROUPS.items()}


def mark_factor_validated(name: str) -> None:
    """Flip the OOS-validation flag once a factor has been validated in backtest."""
    if name not in FACTOR_GROUPS:
        raise KeyError(f"Unknown factor group: {name}")
    FACTOR_GROUPS[name]["oos_validated"] = True


def add_alpha_features(
    df: pd.DataFrame,
    pe: pd.Series | None = None,
    pb: pd.Series | None = None,
    dividend_yield: pd.Series | None = None,
) -> pd.DataFrame:
    """Attach factor-score columns to a copy of the OHLCV frame."""
    out = df.copy()
    close = out["Close"]

    out["alpha_momentum"] = momentum_factor(close)
    out["alpha_mean_reversion"] = mean_reversion_factor(close)
    out["alpha_volatility"] = volatility_factor(close)

    value = value_factor(pe, pb, dividend_yield)
    if value is not None:
        out["alpha_value"] = value.reindex(out.index)

    out.attrs["feature_version"] = settings.ml_feature_version
    return out


def list_alpha_features() -> list[str]:
    return sorted(["alpha_momentum", "alpha_mean_reversion", "alpha_volatility", "alpha_value"])