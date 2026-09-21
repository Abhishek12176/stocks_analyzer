"""Market regime engine for the 20-day forecasting system (Task 7).

Classifies the *state of the market* at each day into simple, honest,
causal buckets that upstream models can use as features/conditions:

- trend   : bull / bear / sideways   (NIFTY vs SMA50 & SMA200)
- vol     : high / normal / low      (NIFTY 20d realised-vol percentile)
- risk    : risk-on / neutral / risk-off (NIFTY trend + India-VIX level percentile)
- breadth : fraction of universe stocks trading above their own SMA20

Semantics (all causal, trailing windows only — no lookahead):

- `regime_trend`  = 1 if close > SMA50 AND close > SMA200 (bull),
                   -1 if both below (bear), else 0 (sideways).
- `regime_vol`    = 1 when the 20d realised-vol percentile (trailing 252d,
                   ranked on its own history) crosses the high threshold,
                   -1 below the low threshold, else 0.
- `regime_risk`   = 1 only when the NIFTY is in a bull trend AND the India
                   VIX level sits in its low historical percentile bucket;
                   -1 only when bear trend AND high VIX; else 0.
- `regime_breadth` = mean(symbol_closes > SMA20(symbol_closes)) across a
                   cross-sectional panel, per day.

Every group is optional: if its input series is not supplied the column is
simply not produced (graceful disable — the model then has no regime signal
from that group rather than a fabricated one).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import settings
from app.ml.market_features import rolling_percentile, rolling_vol
from app.services.indicator_service import sma

TREND_MA_FAST = 50
TREND_MA_SLOW = 200
VOL_WINDOW = 20
VOL_RANK_WINDOW = 252
VOL_HIGH_Q = 0.7
VOL_LOW_Q = 0.3
VIX_RANK_WINDOW = 252
VIX_RISK_Q = 0.5
BREADTH_MA = 20

BULL = 1
SIDEWAYS = 0
BEAR = -1
RISK_ON = 1
RISK_OFF = -1


def trend_regime(close: pd.Series, fast: int = TREND_MA_FAST, slow: int = TREND_MA_SLOW) -> pd.Series:
    """+1 bull / -1 bear / 0 sideways from close vs SMA50 & SMA200."""
    dist_fast = close / sma(close, fast) - 1
    dist_slow = close / sma(close, slow) - 1
    values = np.select(
        [(dist_fast > 0) & (dist_slow > 0), (dist_fast < 0) & (dist_slow < 0)],
        [BULL, BEAR],
        default=SIDEWAYS,
    )
    out = pd.Series(values, index=close.index, dtype=float)
    out.loc[dist_fast.isna() | dist_slow.isna()] = np.nan
    return out


def vol_regime(
    close: pd.Series,
    vol_window: int = VOL_WINDOW,
    rank_window: int = VOL_RANK_WINDOW,
    high_q: float = VOL_HIGH_Q,
    low_q: float = VOL_LOW_Q,
) -> tuple[pd.Series, pd.Series]:
    """(bucket, percentile): +1 high / -1 low / 0 normal from vol percentile."""
    vol = rolling_vol(close, vol_window, annualize=False)
    pctile = rolling_percentile(vol, rank_window)
    bucket = np.select([pctile >= high_q, pctile <= low_q], [1, -1], default=0)
    out = pd.Series(bucket, index=close.index, dtype=float)
    out.loc[pctile.isna()] = np.nan  # warm-up/unknown -> no regime signal
    return out, pctile


def risk_regime(
    nifty_close: pd.Series,
    vix_close: pd.Series,
    fast: int = TREND_MA_FAST,
    slow: int = TREND_MA_SLOW,
    rank_window: int = VIX_RANK_WINDOW,
    risk_q: float = VIX_RISK_Q,
) -> tuple[pd.Series, pd.Series]:
    """(bucket, vix_percentile): +1 risk-on / -1 risk-off / 0 neutral."""
    trend = trend_regime(nifty_close, fast=fast, slow=slow)
    # Classify VIX on its own calendar first (causal), THEN align to the
    # nifty/stock calendar. Without this, the boolean `on`/`off` masks align
    # to the UNION of the nifty & vix calendars, so when the two calendars
    # diverge (e.g. a long 10y history) the bucket outlives the nifty index
    # and `pd.Series(bucket, index=nifty_close.index)` crashes with a length
    # mismatch. Reindexing after the percentile keeps lengths consistent and
    # never reintroduces lookahead.
    vix_pctile = rolling_percentile(vix_close, rank_window).reindex(trend.index)
    on = (trend == BULL) & (vix_pctile <= risk_q)
    off = (trend == BEAR) & (vix_pctile > risk_q)
    bucket = np.select([on, off], [RISK_ON, RISK_OFF], default=0)
    out = pd.Series(bucket, index=trend.index, dtype=float)
    out.loc[trend.isna() | vix_pctile.isna()] = np.nan
    return out, vix_pctile


def breadth(panel: pd.DataFrame, ma_len: int = BREADTH_MA) -> pd.Series:
    """Fraction of symbols (columns) trading above their own SMA `ma_len`."""
    above = panel > panel.rolling(ma_len, min_periods=1).mean()
    return above.mean(axis=1)


def list_regime_features() -> list[str]:
    return sorted(
        {
            "regime_trend",
            "regime_vol",
            "regime_risk",
            "regime_vix_pctile",
            "regime_nifty_vol_pctile",
            "regime_breadth",
        }
    )


def add_regime_features(
    df: pd.DataFrame,
    nifty_close: pd.Series | None = None,
    vix_close: pd.Series | None = None,
    breadth_series: pd.Series | None = None,
) -> pd.DataFrame:
    """Attach regime columns to a copy of the frame.

    Each group is optional — a missing source simply omits its columns
    (no fabricated regime state). Banner series are classified on their own
    calendars first (causal), then the resulting buckets/percentiles are
    reindexed to the stock calendar, so ragged session mismatches never
    contaminate the long percentile windows.
    """
    out = df.copy()
    index = pd.DatetimeIndex(out.index)

    if nifty_close is not None:
        out["regime_trend"] = trend_regime(nifty_close).reindex(index)
        vol_bucket, vol_pctile = vol_regime(nifty_close)
        out["regime_vol"] = vol_bucket.reindex(index)
        out["regime_nifty_vol_pctile"] = vol_pctile.reindex(index)
        if vix_close is not None:
            risk_bucket, vix_pctile = risk_regime(nifty_close, vix_close)
            out["regime_risk"] = risk_bucket.reindex(index)
            out["regime_vix_pctile"] = vix_pctile.reindex(index)

    if breadth_series is not None:
        out["regime_breadth"] = breadth_series.reindex(index)

    out.attrs["feature_version"] = settings.ml_feature_version
    return out