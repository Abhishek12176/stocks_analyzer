"""Slow-moving macro/global feature integration (Task 6).

Macro series (Brent, Gold, USD/INR, India VIX, VIX, S&P 500, ...) are aligned
to the stock's trading calendar and turned into slow-moving features:

- `mac_{sid}_ret_{h}d` : `h`-day return of the macro/global series
- `mac_{sid}_vol_{w}d` : `w`-day annualised rolling volatility
- `mac_{sid}_dd_{w}d`  : `w`-day rolling max drawdown

Rules:
- Series are reindexed to the stock index — rows where the macro series has no
  observation stay NaN (never back-filled, no fabricated data). In the
  downstream model these map to the "graceful disable / neutral factor" path.
- Every feature uses only data known at/prior to the row date (causal).
"""

from __future__ import annotations

import pandas as pd

from app.config import settings
from app.ml.market_features import max_drawdown, rolling_vol

MAC_HORIZONS: tuple[int, ...] = (5, 20, 60)
MAC_VOL_WINDOW = 60
MAC_DRAWDOWN_WINDOW = 60


def _macro_reindex(macro_close: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    return macro_close.reindex(index)


def list_macro_features(series_ids: list[str] | None = None) -> list[str]:
    """Deterministic feature list for the given macro series ids."""
    cols: list[str] = []
    for sid in sorted(series_ids or []):
        for h in MAC_HORIZONS:
            cols.append(f"mac_{sid}_ret_{h}d")
        cols.append(f"mac_{sid}_vol_{MAC_VOL_WINDOW}d")
        cols.append(f"mac_{sid}_dd_{MAC_DRAWDOWN_WINDOW}d")
    return sorted(set(cols))


def add_macro_features(
    df: pd.DataFrame,
    macro_closes: dict[str, pd.Series],
    horizons: tuple[int, ...] = MAC_HORIZONS,
    vol_window: int = MAC_VOL_WINDOW,
    drawdown_window: int = MAC_DRAWDOWN_WINDOW,
) -> pd.DataFrame:
    """Attach slow-moving macro/global features to a copy of the frame.

    `macro_closes`: {series_id: pd.Series of close prices indexed by date}.
    Missing series or missing dates stay NaN (never fabricated).
    """
    out = df.copy()
    index = pd.DatetimeIndex(out.index)

    features: dict[str, pd.Series] = {}
    for sid, macro_close in macro_closes.items():
        aligned = _macro_reindex(macro_close, index).astype(float)
        for h in horizons:
            features[f"mac_{sid}_ret_{h}d"] = aligned.pct_change(h, fill_method=None)
        features[f"mac_{sid}_vol_{vol_window}d"] = rolling_vol(aligned, vol_window)
        features[f"mac_{sid}_dd_{drawdown_window}d"] = max_drawdown(aligned, drawdown_window)

    for name, series in features.items():
        out[name] = series

    out.attrs["feature_version"] = settings.ml_feature_version
    return out