"""Prediction-target builder + chronological splits (Task 7).

The supervised target for the 20-day system is built WITHOUT lookahead into
features: for a row at day T,

- `target_ret_{h}d` = close[T+h] / close[T] - 1   (forward return)
- `target_up_{h}d`  = 1 if that forward return > 0, else 0  (up/down label)

The forward return intentionally looks *forward* (it is the label, stored at
row T and evaluated after the horizon elapses); the no-leakage contract is
that *feature* rows at T must use only data <= T (enforced by every feature
engine). Backtest task (Task 8) evaluates the label after the horizon.

Splits are strictly chronological:
- `split_dates`  : a single contiguous train/test frontier by date.
- `walk_forward_splits`: expanding-window walk-forward folds (train is always
  entirely before test) ready for the purged backtest engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UP = 1
DOWN = 0

DEFAULT_HORIZON = 20


def forward_return(close: pd.Series, horizon: int = DEFAULT_HORIZON) -> pd.Series:
    """20d (or `horizon`-day) forward return stored at row T."""
    return close.shift(-horizon) / close.replace(0, np.nan) - 1


def forward_label(close: pd.Series, horizon: int = DEFAULT_HORIZON) -> pd.Series:
    """1 = up / 0 = down for the forward window (NaN on the last `horizon` rows)."""
    ret = forward_return(close, horizon)
    out = (ret > 0).astype(int)
    out.loc[ret.isna()] = np.nan
    return out


def add_target(df: pd.DataFrame, horizon: int = DEFAULT_HORIZON, close_col: str = "Close") -> pd.DataFrame:
    """Attach forward-return + up/down target columns to a copy of the frame."""
    out = df.copy()
    close = out[close_col]
    out[f"target_ret_{horizon}d"] = forward_return(close, horizon)
    out[f"target_up_{horizon}d"] = forward_label(close, horizon)
    return out


def add_volatility_target(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    close_col: str = "Close",
    k: float = 0.5,
) -> pd.DataFrame:
    """Attach a volatility-thresholded TRAINING label (full binary kept intact).

    Causal, data <= T only: a row at day T gets a *clean* up/down label only
    when its forward return clears the noise band:
        sigma_T = 20-day realized vol of daily returns up to and incl. T
        theta_T = k * sigma_T * sqrt(horizon)
        train_up = 1 if target_ret > +theta_T
                 = 0 if target_ret < -theta_T
                 = NaN if |target_ret| <= theta_T   (noise zone -> DROPPED
                                                     from training)
    The original `target_up_{horizon}d` (full binary) is NEVER modified — it
    stays the EVALUATION label so AUC stays comparable across configs.
    Requires the frame to already carry `target_ret_{horizon}d` (from
    `add_target`); `k <= 0` disables the filter and returns the frame as-is.
    """
    out = df.copy()
    if k is None or k <= 0:
        return out
    ret_col = f"target_ret_{horizon}d"
    if ret_col not in out.columns:
        out = add_target(out, horizon=horizon, close_col=close_col)
    close = out[close_col]
    sigma = close.pct_change().rolling(20).std()
    theta = k * sigma * np.sqrt(horizon)
    clean = np.where(
        out[ret_col] > theta,
        1.0,
        np.where(out[ret_col] < -theta, 0.0, np.nan),
    )
    out[f"train_up_{horizon}d"] = clean
    out[f"vol_theta_{horizon}d"] = theta
    return out


def build_dataset(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    feature_cols: list[str] | None = None,
    target_col: str | None = None,
    dropna_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a feature frame into (X, y) for supervised modelling.

    `feature_cols` defaults to every non-target column. Rows without a
    defined target (the final `horizon` rows) are dropped when
    `dropna_target=True` so the model never sees a row with a NaN label.
    """
    work = add_target(df, horizon=horizon) if target_col is None else df
    if target_col is None:
        target_col = f"target_up_{horizon}d"
    if target_col not in work.columns:
        work = add_target(work, horizon=horizon)
    cols = feature_cols if feature_cols is not None else [
        c for c in work.columns if not c.startswith("target_")
    ]
    X = work[list(cols)]
    y = work[target_col]
    if dropna_target:
        mask = y.notna()
        X, y = X[mask], y[mask]
    return X, y


def split_dates(
    index: pd.DatetimeIndex,
    train_end: pd.Timestamp | str,
    test_end: pd.Timestamp | str | None = None,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Contiguous chronological split: train = dates <= train_end, test = later."""
    train_end = pd.Timestamp(train_end)
    train = pd.DatetimeIndex([d for d in index if d <= train_end])
    if test_end is not None:
        test_end = pd.Timestamp(test_end)
        test = pd.DatetimeIndex([d for d in index if train_end < d <= test_end])
    else:
        test = pd.DatetimeIndex([d for d in index if d > train_end])
    return train, test


def walk_forward_splits(
    index: pd.DatetimeIndex,
    test_size: int,
    step: int | None = None,
    min_train: int = 0,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Expanding-window walk-forward folds.

    Each fold is (train, test) with train strictly before test and growing by
    `step` days every fold. Enforces min_train warm-up and skips a trailing
    partial test window.
    """
    step = step or test_size
    index = pd.DatetimeIndex(sorted(index))
    folds: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    pos = min_train
    while pos + test_size <= len(index):
        train = index[:pos]
        test = index[pos:pos + test_size]
        if len(train) > 0:
            folds.append((train, test))
        pos += step
    return folds