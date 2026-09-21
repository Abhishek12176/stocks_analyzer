"""Beta / market-exposure model (Task 5).

Decomposes a stock's return as  Stock = Market + Residual:

- beta     : rolling OLS sensitivity of stock daily returns to market (NIFTY 50)
- r_squared: rolling R² from the same regression
- alpha    : regression intercept (per-period, not annualised)
- residual : stock_return - beta * market_return  (idiosyncratic component)
- residual_vol: rolling std of the residual (idiosyncratic risk)

All quantities are causal (past window only), computed on common non-NaN
market dates. Sliding-window OLS implemented directly (no external solver).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import settings

DEFAULT_WINDOW = 60
RESIDUAL_VOL_WINDOW = 20
TRADING_DAYS = 252


def _common_returns(stock: pd.Series, market: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Align both price series on common non-NaN dates -> daily returns (no NaNs)."""
    frame = pd.concat([stock, market], axis=1, join="outer").dropna()
    sr = frame.iloc[:, 0].pct_change(fill_method=None)
    mr = frame.iloc[:, 1].pct_change(fill_method=None)
    mask = sr.notna() & mr.notna()
    return sr[mask], mr[mask]


def _ols_window(rs: np.ndarray, rm: np.ndarray) -> tuple[float, float, float]:
    """Single-window OLS: returns (slope/beta, intercept/alpha, R²)."""
    n = len(rs)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    sx = float(np.sum(rm))
    sy = float(np.sum(rs))
    sxy = float(np.dot(rs, rm))
    sxx = float(np.dot(rm, rm))
    denom = n * sxx - sx ** 2
    if denom == 0:
        return float("nan"), float("nan"), float("nan")
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    ss_res = float(np.sum((rs - (intercept + slope * rm)) ** 2))
    ss_tot = float(np.sum((rs - np.mean(rs)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, intercept, r2


def _rolling_ols(rs: pd.Series, rm: pd.Series, window: int) -> pd.DataFrame:
    """Sliding-window OLS over aligned returns; returns df indexed like `rs`
    (NaN in the first `window-1` warm-up rows)."""
    rs_arr = rs.to_numpy(dtype=float)
    rm_arr = rm.to_numpy(dtype=float)
    n = len(rs_arr)
    idx = rs.index
    if n < window:
        return pd.DataFrame(
            {"beta": pd.Series(np.nan, index=idx),
             "alpha": pd.Series(np.nan, index=idx),
             "r2": pd.Series(np.nan, index=idx)}
        )
    beta = np.full(n, np.nan)
    alpha = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    for i in range(window - 1, n):
        b, a, r = _ols_window(rs_arr[i - window + 1: i + 1], rm_arr[i - window + 1: i + 1])
        beta[i], alpha[i], r2[i] = b, a, r
    return pd.DataFrame({"beta": beta, "alpha": alpha, "r2": r2}, index=idx)


def add_beta_features(
    df: pd.DataFrame,
    market_close: pd.Series,
    window: int = DEFAULT_WINDOW,
    residual_vol_window: int = RESIDUAL_VOL_WINDOW,
) -> pd.DataFrame:
    """Attach beta / exposure features to a copy of the OHLCV frame."""
    out = df.copy()
    close = out["Close"]
    sr, mr = _common_returns(close, market_close)

    ols = _rolling_ols(sr, mr, window)

    beta = ols["beta"].reindex(close.index)
    alpha_o = ols["alpha"].reindex(close.index)
    r2 = ols["r2"].reindex(close.index)

    mkt_ret = mr.reindex(close.index)
    residual = sr.reindex(close.index) - beta * mkt_ret
    residual_vol = residual.rolling(residual_vol_window).std(ddof=0)
    residual_vol_pctile = residual_vol.rolling(252).apply(
        lambda x: float(np.mean(x <= x[-1])), raw=True
    )

    for name, series in {
        "beta": beta,
        "beta_alpha": alpha_o,
        "beta_r2": r2,
        "residual_ret": residual,
        "residual_vol": residual_vol,
        "residual_vol_pctile": residual_vol_pctile,
    }.items():
        out[name] = series

    out.attrs["feature_version"] = settings.ml_feature_version
    return out


def list_beta_features() -> list[str]:
    return ["beta", "beta_alpha", "beta_r2", "residual_ret", "residual_vol", "residual_vol_pctile"]