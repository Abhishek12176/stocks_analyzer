"""F&O / options-derivatives feature layer (Task 7) — graceful disable.

Free F&O tape data is noisy and frequently unavailable, so the layer has a
hard contract: **every failure disables it cleanly** — `is_available=False`
and feature columns are simply not fabricated.

Path A — synthetic/metrics snapshots:
    `build_options_features(df, snapshots)` merges per-date snapshot dicts
    (keys `pcr`, `total_oi`, `atm_iv`, `delta`, `gamma`, `theta`, `vega`,
    `rho`, plus an `available_at`/`as_of` date) into point-in-time columns
    `opt_*` using the same causal merge_asof semantics as fundamentals.

Path B — live yfinance chain (optional):
    `chain_to_metrics()` turns one option chain (calls/puts frames with
    strike/openInterest/impliedVolatility) into a metrics dict including
    Black-Scholes greeks for the reference ATM call. `get_options_inputs()`
    is the thin network boundary: any exception -> graceful disable dict.

Nothing is backfilled; a snapshot published on day D is only visible from
day D onwards.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.config import settings
from app.ml.fundamental_features import pit_series, _prepare_snapshots

OPTION_COLUMNS: dict[str, str] = {
    "pcr": "pcr",
    "total_oi": "total_oi",
    "atm_iv": "atm_iv",
    "delta": "delta",
    "gamma": "gamma",
    "theta": "theta",
    "vega": "vega",
    "rho": "rho",
}

def list_options_features() -> list[str]:
    return [f"opt_{c}" for c in OPTION_COLUMNS.values()]


# ---------------------------------------------------------------------------
# Black-Scholes greeks (reference ATM call)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_greeks(
    spot: float,
    strike: float,
    ttm_years: float,
    sigma: float,
    risk_free: float | None = None,
    dividend_yield: float = 0.0,
) -> dict[str, float]:
    """Black-Scholes greeks for a European CALL.

    Returns {delta, gamma, theta, vega, rho} where theta is per calendar day
    and vega is per 1.0 point of volatility (standard conventions).
    """
    if risk_free is None:
        risk_free = settings.ml_risk_free_rate
    if sigma is None or sigma <= 0 or ttm_years is None or ttm_years <= 0:
        raise ValueError("sigma and ttm_years must be positive for greeks")
    r = float(risk_free)
    q = float(dividend_yield)
    sq = math.sqrt(ttm_years)
    d1 = (math.log(spot / strike) + (r - q + sigma * sigma / 2.0) * ttm_years) / (sigma * sq)
    d2 = d1 - sigma * sq
    delta = math.exp(-q * ttm_years) * _norm_cdf(d1)
    gamma = math.exp(-q * ttm_years) * _norm_pdf(d1) / (max(1e-12, spot) * sigma * sq)
    theta = (
        -math.exp(-q * ttm_years) * _norm_pdf(d1) * sigma / (2.0 * sq)
        - r * strike * math.exp(-r * ttm_years) * _norm_cdf(d2)
        + q * spot * math.exp(-q * ttm_years) * _norm_cdf(d1)
    ) / 365.0
    vega = spot * math.exp(-q * ttm_years) * _norm_pdf(d1) * sq
    rho = strike * ttm_years * math.exp(-r * ttm_years) * _norm_cdf(d2)
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}


# ---------------------------------------------------------------------------
# yfinance chain -> metrics (Path B)
# ---------------------------------------------------------------------------

def chain_to_metrics(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: float,
    expiry: str | date | None = None,
    days_to_expiry: int | None = None,
) -> dict[str, float]:
    """Derive {pcr, total_oi, atm_iv, delta, gamma, theta, vega, rho} from a
    chain. Requires calls/puts carrying 'strike', 'openInterest' and
    'impliedVolatility' columns. Falls back to NaN (never invents numbers).
    """
    calls = calls[["strike", "openInterest", "impliedVolatility"]].dropna() if not calls.empty else calls
    puts = puts[["strike", "openInterest", "impliedVolatility"]].dropna() if not puts.empty else puts

    total_call_oi = float(calls["openInterest"].sum(skipna=True)) if not calls.empty else np.nan
    total_put_oi = float(puts["openInterest"].sum(skipna=True)) if not puts.empty else np.nan
    pcr = total_put_oi / total_call_oi if (total_call_oi and total_call_oi > 0) else np.nan

    chain = pd.concat([calls, puts]) if not (calls.empty and puts.empty) else None
    atm_iv = np.nan
    atm_strike = np.nan
    if chain is not None and not chain.empty:
        row = chain.iloc[(chain["strike"] - spot).abs().to_numpy().argmin()]
        atm_strike = float(row["strike"])
        atm_iv = float(row["impliedVolatility"])

    greeks = {"delta": np.nan, "gamma": np.nan, "theta": np.nan, "vega": np.nan, "rho": np.nan}
    ttm = np.nan
    if expiry is not None and days_to_expiry is None:
        d0 = pd.Timestamp(expiry)
        days_to_expiry = (d0 - pd.Timestamp.now().normalize()).days
    if days_to_expiry is not None:
        ttm = max(days_to_expiry, 1) / 365.0
    if not math.isnan(atm_iv) and spot > 0 and not math.isnan(ttm):
        try:
            greeks = bs_greeks(spot=spot, strike=atm_strike, ttm_years=ttm, sigma=atm_iv)
        except ValueError:
            greeks = {"delta": np.nan, "gamma": np.nan, "theta": np.nan, "vega": np.nan, "rho": np.nan}

    return {
        "pcr": pcr,
        "total_oi": total_oi_or_nan(total_call_oi, total_put_oi),
        "atm_iv": atm_iv,
        **greeks,
    }


def total_oi_or_nan(call_oi: float, put_oi: float) -> float:
    vals = [v for v in (call_oi, put_oi) if not math.isnan(v)]
    return float(sum(vals)) if vals else np.nan


def _load_option_chain(symbol: str, expiry: str | None = None) -> dict[str, Any]:
    """Network boundary (yfinance). Separated so tests can monkeypatch it."""
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    expiries = list(ticker.options or [])
    if not expiries:
        raise ValueError(f"no options expiries for {symbol}")
    expiry = expiry or expiries[0]
    chain = ticker.option_chain(expiry)
    return {"symbol": symbol, "expiry": expiry, "calls": chain.calls, "puts": chain.puts}


def get_options_inputs(
    symbol: str,
    spot: float,
    days_to_expiry: int | None = None,
    expiry: str | None = None,
) -> dict[str, Any]:
    """Fetch a live options snapshot — or report a graceful disable.

    Returns: {symbol, is_available, fetched_at, error, expiry, metrics}.
    """
    result: dict[str, Any] = {
        "symbol": symbol,
        "is_available": False,
        "fetched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "error": None,
        "expiry": expiry,
        "metrics": None,
    }
    if not settings.ml_options_enabled:
        result["error"] = "options layer disabled (ml_options_enabled=False)"
        return result
    try:
        raw = _load_option_chain(symbol, expiry)
        result["expiry"] = raw["expiry"]
        if days_to_expiry is None:
            days_to_expiry = (pd.Timestamp(raw["expiry"]) - pd.Timestamp.now().normalize()).days
        metrics = chain_to_metrics(raw["calls"], raw["puts"], spot, expiry=raw["expiry"])
        metrics["days_to_expiry"] = float(days_to_expiry)
        result["metrics"] = metrics
        result["is_available"] = not math.isnan(metrics.get("pcr", np.nan)) or not math.isnan(
            metrics.get("total_oi", np.nan)
        ) or not math.isnan(metrics.get("atm_iv", np.nan))
        if not result["is_available"]:
            result["error"] = "chain loaded but no usable metrics (all NaN)"
    except Exception as exc:  # noqa: BLE001 — graceful disable on any failure
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


# ---------------------------------------------------------------------------
# Orchestrator (Path A)
# ---------------------------------------------------------------------------

def _prepare_options_snapshots(
    snapshot: dict[str, Any] | None,
    snapshots: list[dict[str, Any]] | None,
    default_available_at: pd.Timestamp,
) -> list[dict[str, Any]]:
    return _prepare_snapshots(snapshot, snapshots, default_available_at)


def add_options_features(
    df: pd.DataFrame,
    snapshot: dict[str, Any] | None = None,
    snapshots: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Attach PIT `opt_*` columns to a copy of the frame from options
    snapshots. Missing keys are skipped; an absent chain simply yields no
    `opt_*` columns (graceful disable).
    """
    out = df.copy()
    defaults = out.index.max()
    prepared = _prepare_options_snapshots(snapshot, snapshots, defaults)

    seen_keys = {k for s in prepared for k in s if k in OPTION_COLUMNS}
    if not seen_keys:
        # no options data at all -> carry on without opt_* columns
        out.attrs["feature_version"] = settings.ml_feature_version
        return out

    index = pd.DatetimeIndex(out.index)
    for raw_key, col in OPTION_COLUMNS.items():
        if raw_key not in seen_keys:
            continue
        out[f"opt_{col}"] = pit_series(prepared, raw_key, index)
    out.attrs["feature_version"] = settings.ml_feature_version
    return out