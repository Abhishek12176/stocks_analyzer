"""Data ingestion layer for the 20-day forecasting system (Task 2).

Sources (all free, via yfinance):
- NSE stocks  : adjusted 5y OHLCV (task-level function `fetch_nse_ohlcv`)
- Indices     : NIFTY 50, Bank Nifty, India VIX
- Global      : S&P 500, Nasdaq, Nikkei 225, Hang Seng, CBOE VIX
- Macro       : Brent crude, Gold futures, USD/INR

Design rules:
- Every source carries a freshness timestamp (`fetched_at`) + `is_available`.
- Network failures never crash the caller: failed sources are marked
  `is_available: False` with a readable `error` and are simply excluded from
  alignment ("graceful disable"); nothing is fabricated.
- Raw OHLCV frames are cached in `cache_service.market_cache` (DataFrames),
  so later feature-engineering tasks can reuse them without re-downloading.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from app.services.cache_service import cache_service
from app.utils.validators import add_exchange_suffix


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SERIES_REGISTRY: dict[str, dict] = {
    # India indices
    "nifty50": {"ticker": "^NSEI", "category": "index", "name": "NIFTY 50"},
    "banknifty": {"ticker": "^NSEBANK", "category": "index", "name": "NIFTY Bank"},
    "cnxit": {"ticker": "^CNXIT", "category": "index", "name": "NIFTY IT"},
    "india_vix": {"ticker": "^INDIAVIX", "category": "index_vol", "name": "India VIX"},
    # Global equities
    "snp500": {"ticker": "^GSPC", "category": "global", "name": "S&P 500"},
    "nasdaq": {"ticker": "^IXIC", "category": "global", "name": "Nasdaq Composite"},
    "nikkei": {"ticker": "^N225", "category": "global", "name": "Nikkei 225"},
    "hangseng": {"ticker": "^HSI", "category": "global", "name": "Hang Seng"},
    "vix": {"ticker": "^VIX", "category": "global_vol", "name": "CBOE Volatility Index"},
    # Macro
    "brent": {"ticker": "BZ=F", "category": "macro", "name": "Brent Crude"},
    "gold": {"ticker": "GC=F", "category": "macro", "name": "Gold Futures"},
    "usd_inr": {"ticker": "USDINR=X", "category": "macro", "name": "USD/INR"},
}

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def get_available_series() -> list[dict]:
    """Return the list of known market series (id + name + category + ticker)."""
    return [
        {"series_id": sid, **meta}
        for sid, meta in SERIES_REGISTRY.items()
    ]


# ---------------------------------------------------------------------------
# Low-level helpers (the only place yfinance is touched)
# ---------------------------------------------------------------------------

def _clean_ohlcv(df: pd.DataFrame | None) -> pd.DataFrame:
    """Normalise a yfinance frame: single-level columns, numeric, dated index.

    yfinance DailyBars can return a tz-aware index (Asia/Kolkata, America/...);
    it is converted to a tz-naive *local wall-clock* date index so every source
    shares one calendar and cross-source joins (align_closes, market features)
    never mix aware/naive datetimes.
    """
    if df is None:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    out = pd.DataFrame(index=idx)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col].to_numpy(), errors="coerce")
    if "Adj Close" in df.columns:
        out["Adj Close"] = pd.to_numeric(df["Adj Close"].to_numpy(), errors="coerce")
    else:
        out["Adj Close"] = out["Close"].astype(float)
    if "Close" in out.columns and not out.empty:
        out = out.dropna(subset=["Close"])
    return out.sort_index()


def _load_series_data(
    series: dict, period: str = "5y", interval: str = "1d"
) -> pd.DataFrame:
    """Download a registry series via yfinance (network boundary for tests)."""
    df = yf.Ticker(series["ticker"]).history(
        period=period, interval=interval, auto_adjust=False
    )
    return _clean_ohlcv(df)


def _load_stock_data(symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Download adjusted NSE OHLCV for a stock (network boundary for tests)."""
    df = yf.Ticker(add_exchange_suffix(symbol)).history(
        period=period, interval=interval, auto_adjust=False
    )
    return _clean_ohlcv(df)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _get_series_frame(
    series_id: str, period: str = "5y", interval: str = "1d"
) -> tuple[pd.DataFrame | None, dict]:
    """Return (cached DataFrame, status). Status holds `is_available` + error."""
    series = SERIES_REGISTRY.get(series_id)
    if series is None:
        return None, {"is_available": False, "error": f"Unknown series_id: {series_id}"}

    cache_key = f"mdl_{series_id}_{period}_{interval}"
    cached = cache_service.get(cache_service.market_cache, cache_key)
    if isinstance(cached, pd.DataFrame):
        return cached, {"is_available": True}

    try:
        df = _load_series_data(series, period=period, interval=interval)
        if df is None or df.empty:
            raise ValueError("no data returned")
        cache_service.set(cache_service.market_cache, cache_key, df)
        return df, {"is_available": True}
    except Exception as exc:  # noqa: BLE001 — graceful disable, never crash caller
        return None, {
            "is_available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _frame_to_records(df: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    for idx, row in df[OHLCV_COLUMNS].iterrows():
        records.append(
            {
                "date": idx.isoformat(),
                "open": None if pd.isna(row["Open"]) else round(float(row["Open"]), 4),
                "high": None if pd.isna(row["High"]) else round(float(row["High"]), 4),
                "low": None if pd.isna(row["Low"]) else round(float(row["Low"]), 4),
                "close": None if pd.isna(row["Close"]) else round(float(row["Close"]), 4),
                "adj_close": None if pd.isna(row["Adj Close"]) else round(float(row["Adj Close"]), 4),
                "volume": None if pd.isna(row["Volume"]) else int(round(float(row["Volume"]))),
            }
        )
    return records


def fetch_series_history(
    series_id: str, period: str = "5y", interval: str = "1d"
) -> dict:
    """Fetch OHLCV history for one registry series (graceful on failure)."""
    series = SERIES_REGISTRY.get(series_id, {})
    df, meta = _get_series_frame(series_id, period=period, interval=interval)

    payload: dict = {
        "series_id": series_id,
        "name": series.get("name", series_id),
        "category": series.get("category", "unknown"),
        "ticker": series.get("ticker", ""),
        "is_available": meta.get("is_available", False),
        "fetched_at": utc_now_iso(),
        "error": meta.get("error"),
        "rows": 0,
        "data_start": None,
        "data_end": None,
        "history": [],
    }
    if df is not None:
        payload.update(
            {
                "is_available": True,
                "rows": int(len(df)),
                "data_start": df.index[0].isoformat(),
                "data_end": df.index[-1].isoformat(),
                "history": _frame_to_records(df),
            }
        )
    return payload


def fetch_nse_ohlcv(symbol: str, period: str = "5y", interval: str = "1d") -> dict:
    """Fetch adjusted NSE OHLCV (5y default) for a single stock, gracefully."""
    symbol = symbol.upper()
    cache_key = f"ohlcv_{symbol}_{period}_{interval}"
    cached = cache_service.get(cache_service.market_cache, cache_key)
    if isinstance(cached, pd.DataFrame):
        df = cached
    else:
        try:
            df = _load_stock_data(symbol, period=period, interval=interval)
            if df is None or df.empty:
                raise ValueError("no data returned")
            cache_service.set(cache_service.market_cache, cache_key, df)
        except Exception as exc:  # noqa: BLE001
            return {
                "symbol": symbol,
                "is_available": False,
                "fetched_at": utc_now_iso(),
                "error": f"{type(exc).__name__}: {exc}",
                "rows": 0,
                "data_start": None,
                "data_end": None,
                "history": [],
            }

    return {
        "symbol": symbol,
        "is_available": True,
        "fetched_at": utc_now_iso(),
        "error": None,
        "rows": int(len(df)),
        "data_start": df.index[0].isoformat(),
        "data_end": df.index[-1].isoformat(),
        "history": _frame_to_records(df),
    }


def align_closes(series_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Outer-join daily Close of every series on its date index.

    Later series start dates surface as NaN for early dates (no lookahead).
    """
    aligned = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    for sid, df in series_map.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        s = df["Close"].astype(float).rename(sid)
        idx = pd.DatetimeIndex(s.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        s.index = idx
        aligned = aligned.join(s, how="outer")
    return aligned.sort_index()


def get_market_context(period: str = "5y", interval: str = "1d") -> dict:
    """Fetch every registry series and align their closes (for regime/market features)."""
    frames: dict[str, pd.DataFrame] = {}
    sources: dict[str, dict] = {}
    fetched_at = utc_now_iso()

    for sid, series in SERIES_REGISTRY.items():
        df, meta = _get_series_frame(sid, period=period, interval=interval)
        sources[sid] = {
            "name": series["name"],
            "category": series["category"],
            "ticker": series["ticker"],
            "is_available": meta.get("is_available", False),
            "rows": int(len(df)) if df is not None else 0,
            "data_end": df.index[-1].isoformat() if df is not None else None,
            "error": meta.get("error"),
            "fetched_at": fetched_at,
        }
        if df is not None:
            frames[sid] = df

    aligned = align_closes(frames)
    aligned_records: list[dict] = []
    for date_idx, row in aligned.iterrows():
        rec: dict = {"date": date_idx.isoformat()}
        for sid in frames:
            value = row.get(sid, float("nan"))
            rec[sid] = None if pd.isna(value) else round(float(value), 4)
        aligned_records.append(rec)

    return {
        "fetched_at": fetched_at,
        "sources": sources,
        "alignment": {
            "series_count": len(frames),
            "aligned_dates": int(len(aligned)),
            "full_coverage_dates": int(aligned.dropna().shape[0]),
        },
        "aligned": aligned_records,
    }