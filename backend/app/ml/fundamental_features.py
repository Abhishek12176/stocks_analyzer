"""Point-in-time (PIT) fundamental features (Task 6).

Fundamentals are slow-moving and published with a delay. Applying a *current*
snapshot (e.g. today's trailing P/E) to historical rows would leak the future,
so this module deliberately produces stepwise-constant, causal series:

- A PIT snapshot is a dict of raw fundamental values plus an `available_at`
  date — the first moment that data was knowable. Its values apply ONLY to
  rows on/after `available_at`, and a later snapshot supersedes an earlier one
  (merge_asof semantics).
- Rows before the first snapshot stay NaN (never fabricated / backfilled).
- A single bare "current" snapshot (e.g. the result of
  `fundamental_service.get_fundamentals()`) defaults to `available_at` = the
  last row of the frame, so historical rows never see it.

Keys follow the `fundamental_service.get_fundamentals()` output (pe_ratio,
eps, roe, roce, debt_to_equity, operating_margin, revenue_growth,
profit_growth, fundamental_score). Returned series can additionally be fed to
`alpha.add_alpha_features(..., pe=..., ...)` for the value factor.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.config import settings

FUNDAMENTAL_COLUMNS: dict[str, str] = {
    "pe_ratio": "pe",
    "eps": "eps",
    "roe": "roe",
    "roce": "roce",
    "debt_to_equity": "de",
    "operating_margin": "opm",
    "revenue_growth": "rev_growth",
    "profit_growth": "profit_growth",
    "fundamental_score": "score",
}

AVAILABLE_AT_KEYS = ("available_at", "as_of", "available_timestamp")


# ---------------------------------------------------------------------------
# PIT expansion
# ---------------------------------------------------------------------------

def _prepare_snapshots(
    snapshot: dict[str, Any] | None,
    snapshots: list[dict[str, Any]] | None,
    default_available_at: pd.Timestamp,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    if snapshot is not None:
        snap = dict(snapshot)
        for key in AVAILABLE_AT_KEYS:
            if snap.get(key, None) is not None:
                snap["available_at"] = pd.Timestamp(snap[key]).normalize()
                break
        else:
            snap["available_at"] = pd.Timestamp(default_available_at).normalize()
        prepared.append(snap)
    for snap in snapshots or []:
        snap = dict(snap)
        available_at = None
        for key in AVAILABLE_AT_KEYS:
            if snap.get(key, None) is not None:
                available_at = pd.Timestamp(snap[key]).normalize()
                break
        if available_at is None:
            raise ValueError("each PIT snapshot must carry an available_at/as_of date")
        snap["available_at"] = available_at
        prepared.append(snap)
    prepared.sort(key=lambda s: s["available_at"])
    return prepared


def _latest_known(
    avail_dates: np.ndarray,
    values: np.ndarray,
    target: np.ndarray,
) -> pd.Series:
    """Latest value whose available_at <= target date (merge_asof, causal)."""
    if len(avail_dates) == 0:
        return pd.Series(np.nan, index=target)
    idx = np.searchsorted(avail_dates, target, side="right") - 1
    vals = np.where(idx >= 0, values[np.maximum(idx, 0)], np.nan)
    return pd.Series(vals, index=target, dtype=float)


def pit_series(
    snapshots: list[dict[str, Any]],
    key: str,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """Stepwise constant PIT series for one fundamental key over `index`."""
    dates = np.array(
    [pd.Timestamp(s["available_at"]).normalize().to_datetime64() for s in snapshots]
)
    values = np.array(
        [
            (float(s[key]) if s.get(key, None) is not None else np.nan)
            for s in snapshots
        ],
        dtype=float,
    )
    target = pd.DatetimeIndex(index).normalize().to_numpy(dtype="datetime64[ns]")
    out = _latest_known(dates, values, target)
    out.index = index
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def add_fundamental_features(
    df: pd.DataFrame,
    snapshot: dict[str, Any] | None = None,
    snapshots: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Attach PIT fundamental columns to a copy of the OHLCV frame.

    `snapshot`: single fundamental dict (usually current values). Without an
    `available_at`/`as_of` key it only becomes visible from the last row.
    `snapshots`: optional list of PIT snapshots (each needs an
    `available_at`); later snapshots supersede earlier ones.
    """
    out = df.copy()
    defaults = out.index.max()
    prepared = _prepare_snapshots(snapshot, snapshots, defaults)

    seen_keys = {k for s in prepared for k in s} - set(AVAILABLE_AT_KEYS)
    for raw_key, col in FUNDAMENTAL_COLUMNS.items():
        if raw_key not in seen_keys:
            continue
        out[f"fund_{col}"] = pit_series(prepared, raw_key, pd.DatetimeIndex(out.index))

    out.attrs["feature_version"] = settings.ml_feature_version
    return out


def list_fundamental_features() -> list[str]:
    """All possible PIT fundamental columns (present columns depend on input)."""
    return sorted(f"fund_{col}" for col in FUNDAMENTAL_COLUMNS.values())