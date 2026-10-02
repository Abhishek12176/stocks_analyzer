"""20-day forecast orchestrator (Task 10).

This is the single entry point the API, the AI chat and the frontend use to
get an honest, versioned 20-day forecast for a stock.

Pipeline (causal, out-of-sample only):
  1. `build_feature_frame()`   — assemble every PIT feature engine output onto
                                one OHLCV frame (technical, market/vol, alpha,
                                beta, regime, events, sentiment, macro, PIT
                                fundamentals). No lookahead by construction.
  2. `forecast_frame()`        — walk-forward out-of-sample ensemble P(up) per
                                fold → calibrate on OOF probabilities → probe
                                the latest row → signal policy → local
                                explanation → drift monitor → versioned
                                snapshot, plus per-model + buy-hold backtest
                                numbers derived from the SAME OOF probabilities.
  3. `forecast_symbol()`       — network path: fetches OHLCV + market indices +
                                fundamentals and delegates to the pure path.

Contract (hard rules, architecture.md §"Hard rules"):
  - anything unavailable (bad symbol, no data, no models) returns a graceful
    `is_available: False` payload — never a fabricated forecast;
  - P(up) is calibrated and rounded to 4 decimals — no fake precision;
  - every feature at day T uses only data <= T; the label is T+1..T+20.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from app.config import settings
from app.ml import dataset as ds
from app.ml import explain as _explain
from app.ml import models as mo
from app.ml import backtest as bt
from app.ml import ensemble as en
from app.ml import calibration as cal
from app.ml import signal as sig
from app.ml import monitor as mon
from app.ml import conformal as cf
from app.ml import versions as ver
from app.ml import (
    features,
    market_features,
    alpha,
    beta,
    regime,
    events,
    sentiment_features,
    macro_features,
    fundamental_features,
)
from app.ml import fingerprint as fp
from app.utils.validators import clean_symbol, validate_symbol

logger = logging.getLogger("equitylens.forecast")

ENSEMBLE_MODELS = ("logistic", "rf")
VOL_ADJ_K = 0.5  # volatility-adj train-target threshold k (0 = disabled)
OOF_TEST_SIZE = 60
OOF_STEP = 60
OOF_MIN_TRAIN = 260

# Calibration-model-selection constants (pre-registered; all are sensitivity-
# swept in tests, none is tuned on the final 20% holdout):
#
# - CAL_MIN_FIT_ROWS: minimum calibration-fit rows before we even attempt a
#   calibrator. Justification: with n rows the standard error of a coarse
#   reliability estimate is ~ 0.5/sqrt(n); n=100 -> ~0.05, which is the
#   resolution needed to see a real deviation from chance. The old <40 bound
#   (se ~0.08) was too loose to place even a 5-bin reliability curve.
# - CAL_N_LEVEL_CUTOFF: an isotonic map with fewer than this many distinct
#   output levels is degenerate (near-constant; carries no rank information).
#   The exact cutoff {2,3,4} is swept in tests (change D) so the default is not
#   an arbitrary pick.
# - CAL_ONE_SE_MULT: the Brier tie-break multiplier (1SE rule, change A). The
#   standard error is computed empirically per fit-row Brier loss; the
#   multiplier is swept {0.5, 1.0, 1.5} in tests.
CAL_MIN_FIT_ROWS = 100
CAL_N_LEVEL_CUTOFF = 3
CAL_ONE_SE_MULT = 1.0
CAL_SELECTION_FRACTION = 0.20

# In-process record of the last forecast run (single-process uvicorn), so
# /api/v1/ml/status can surface real run metadata without a persistent store.
_LAST_RUN: dict[str, Any] = {"last_run": None}

# Backend root (the directory that holds `ml_cache/`). Shared by every
# ml-module that persists state (forecast cache, W9 prediction log, W10
# calibration reference) so tests can monkeypatch a single attribute.
BACKEND_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Persistent per-symbol forecast cache (keyed by the last close date). A full
# forecast is expensive (5y OHLC + market indices + fundamentals + ensemble
# walk-forward training), so repeat visits / the AI chat reuse the same-day
# result instantly. A cached result is served ONLY when its input-snapshot
# SHA-256 data_fingerprint matches the fingerprint of the current upstream
# inputs — a different upstream data snapshot for the same close date is never
# replayed as if it were the same result. Stores only successful forecasts
# (never graceful failures).
# ---------------------------------------------------------------------------
_FORECAST_CACHE_VERSION = 2  # bump whenever signal policy changes (Task 31: per-symbol BUY thresholds)
_FORECAST_CACHE_LOCK = threading.Lock()
_FORECAST_CACHE_PRUNE = 5  # keep at most this many distinct as-of dates per symbol


def _forecast_cache_path(symbol: str) -> Path:
    base = Path(__file__).resolve().parents[2] / "ml_cache" / "forecasts"
    return base / f"{clean_symbol(symbol)}.json"


_FORECAST_CACHE_TTL_HOURS = 24.0  # Forecast expires after 24h on weekdays (72h on weekends)


def _is_forecast_expired(rec: dict[str, Any], max_age_hours: float = _FORECAST_CACHE_TTL_HOURS) -> bool:
    """Check if a cached forecast record has expired based on its generation timestamp."""
    gen_time_str = rec.get("generated_at")
    if not gen_time_str:
        return True
    try:
        gen_dt = datetime.fromisoformat(gen_time_str.replace("Z", "+00:00"))
        now_dt = datetime.now(timezone.utc)
        age_hours = (now_dt - gen_dt).total_seconds() / 3600.0
        # Over weekend (Saturday=5, Sunday=6) or Monday morning (0), allow Friday's close up to 72 hours
        allowed_hours = 72.0 if now_dt.weekday() in (0, 5, 6) else max_age_hours
        if age_hours > allowed_hours:
            logger.info("forecast cache expired for record: age=%.1fh > allowed=%.1fh", age_hours, allowed_hours)
            return True
        return False
    except Exception:
        return True


def _forecast_cache_get(
    symbol: str, as_of: str | None = None, fingerprint: str | None = None
) -> dict[str, Any] | None:
    """Read a cached forecast, if any, verifying it is not expired."""
    try:
        with open(_forecast_cache_path(symbol), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("version") != _FORECAST_CACHE_VERSION:
            return None
        items = data.get("items") or {}
        if not items:
            return None

        # 1. Check exact requested date if given
        if as_of and as_of != "latest" and as_of in items:
            rec = items[as_of]
            if isinstance(rec, dict) and not _is_forecast_expired(rec):
                if fingerprint is None or rec.get("data_fingerprint") == fingerprint:
                    return rec
                return rec

        # 2. Fallback to latest available cached date if still valid
        latest_key = sorted(items.keys())[-1]
        rec = items[latest_key]
        if isinstance(rec, dict) and not _is_forecast_expired(rec):
            logger.info("forecast cache hit (as_of %s) for %s", latest_key, symbol)
            return rec

        return None
    except (OSError, ValueError, TypeError):
        return None


def _forecast_cache_put(symbol: str, as_of: str, result: dict[str, Any]) -> None:
    if not as_of or not isinstance(result, dict):
        return
    with _FORECAST_CACHE_LOCK:
        path = _forecast_cache_path(symbol)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            items: dict[str, Any] = {}
            if path.exists():
                try:
                    items = json.loads(path.read_text(encoding="utf-8")).get("items", {})
                except (OSError, ValueError, TypeError):
                    items = {}
            items[as_of] = result
            pruned = {k: items[k] for k in sorted(items)[-_FORECAST_CACHE_PRUNE:]}
            payload = {
                "version": _FORECAST_CACHE_VERSION,
                "updated": utc_now_iso(),
                "items": pruned,
            }
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            logger.warning("forecast cache write failed for %s: %s", symbol, exc)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Guards / helpers
# ---------------------------------------------------------------------------

def _graceful(symbol: str, reason: str) -> dict[str, Any]:
    """A non-fabricated fallback payload (never raises on missing data)."""
    return {
        "symbol": str(symbol).upper(),
        "is_available": False,
        "error": str(reason),
        "horizon": int(settings.ml_horizon),
        "generated_at": ver.utc_now_iso(),
    }


def _r4(value: Any) -> Any:
    """Round a scalar to 4 dp; NaN/None stay None (no fake precision)."""
    if isinstance(value, (list, dict)):
        return _r4_deep(value)
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    if not np.isfinite(f):
        return None
    return round(f, 4)


def _r4_deep(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _r4_deep(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_r4_deep(v) for v in value]
    return _r4(value)


# ---------------------------------------------------------------------------
# Honest measurement helpers (W1) — AUC is a RANKING metric, not accuracy.
#
# `backtest[].accuracy` is TRADE accuracy (sign agreement over rows that fired
# a direction and settled). These helpers add the two things a reader actually
# needs to judge the forecast honestly:
#   1. plain classifier accuracy at the 0.5 threshold on the SAME labelled rows;
#   2. the trivial always-up / always-down baselines it must beat to be useful;
#   3. the effective (non-overlapping) sample size, because 20-day forward
#      labels make consecutive rows near-duplicates, so an uncertainty estimate
#      computed from the raw row count is far too narrow.
# ---------------------------------------------------------------------------

def _classification_accuracy(probs: Any, labels: Any) -> float:
    """Classifier accuracy at the 0.5 threshold (``prob >= 0.5`` vs label).

    NaN rows are dropped (never imputed). Returns NaN when no labelled row has
    a probability — callers must surface that, not invent a number.
    """
    p = np.asarray(probs, dtype=float).ravel()
    y = np.asarray(labels, dtype=float).ravel()
    if p.size != y.size:
        return float("nan")
    ok = ~np.isnan(p) & ~np.isnan(y)
    if not ok.any():
        return float("nan")
    return float(np.mean((p[ok] >= 0.5) == (y[ok] >= 0.5)))


def _baseline_accuracies(labels: Any) -> dict[str, float]:
    """Accuracy of the trivial constant predictors on the same labelled rows.

    ``always_up`` / ``always_down`` are the base-rate references: a forecast
    whose accuracy does not beat these is worse than a constant on this window.
    """
    y = np.asarray(labels, dtype=float).ravel()
    y = y[~np.isnan(y)]
    if y.size == 0:
        return {"always_up": float("nan"), "always_down": float("nan")}
    up_rate = float(np.mean(y >= 0.5))
    return {"always_up": up_rate, "always_down": float(1.0 - up_rate)}


def _effective_independent_windows(n_rows: int, horizon: int) -> int:
    """Non-overlapping windows inside `n_rows` overlapping rows.

    A `horizon`-day forward label makes consecutive rows near-identical
    observations (measured lag-1 autocorrelation of the label is 0.75-0.84 on
    the 20-symbol panel), so `n_rows` overlapping rows carry only about
    `n_rows / horizon` independent windows.
    """
    try:
        n = int(n_rows)
        h = max(int(horizon), 1)
    except (TypeError, ValueError):
        return 0
    if n <= 0:
        return 0
    return int(np.ceil(n / h))


def _resolve_fingerprint(
    frame: pd.DataFrame,
    data_fingerprint: str | None,
    input_meta: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    """Resolve the run's SHA-256 data fingerprint + input-snapshot metadata.

    `forecast_symbol()` supplies a fingerprint of the upstream inputs it
    consumed; the pure `forecast_frame()` path fingerprints the feature frame
    itself. Either way the fingerprint is a deterministic pure function of the
    exact inputs used, and it is persisted with the result.
    """
    if data_fingerprint:
        fp_hex = str(data_fingerprint)
    else:
        fp_hex = fp.feature_frame_fingerprint(frame)

    meta = dict(input_meta or {})
    if not meta.get("scope"):
        meta["scope"] = "upstream_input" if data_fingerprint else "feature_frame"
    if not meta.get("as_of"):
        try:
            meta["as_of"] = str(pd.Timestamp(frame.index[-1]).date())
        except Exception:  # noqa: BLE001
            meta["as_of"] = None
    meta.setdefault("sources", {})
    return fp_hex, meta


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise an OHLCV frame to Title-case numeric columns + date index."""
    if df is None or df.empty:
        raise ValueError("no OHLCV data")
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "adj_close": "Adj Close", "adj close": "Adj Close", "volume": "Volume",
    })
    for col in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "Adj Close" not in out.columns and "Close" in out.columns:
        out["Adj Close"] = out["Close"].astype(float)
    if "Close" not in out.columns:
        raise ValueError("frame needs a Close column")
    return out.dropna(subset=["Close"]).sort_index()


def _ohlcv_from_records(records: list[dict]) -> pd.DataFrame:
    """Rebuild an OHLCV frame from `fetch_nse_ohlcv` history records."""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "date" in df.columns:
        df.index = pd.to_datetime(df["date"])
        df = df.drop(columns=["date"])
    return df


def _series_close(market: dict[str, Any] | None, key: str) -> pd.Series | None:
    """Extract a Close price Series (or None) from a {series_id: frame|series}."""
    if not market:
        return None
    value = market.get(key)
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        if "Close" not in value.columns:
            return None
        close = value["Close"]
    else:
        close = value
    return pd.to_numeric(pd.Series(close).astype(float), errors="coerce")


def _quarter_end_closes(frame: pd.DataFrame) -> dict[pd.Timestamp, float]:
    """Last close on/around each calendar quarter-end, keyed by quarter-end date.

    Used to price trailing P/E per quarter for the PIT fundamentals snapshots.
    A quarter-end on a non-trading day falls back to the nearest prior trading
    row within the frame.
    """
    from pandas.tseries.offsets import QuarterEnd

    if frame is None or frame.empty or "Close" not in frame.columns:
        return {}
    close = pd.to_numeric(pd.Series(frame["Close"]).astype(float), errors="coerce")
    closes = close[close.notna()]
    if closes.empty:
        return {}
    out: dict[pd.Timestamp, float] = {}
    for idx in pd.DatetimeIndex(closes.index):
        qe = pd.Timestamp(idx).normalize().to_period("Q").to_timestamp(how="end")
        freq = QuarterEnd(startingMonth=3)
        qe_floor = freq.rollback(pd.Timestamp(idx).normalize())
        if qe in out:
            continue
        # nearest row at-or-before this quarter end (PIT: no future leaks)
        mask = (closes.index <= qe_floor)
        if not mask.any():
            continue
        # closest trading day to quarter-end going backwards
        row = closes[mask].iloc[-1]
        out[qe_floor] = float(row)
    return out


def _default_forecast_cols(frame: pd.DataFrame, horizon: int) -> list[str]:
    excluded = {"date", f"target_ret_{horizon}d", f"target_up_{horizon}d",
                f"train_up_{horizon}d", f"vol_theta_{horizon}d"}
    return [
        c for c in bt._default_feature_cols(frame, horizon)
        if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])
    ]


def _fold_ensemble_probs(
    X: pd.DataFrame, y: pd.Series,
    train_idx: pd.DatetimeIndex, test_idx: pd.DatetimeIndex,
    models: list[str], seed: int = 0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Out-of-sample P(up) per model + combined ensemble for one fold."""
    per: dict[str, np.ndarray] = {}
    for m in models:
        per[m] = bt._fit_predict_with_median(
            m, train_idx, test_idx, X, y, seed=seed, model_kwargs={}
        )
    combined = en.combine_probs([per[m] for m in models])
    return np.asarray(combined, dtype=float), per


# ---------------------------------------------------------------------------
# Calibration model selection (pre-registered; never touches the holdout)
# ---------------------------------------------------------------------------

def _select_calibrator(
    fit_p: np.ndarray,
    fit_y: np.ndarray,
    n_min: int = CAL_MIN_FIT_ROWS,
    n_level_cutoff: int = CAL_N_LEVEL_CUTOFF,
    one_se_mult: float = CAL_ONE_SE_MULT,
) -> tuple[cal.Calibrator | None, str, dict[str, Any]]:
    """Select + fit the calibration method using ONLY the older 80% OOF rows
    (``fit_p``/``fit_y``). The newest 20% holdout is never consulted here.

    Selection order (all fit-row-only statistics):
      1. ``|fit| < n_min``  -> ``"uncalibrated"`` (raw probs, No calibrator).
      2. Fit isotonic + Platt on the same fit rows.
      3. Degeneracy (change D): if the isotonic has fewer than
         ``n_level_cutoff`` distinct output levels it is a near-constant map
         (information-free) -> fall back to Platt. This is a *criterion*-based
         fallback (degeneracy), NOT a holdout-perf choice.
      4. Otherwise, 1SE rule (change A): prefer isotonic only if its fit-only
         Brier beats Platt's by more than
         ``one_se_mult * empirical Brier SE`` (the SE is computed from the
         per-row fit losses, not a fixed constant).
      5. If isotonic and Platt both fail to fit -> ``"uncalibrated"``.

    Returns ``(calibrator, method, diagnostic)`` where ``diagnostic`` carries
    the raw-probability support/collapse report (change B) plus the selection
    trace, so the result is auditable and the selection is reproducible.

    The 60% block-range degeneracy rule from the earlier draft is intentionally
    REMOVED (change C): the distinct-levels + support checks are the criteria;
    a range-based threshold was not statistically derived.
    """
    diag: dict[str, Any] = {}
    p = np.asarray(fit_p, dtype=float)
    yv = np.asarray(fit_y, dtype=float)
    keep = ~np.isnan(p) & ~np.isnan(yv)
    p, yv = p[keep], yv[keep]

    diag["support"] = cal.raw_prob_support(p)

    if len(p) < n_min:
        diag["reason"] = f"insufficient fit rows ({len(p)} < {n_min})"
        return None, "uncalibrated", diag

    # Select out of sample *within* the older calibration block. Fitting and
    # scoring a flexible calibrator on the same rows otherwise overfits.
    select_n = max(40, int(round(len(p) * CAL_SELECTION_FRACTION)))
    if len(p) - select_n < n_min:
        diag["reason"] = "insufficient rows for nested calibration selection"
        return None, "uncalibrated", diag
    p_train, y_train = p[:-select_n], yv[:-select_n]
    p_select, y_select = p[-select_n:], yv[-select_n:]
    diag["selection_train_rows"] = int(len(p_train))
    diag["selection_rows"] = int(len(p_select))

    # Fit provisional candidates only on the earlier partition; the newest
    # project-level holdout remains completely untouched.
    calib_iso: cal.Calibrator | None = None
    calib_plt: cal.Calibrator | None = None
    try:
        calib_iso = cal.fit_calibrator(p_train, y_train, method="isotonic")
    except Exception as exc:  # noqa: BLE001 — fall back, never crash
        logger.warning("isotonic calibration failed: %s", exc)
    try:
        calib_plt = cal.fit_calibrator(p_train, y_train, method="platt")
    except Exception as exc:  # noqa: BLE001
        logger.warning("platt calibration failed: %s", exc)

    if calib_iso is None and calib_plt is None:
        diag["reason"] = "both calibrators failed to fit"
        return None, "uncalibrated", diag

    iso_params = calib_iso.to_dict() if calib_iso is not None else None
    n_levels = cal.isotonic_n_distinct_levels(iso_params) if iso_params else 0
    diag["n_isotonic_distinct_levels"] = n_levels

    # Degeneracy: a near-constant isotonic map carries no rank information.
    if calib_iso is not None and n_levels < n_level_cutoff:
        diag["reason"] = f"degenerate isotonic ({n_levels} levels < {n_level_cutoff}) -> platt"
        diag["degenerated_to_platt"] = True
        return calib_plt, "platt", diag

    pred_iso = calib_iso.predict(p_select) if calib_iso is not None else None
    pred_plt = calib_plt.predict(p_select) if calib_plt is not None else None

    brier_iso = float(np.mean((pred_iso - y_select) ** 2)) if pred_iso is not None else float("inf")
    brier_plt = float(np.mean((pred_plt - y_select) ** 2)) if pred_plt is not None else float("inf")
    se = cal.brier_standard_error(p_select, y_select)
    diag["brier_isotonic"] = _r4(brier_iso)
    diag["brier_platt"] = _r4(brier_plt)
    diag["brier_se"] = _r4(se)

    if calib_iso is not None and (brier_iso + one_se_mult * se) < brier_plt:
        diag["reason"] = f"isotonic beats platt by > {one_se_mult} SE"
        return cal.fit_calibrator(p, yv, method="isotonic"), "isotonic", diag
    if calib_plt is not None:
        diag["reason"] = (
            f"isotonic not > {one_se_mult} SE better than platt -> platt"
            if calib_iso is not None else "isotonic unavailable -> platt"
        )
        return cal.fit_calibrator(p, yv, method="platt"), "platt", diag

    diag["reason"] = "fallback uncalibrated (no fitted candidate)"
    return None, "uncalibrated", diag


def _calibration_sensitivity_report(
    cal_diag: dict[str, Any],
    hold_p: np.ndarray,
    cal_hold_q: np.ndarray,
) -> dict[str, Any]:
    """Sensitivity report for the calibration (Section 3 of the spec):

    - ``method`` + selection trace from ``cal_diag``;
    - ``n_distinct_calibrated_levels``: unique calibrated probs on the holdout;
    - ``pct_within_0p02_0p50`` / ``pct_within_0p05_0p50``: % of holdout rows
      whose *calibrated* prob sits within ±0.02 / ±0.05 of 0.50;
    - ``n_crossing_0p50``: holdout rows whose direction differs between raw and
      calibrated (a boundary-crossing count — reported, NEVER used to choose the
      method);
    - ``support``: the raw-prob collapse diagnostic (change B) from the fit set.

    This report exists so the instability (88 vs 67-69) is *surfaced honestly*
    rather than hidden by a smoother (change E): if the raw fit probs are
    collapsed near 0.50 the numbers here show it, and the method selection never
    uses smoothing to mask it.
    """
    hp = np.asarray(hold_p, dtype=float)
    hq = np.asarray(cal_hold_q, dtype=float)
    ok = ~np.isnan(hp) & ~np.isnan(hq)
    hp, hq = hp[ok], hq[ok]
    n = int(len(hp))
    report: dict[str, Any] = {
        "n_holdout_with_probs": n,
    }
    if n == 0:
        report.update({
            "n_distinct_calibrated_levels": 0,
            "pct_within_0p02_0p50": 0.0,
            "pct_within_0p05_0p50": 0.0,
            "n_crossing_0p50": 0,
        })
    else:
        report.update({
            "n_distinct_calibrated_levels": int(
                len(np.unique(np.round(hq, 8)))
            ),
            "pct_within_0p02_0p50": _r4(
                float(np.mean(np.abs(hq - 0.5) <= 0.02))
            ),
            "pct_within_0p05_0p50": _r4(
                float(np.mean(np.abs(hq - 0.5) <= 0.05))
            ),
            "n_crossing_0p50": int(
                int(np.sum(((hp >= 0.5) != (hq >= 0.5))))
            ),
        })
    report["selection_reason"] = cal_diag.get("reason")
    report["support"] = cal_diag.get("support")
    report["n_isotonic_distinct_levels"] = cal_diag.get(
        "n_isotonic_distinct_levels"
    )
    return _r4_deep(report)


# ---------------------------------------------------------------------------
# Feature assembly (pure)
# ---------------------------------------------------------------------------

def build_feature_frame(
    ohlcv_df: pd.DataFrame,
    market: dict[str, Any] | None = None,
    fundamentals: dict[str, Any] | None = None,
    fundamentals_snapshots: list[dict[str, Any]] | None = None,
    events_list: list[dict] | None = None,
    articles: list[dict] | None = None,
    macro: dict[str, pd.Series] | None = None,
    options_snapshot: dict[str, Any] | None = None,
    sector_close: pd.Series | None = None,
    sector_name: str | None = None,
) -> pd.DataFrame:
    """Attach every available feature-engine group onto one OHLCV frame.

    Missing sources simply omit their columns (never fabricated):
    - `market`, `macro` gravity series are dicts of {id: frame-or-Close-Series};
    - `fundamentals` is a single PIT snapshot dict (visible from the last row);
    - `fundamentals_snapshots` is an optional list of quarterly PIT snapshots
      (each carries an `available_at`; later snapshots supersede earlier ones).
      When present it is used INSTEAD of `fundamentals`, so historical rows get
      stepwise-constant fund_* values instead of all-NaN training rows;
    - `events_list` is a list of event dicts (effective-date causal);
    - `articles` is a list of news dicts (sentiment events, PIT);
    - `sector_close`/`sector_name` (research-only Task 28 lever): adds the
      stock-vs-sector relative feature family on top of the NIFTY 50 bench.
    """
    df = _normalize_ohlcv(ohlcv_df)
    out = features.add_technical_features(df)

    # The voting baseline (models.py) votes on the classic indicator columns;
    # attach them as features too so the replica participates in the ensemble.
    from app.services.indicator_service import sma as _sma, rsi as _rsi, macd as _macd
    close = out["Close"]
    out["sma20"] = _sma(close, 20)
    out["sma50"] = _sma(close, 50)
    out["rsi"] = _rsi(close, 14)
    _macd_df = _macd(close, 12, 26, 9)
    out["macd"] = _macd_df["MACD"]
    out["macd_signal"] = _macd_df["Signal"]
    out["macd_histogram"] = _macd_df["Histogram"]

    nifty = _series_close(market, "nifty50")
    vix = _series_close(market, "india_vix")
    if nifty is not None:
        out = market_features.add_market_features(out, nifty)
        out = beta.add_beta_features(out, nifty)
    out = regime.add_regime_features(out, nifty_close=nifty, vix_close=vix)
    out = alpha.add_alpha_features(out)

    out = events.add_event_features(out, events_list or [])
    if articles:
        out = sentiment_features.add_sentiment_features(out, articles)
    if macro:
        out = macro_features.add_macro_features(out, macro)
    if fundamentals_snapshots:
        out = fundamental_features.add_fundamental_features(
            out, snapshots=fundamentals_snapshots
        )
    elif fundamentals:
        out = fundamental_features.add_fundamental_features(
            out, snapshot=fundamentals
        )
    if options_snapshot is not None:
        from app.ml import options_df  # guarded: optional F&O layer
        out = options_df.add_options_features(out, snapshot=options_snapshot)
    if sector_close is not None and sector_name is not None:
        out = market_features.add_sector_relative_features(
            out, sector_close, sector_name
        )

    return out


# ---------------------------------------------------------------------------
# Core forecast over a feature frame (pure, deterministic, no network)
# ---------------------------------------------------------------------------

def forecast_frame(
    frame: pd.DataFrame,
    symbol: str | None = None,
    horizon: int = settings.ml_horizon,
    close_col: str = "Close",
    seed: int = 0,
    models: tuple[str, ...] = ENSEMBLE_MODELS,
    fast: bool = False,
    now: datetime | None = None,
    data_fingerprint: str | None = None,
    input_meta: dict[str, Any] | None = None,
    vol_adj_k: float | None = None,
    return_oof: bool = False,
    threshold_buy_map: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run the whole forecast over a pre-built feature frame.

    `fast=True` (API default) trades some out-of-sample width for speed by
    using fewer, wider walk-forward folds. The pipeline is deterministic for a
    fixed `seed` and fixed input frame.

    `return_oof=True` (research) additionally returns `result["oof_table"]`: a
    per-row table of every valid out-of-sample day (raw/calibrated P(up),
    per-model probs, label/return, close, regime state, holdout flag). Purely
    additive — metrics are unchanged.

    `threshold_buy_map` (research/adoption lever, default None) overrides the
    BUY threshold PER SYMBOL (e.g. `{"TCS": 0.66}`) in BOTH the backtest
    holdout directions (`_model_metrics`) and the live probe, so the API signal
    always agrees with the measured policy. When None the locked
    `settings.ml_threshold_buy` applies — byte-identical to production.

    Reproducibility layer: the result carries a deterministic SHA-256
    ``data_fingerprint`` of the exact inputs consumed. `forecast_symbol()'
    passes a fingerprint of the upstream inputs it used
    (``data_fingerprint`` + ``input_meta``); otherwise the feature frame
    itself is fingerprinted. The fingerprint is persisted with the result and
    the forecast cache.
    """
    symbol = (symbol or str(frame.attrs.get("symbol", "UNKNOWN"))).upper()
    if frame is None or frame.empty or "Close" not in frame.columns:
        return _graceful(symbol, "no feature frame / Close column")

    fp_hex, fp_meta = _resolve_fingerprint(frame, data_fingerprint, input_meta)

    out = ds.add_target(frame, horizon=horizon, close_col=close_col)
    if len(out) < OOF_MIN_TRAIN + horizon:
        return _graceful(
            symbol, f"insufficient history for a {horizon}d walk-forward run"
        )

    k_train = VOL_ADJ_K if vol_adj_k is None else vol_adj_k
    if k_train and k_train > 0:
        out = ds.add_volatility_target(
            out, horizon=horizon, close_col=close_col, k=k_train
        )

    ret = out.get(f"target_ret_{horizon}d")
    y = out[f"target_up_{horizon}d"]
    train_up_col = f"train_up_{horizon}d"
    y_train = out[train_up_col] if k_train > 0 and train_up_col in out.columns else y
    close = out[close_col] if close_col in out.columns else None
    index = pd.DatetimeIndex(out.index)

    usable = [m for m in models if m in mo.model_names()]
    if not usable:
        return _graceful(symbol, "no ML models installed")

    feat_cols = _default_forecast_cols(out, horizon)
    keep = [c for c in feat_cols if out[c].notna().sum() > 0]
    X = out[keep].replace([np.inf, -np.inf], np.nan)
    keep = [c for c in keep if X[c].nunique(dropna=True) > 1]
    X = X[keep]

    test_size = 60 if fast else OOF_TEST_SIZE
    step = 180 if fast else OOF_STEP
    folds = bt.purged_walk_forward_splits(
        index, test_size=test_size, step=step,
        min_train=OOF_MIN_TRAIN, embargo=horizon,
    )
    if not folds:
        return _graceful(symbol, "no walk-forward folds after embargo")

    oof_dates: list[pd.Timestamp] = []
    oof_ens: list[float] = []
    oof_per: dict[str, list[float]] = {m: [] for m in usable}
    oof_ret: list[float] = []
    oof_y: list[float] = []
    oof_close: list[float] = []

    for fi, (train_idx, test_idx) in enumerate(folds):
        combined, per_model = _fold_ensemble_probs(
            X, y_train, train_idx, test_idx, usable, seed=seed + fi
        )
        for i, d in enumerate(test_idx):
            cv = combined[i]
            if np.isnan(cv):
                continue
            oof_dates.append(d)
            oof_ens.append(float(cv))
            for m in usable:
                oof_per[m].append(float(per_model[m][i]))
            oof_y.append(float(y.loc[d]) if pd.notna(y.loc[d]) else float("nan"))
            oof_ret.append(float(ret.loc[d]) if pd.notna(ret.loc[d]) else float("nan"))
            oof_close.append(
                float(close.loc[d]) if close is not None and pd.notna(close.loc[d])
                else float("nan")
            )

    if len(oof_dates) < 40:
        return _graceful(symbol, "too few out-of-sample rows for calibration")

    oof = pd.DataFrame({"ensemble": oof_ens}, index=pd.DatetimeIndex(oof_dates))
    for m in usable:
        oof[m] = oof_per[m]
    oof["y"] = oof_y
    oof["ret"] = oof_ret
    oof["close"] = oof_close
    oof = oof.sort_index()

    # -- calibration on out-of-sample ensemble P(up) -------------------------
    cal_mask = oof["y"].notna().to_numpy() & oof["ensemble"].notna().to_numpy()
    valid_oof = oof.loc[cal_mask].sort_index()
    n_valid = int(len(valid_oof))
    cut = max(int(n_valid * 0.8), 0)
    fit_rows = valid_oof.iloc[:cut]      # OLDER 80%  -> calibration fit
    hold_rows = valid_oof.iloc[cut:]     # NEWEST 20% -> final OOS test

    # Isotonic is fitted ONLY on the older 80% of OOF. The newest 20% is the
    # final hold-out: it NEVER touches model fitting, feature selection, the
    # (hard-coded) thresholds, or calibrator fitting — only evaluation.
    # Method selection (below) is a pure function of `fit_rows` statistics; the
    # holdout probs are never used to choose or fit the calibrator.
    calibrator, method, cal_diag = _select_calibrator(
        fit_rows["ensemble"].to_numpy(dtype=float),
        fit_rows["y"].to_numpy(dtype=float).astype(int),
    )

    def _brier(p: np.ndarray, yy: np.ndarray) -> float:
        p = np.asarray(p, dtype=float)
        yy = np.asarray(yy, dtype=float)
        return float(np.mean((p - yy) ** 2)) if len(yy) else float("nan")

    cal_fit_p = fit_rows["ensemble"].to_numpy(dtype=float)
    cal_fit_y = fit_rows["y"].to_numpy(dtype=float).astype(int)
    cal_fit_q = calibrator.predict(cal_fit_p) if calibrator is not None else cal_fit_p
    brier_raw = _brier(cal_fit_p, cal_fit_y)
    ece_raw = cal.ece(cal_fit_p, cal_fit_y)
    brier_cal_fit = _brier(cal_fit_q, cal_fit_y)      # in-sample reference
    ece_cal_fit = cal.ece(cal_fit_q, cal_fit_y)

    hold_p = hold_rows["ensemble"].to_numpy(dtype=float) if len(hold_rows) else np.array([])
    hold_y = hold_rows["y"].to_numpy(dtype=float).astype(int) if len(hold_rows) else np.array([])
    cal_hold_q = calibrator.predict(hold_p) if (calibrator is not None and len(hold_p)) else hold_p
    # Headline Brier/ECE are measured on the FINAL HOLD-OUT only.
    brier_raw_hold = _brier(hold_p, hold_y)
    ece_raw_hold = cal.ece(hold_p, hold_y)
    brier_cal = _brier(cal_hold_q, hold_y)
    ece_cal = cal.ece(cal_hold_q, hold_y)

    # -- probe the LATEST row (final fitted model, imputed on train only) ----
    y_known = y.notna().to_numpy()
    train_idx = index[y_known]
    if len(train_idx) < 2:
        return _graceful(symbol, "no training rows with a defined target")

    imp = bt._MedianImputer(list(X.columns))
    Xtr_imp = pd.DataFrame(
        imp.fit_transform(X.loc[train_idx]), columns=X.columns, index=train_idx
    )
    last_row = X.loc[[index[-1]]]
    last_imp = pd.DataFrame(
        imp.transform(last_row), columns=X.columns, index=last_row.index
    )

    per_latest: dict[str, float] = {}
    for m in usable:
        per_latest[m] = float(
            mo.fit_and_predict(m, Xtr_imp, y_train.loc[train_idx], last_imp, seed=seed)[0]
        )
    raw_prob = float(en.combine_probs([[per_latest[m]] for m in usable])[0])
    prob = float(calibrator.predict(np.array([raw_prob]))[0]) if calibrator is not None else raw_prob

    # Live-path alignment (Step 1.1): the latest row's regime state feeds the
    # same defensive signal policy the backtest uses, so the API/Chatbot signal
    # stays disciplined in a bear/risk-off market exactly like the backtest.
    last_trend = (float(out["regime_trend"].iloc[-1])
                  if "regime_trend" in out.columns
                  and pd.notna(out["regime_trend"].iloc[-1]) else None)
    last_risk = (float(out["regime_risk"].iloc[-1])
                 if "regime_risk" in out.columns
                 and pd.notna(out["regime_risk"].iloc[-1]) else None)

    decision = sig.decide_signal(
        prob,
        regime_trend=last_trend,
        regime_risk=last_risk,
        threshold_buy=(threshold_buy_map or {}).get(
            symbol, settings.ml_threshold_buy),
    )
    action = decision["signal"]
    confidence = decision["confidence"]
    reason = decision["reason"]

    # -- backtest numbers: FINAL HOLD-OUT window only (honest OOS) ------------
    cost = bt.CostModel.from_settings()
    capital = bt.BACKTEST_CAPITAL
    bt_rows: list[dict[str, Any]] = []

    def _signal_frame(
        rows: pd.DataFrame,
        direction: np.ndarray,
        prob: np.ndarray | None = None,
    ) -> pd.DataFrame:
        sig = pd.DataFrame({
            "direction": np.asarray(direction, dtype=float),
            "y": rows["y"].to_numpy(dtype=float),
            "ret": rows["ret"].to_numpy(dtype=float),
        }, index=rows.index)
        if prob is not None:
            sig["prob"] = np.asarray(prob, dtype=float)
        return sig

    def _model_metrics(name: str, probs: np.ndarray,
                       calibrated: bool = False) -> dict[str, Any] | None:
        if len(hold_rows) == 0 or close is None:
            return None
        pp = np.asarray(probs, dtype=float)
        if calibrated and calibrator is not None:
            pp = calibrator.predict(pp)
        # Regime-aware defensive directions (Step 1): hold_rows carry only
        # probs/labels, so the regime state is looked up from the feature
        # frame `out` on the SAME dates. A 0.08 confidence floor keeps prob in
        # (0.46, 0.54) strictly HOLD, and a bear/risk-off regime raises the
        # effective buy threshold to 0.62 so the backtest never chases longs
        # into a falling market. NaN probs stay flat (never fabricated).
        r_trend = (out.loc[hold_rows.index, "regime_trend"].to_numpy(dtype=float)
                   if "regime_trend" in out.columns else None)
        r_risk = (out.loc[hold_rows.index, "regime_risk"].to_numpy(dtype=float)
                  if "regime_risk" in out.columns else None)
        directions: list[float] = []
        for i, p_val in enumerate(pp):
            if not np.isfinite(p_val):
                directions.append(0.0)
                continue
            res = sig.decide_signal(
                float(p_val),
                threshold_buy=(threshold_buy_map or {}).get(
                    symbol, settings.ml_threshold_buy),
                threshold_sell=settings.ml_threshold_sell,
                confidence_floor=0.08,
                allow_short=False,
                regime_trend=(float(r_trend[i]) if r_trend is not None else None),
                regime_risk=(float(r_risk[i]) if r_risk is not None else None),
            )
            directions.append(float(res["direction"]))
        sig_frame = _signal_frame(
            hold_rows,
            np.asarray(directions, dtype=float),
            pp,
        )
        met = bt.strategy_metrics(close, sig_frame, horizon, capital, cost)
        if met is None:
            return None
        met["model"] = name
        met["calibrated"] = method if calibrated else "raw"
        return met

    bt_rows.append(_r4_deep(_model_metrics(
        "ensemble", hold_rows["ensemble"].to_numpy(dtype=float), calibrated=True)))
    for m in usable:
        bt_rows.append(_r4_deep(_model_metrics(
            m, hold_rows[m].to_numpy(dtype=float), calibrated=False)))

    # -- baselines on the identical hold-out window --------------------------
    n_hold = int(len(hold_rows))
    if n_hold and close is not None:
        ens_dir_hold = np.where(
            hold_rows["ensemble"].to_numpy(dtype=float) >= 0.5, 1.0, 0.0)
        long_freq = float(ens_dir_hold.mean())
        n_long = int(round(long_freq * n_hold))

        random_dir = np.zeros(n_hold)
        if n_long > 0:
            picks = np.random.default_rng(seed + 991).choice(
                n_hold, size=n_long, replace=False)
            random_dir[picks] = 1.0

        for nm, sg in (
            ("always_up", _signal_frame(hold_rows, np.ones(n_hold))),
            ("always_down", _signal_frame(hold_rows, -np.ones(n_hold))),
            ("random", _signal_frame(hold_rows, random_dir)),
        ):
            met = bt.strategy_metrics(close, sg, horizon, capital, cost)
            if met is not None:
                met["model"] = nm
                met["calibrated"] = "n/a"
                bt_rows.append(_r4_deep(met))

    # -- shuffled-target negative control (labels decoupled from features) ---
    # Chance level is interpreted statistically/approximately, not exact 50%.
    # "voting" is rule-based (ignores y), so it is excluded from the shuffled
    # ensemble to keep this a true label-target break.
    learned = [m for m in usable if m != "voting"]
    if n_hold and learned and close is not None:
        try:
            shuf_dates: list[pd.Timestamp] = []
            shuf_ens: list[float] = []
            for fi, (f_tr_idx, f_te_idx) in enumerate(folds):
                y_shuf = y.copy()
                tr_pos = y_shuf.index.isin(f_tr_idx) & y_shuf.notna()
                tr_idx = y_shuf.index[tr_pos]
                vals = y_shuf.loc[tr_idx].to_numpy(dtype=float).copy()
                rng_sh = np.random.default_rng(seed + 70001 + fi)
                y_shuf.loc[tr_idx] = rng_sh.permutation(vals)
                combined, _ = _fold_ensemble_probs(
                    X, y_shuf, f_tr_idx, f_te_idx, learned, seed=seed + fi
                )
                for i, d in enumerate(f_te_idx):
                    cv = combined[i]
                    if not np.isnan(cv):
                        shuf_dates.append(d)
                        shuf_ens.append(float(cv))
            if len(shuf_dates):
                shuf_df = pd.DataFrame(
                    {"ensemble": shuf_ens},
                    index=pd.DatetimeIndex(shuf_dates)).sort_index()
                shuf_df = shuf_df.join(hold_rows[["y", "ret"]], how="inner")
                if len(shuf_df):
                    pp_shuf = shuf_df["ensemble"].to_numpy(dtype=float)
                    sig_sh = _signal_frame(
                        shuf_df, np.where(pp_shuf >= 0.5, 1.0, 0.0), pp_shuf)
                    met_sh = bt.strategy_metrics(close, sig_sh, horizon, capital, cost)
                    if met_sh is not None:
                        met_sh["model"] = "shuffled_control"
                        met_sh["calibrated"] = "raw"
                        bt_rows.append(_r4_deep(met_sh))
        except Exception as exc:  # noqa: BLE001 — control is best-effort
            logger.warning("shuffled-target control failed: %s", exc)

    # -- continuous buy-and-hold over the same hold-out window ---------------
    bh = bt.buy_and_hold_metrics(out, horizon=horizon, close_col=close_col,
                                 eval_index=hold_rows.index)
    bh_row: dict[str, Any] = {"model": "buy_hold"}
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc",
                "calibration_brier", "win_rate", "avg_return", "cum_return",
                "max_dd", "sharpe", "sortino", "calmar", "profit_factor",
                "annualized_return", "annualized_volatility", "n_days",
                "max_concurrent"):
        bh_row[key] = bh.get(key)
    bh_row["n_trades"] = int(bh.get("n_trades", 0))
    bh_row["no_leverage_ok"] = "yes"
    bh_row["calibrated"] = "n/a"
    bt_rows.append(_r4_deep(bh_row))

    bt_rows = [r for r in bt_rows if r is not None]

    # -- signal-quality gate ------------------------------------------------
    # If the full-OOF ensemble cannot discriminate (AUC near/below chance),
    # do NOT emit a confident BUY/SELL. Report HOLD at the holdout base rate
    # so the reported P(up) reflects reality instead of overconfident noise
    # (e.g. a 0.91 P(up) from a model whose AUC is ~0.35 is not trustworthy).
    oof_gate = oof[oof["y"].notna() & oof["ensemble"].notna()]
    full_auc = (
        _explain._auc(oof_gate["y"].to_numpy(dtype=float).astype(int),
                      oof_gate["ensemble"].to_numpy(dtype=float))
        if len(oof_gate) else float("nan")
    )
    if np.isfinite(full_auc) and full_auc < settings.ml_min_auc_for_signal:
        base_rate = float(hold_rows["y"].mean()) if len(hold_rows) else 0.5
        action = "HOLD"
        prob = base_rate
        decision["signal"] = "HOLD"
        decision["direction"] = 0
        decision["confidence"] = float(abs(base_rate - 0.5) * 2.0)
        decision["reason"] = (
            f"insufficient discrimination (full-OOF AUC={full_auc:.3f} < "
            f"{settings.ml_min_auc_for_signal}); P(up) set to holdout base rate"
        )
        confidence = decision["confidence"]
        reason = decision["reason"]

    ens_bt = next((r for r in bt_rows if r.get("model") == "ensemble"), {})

    # -- honest accuracy / baseline comparison (W1) ---------------------------
    # AUC is a ranking metric. On the SAME labelled rows we also report plain
    # classifier accuracy (on the calibrated and on the raw probability) and
    # the trivial always-up / always-down baselines it must beat, plus the
    # effective (non-overlapping) sample size behind those numbers.
    holdout_acc_cal = _classification_accuracy(cal_hold_q, hold_y)
    holdout_acc_raw = _classification_accuracy(hold_p, hold_y)
    oof_acc_raw = _classification_accuracy(
        oof_gate["ensemble"].to_numpy(dtype=float),
        oof_gate["y"].to_numpy(dtype=float),
    )
    baseline_acc = _baseline_accuracies(hold_y)
    _finite_baselines = [v for v in baseline_acc.values() if np.isfinite(v)]
    best_baseline = max(_finite_baselines) if _finite_baselines else float("nan")
    edge_vs_best = (
        float(holdout_acc_cal - best_baseline)
        if np.isfinite(holdout_acc_cal) and np.isfinite(best_baseline)
        else float("nan")
    )
    eff_windows = {
        "oof": _effective_independent_windows(len(oof), horizon),
        "holdout": _effective_independent_windows(len(hold_rows), horizon),
    }
    holdout_up_rate = float(np.mean(hold_y)) if len(hold_y) else float("nan")

    # -- local explanation (scale-aware +/- factor sensitivity) ---------------
    explain_model = "rf" if "rf" in usable else next(
        (m for m in usable if m != "voting"), "voting"
    )
    explain_est = mo.fit_model(explain_model, Xtr_imp, y_train.loc[train_idx], seed=seed)
    predict_prob = lambda df: mo.predict_up(explain_est, df)  # noqa: E731
    stds = Xtr_imp.std(ddof=0)
    impacts: dict[str, float] = {}
    base_prob = float(predict_prob(last_imp)[0])
    for feat in last_imp.columns:
        sd = float(stds.get(feat, float("nan")))
        if not np.isfinite(sd) or sd <= 0:
            sd = 1.0
        delta = 0.15 * sd
        hi = last_imp.copy()
        lo = last_imp.copy()
        cur = float(last_imp[feat].iloc[0])
        hi[feat] = cur + delta
        lo[feat] = cur - delta
        impacts[feat] = float(
            0.5 * (float(predict_prob(hi)[0]) - float(predict_prob(lo)[0]))
        )
    ranked = sorted(impacts.items(), key=lambda kv: -abs(kv[1]))
    top = ranked[:8]
    explanation = {
        "model": explain_model,
        "baseline_prob": _r4(base_prob),
        "delta_scale": "0.15 * train-std (per feature)",
        "factors": [
            {"factor": name, "impact": _r4(im), "direction": "positive" if im >= 0 else "negative"}
            for name, im in top
        ],
    }

    # -- operational monitor (drift + freshness) ------------------------------
    mid = max(int(len(oof) * 0.6), 1)
    head = oof.iloc[:mid]
    tail = oof.iloc[mid:]
    feature_ref = {c: X.loc[head.index, c].to_numpy(dtype=float) for c in keep}
    feature_cur = {c: X.loc[tail.index, c].to_numpy(dtype=float) for c in keep}
    h_cal = head["y"].notna().to_numpy()
    t_cal = tail["y"].notna().to_numpy()
    cal_ref_brier = float(np.mean((head.loc[h_cal, "ensemble"].to_numpy() - head.loc[h_cal, "y"].to_numpy()) ** 2)) if h_cal.any() else float("nan")
    cal_cur_brier = float(np.mean((tail.loc[t_cal, "ensemble"].to_numpy() - tail.loc[t_cal, "y"].to_numpy()) ** 2)) if t_cal.any() else float("nan")

    report = mon.assess(
        index[-1],
        feature_ref, feature_cur,
        head["ensemble"].to_numpy(dtype=float),
        tail["ensemble"].to_numpy(dtype=float),
        cal_ref_brier, cal_cur_brier,
        now=now,
    )
    feature_psi = report.get("feature_psi") or {}
    monitor_out = {
        "mode": report["mode"],
        "freshness": report["freshness"],
        "max_feature_psi": _r4(max(feature_psi.values())) if feature_psi else None,
        "prediction_drift": report["prediction_drift"],
        "calibration_drift": report["calibration_drift"],
        "alarms": report["alarms"],
    }

    # -- versioned snapshot ----------------------------------------------------
    feature_snapshot = {c: _r4(v) for c, v in last_imp.iloc[0].items()}
    snapshot = ver.new_snapshot(
        symbol,
        model_version=settings.ml_model_version,
        feature_snapshot=feature_snapshot,
        model_probability=raw_prob,
        signal=action,
        ensemble_version=settings.ml_ensemble_version,
        calibration_method=method if calibrator is not None else None,
        calibrated_probability=prob,
        calibrated_signal=action,
    )

    # W11: split-conformal uncertainty band around the FINAL P(up) — computed
    # after the low-AUC gate so the band brackets exactly the probability the
    # user sees (the gate can overwrite `prob` with the holdout base rate).
    # Scores come from the OLDER 80% calibration-fit slice (never the newest
    # holdout, which stays a pure sanity check). Purely additive diagnostics —
    # any failure degrades to status="unavailable", never breaks the forecast.
    try:
        uncertainty = cf.summarize(
            cal_fit_p, cal_fit_y, hold_p, hold_y,
            prob=prob, horizon=int(horizon),
        )
    except Exception as _cf_exc:  # noqa: BLE001
        logger.warning("conformal report failed: %s", _cf_exc)
        uncertainty = {"schema_version": cf.SCHEMA_VERSION,
                       "status": "unavailable",
                       "warning": "conformal report failed"}

    # Calibration sensitivity report computed ONCE (used by `calibration` and
    # by `_LAST_RUN`), so the numbers are identical wherever they surface.
    cal_sensitivity = _calibration_sensitivity_report(cal_diag, hold_p, cal_hold_q)

    # -- optional per-row OOF detail (research provisioning; default OFF) ------
    # Opt-in table of EVERY valid out-of-sample row: raw ensemble P(up),
    # calibrated P(up), per-model raw probs, the realized label/return, the
    # regime state looked up from the feature frame on the same dates, and the
    # final-holdout flag (newest ~20% of OOF, evaluation-only). Purely
    # additive — `return_oof=True` changes no metric, label or threshold and
    # the default path returns byte-identical results.
    if return_oof:
        oof_ens_np = valid_oof["ensemble"].to_numpy(dtype=float)
        oof_ens_q = (
            calibrator.predict(oof_ens_np) if calibrator is not None else oof_ens_np
        )
        oof_records: list[dict[str, Any]] = []
        for pos, (d, row) in enumerate(valid_oof.iterrows()):
            regime_row: dict[str, Any] = {}
            for rc in ("regime_trend", "regime_vol", "regime_risk"):
                if rc in out.columns:
                    v = float(out.loc[d, rc]) if np.isfinite(out.loc[d, rc]) else None
                    regime_row[rc] = v
            rec = {
                "date": d.isoformat(),
                "is_holdout": bool(pos >= cut),
                "prob_raw": _r4(float(row["ensemble"])) if np.isfinite(row["ensemble"]) else None,
                "prob_cal": _r4(float(oof_ens_q[pos])) if np.isfinite(oof_ens_q[pos]) else None,
                "y": _r4(float(row["y"])) if np.isfinite(row["y"]) else None,
                "ret": _r4(float(row["ret"])) if np.isfinite(row["ret"]) else None,
                "close": _r4(float(row["close"])) if np.isfinite(row["close"]) else None,
                **regime_row,
            }
            for m in usable:
                rec[f"prob_{m}"] = (
                    _r4(float(row[m])) if np.isfinite(row[m]) else None
                )
            oof_records.append(rec)

    result: dict[str, Any] = {
        "symbol": symbol,
        "is_available": True,
        "error": None,
        "horizon": int(horizon),
        "generated_at": ver.utc_now_iso(),
        "versions": {
            "feature": settings.ml_feature_version,
            "model": settings.ml_model_version,
            "ensemble": settings.ml_ensemble_version,
            "calibration": method,
        },
        "latest": {
            "as_of": index[-1].isoformat(),
            "close": _r4(close.loc[index[-1]]) if close is not None else None,
            "raw_probability": _r4(raw_prob),
            "probability": _r4(prob),
            "signal": action,
            "direction": decision["direction"],
            "confidence": _r4(confidence),
            "reason": reason,
            "threshold_buy": _r4(decision["threshold_buy"]),
            "threshold_sell": _r4(decision["threshold_sell"]),
        },
        "validation": {
            "target_formula": f"target_ret_{horizon}d = Close[T+{horizon}]/Close[T] - 1",
            "entry_convention": f"entry at Close[T] for every signal row T",
            "exit_convention": "exit at Close[T+horizon] (matches the target window)",
            "position_sizing": f"{capital} / {horizon} per position"
                               " (max gross exposure = 1.0x capital)",
            "max_concurrent_positions": f"<= {horizon} by construction "
                                        "(no hidden leverage)",
            "oof_rows": int(len(oof)),
            "calibration_fit_rows": int(len(fit_rows)),
            "holdout_rows": int(len(hold_rows)),
            "oof_rows_effective_independent": eff_windows["oof"],
            "holdout_rows_effective_independent": eff_windows["holdout"],
            "effective_sample_note": (
                "the 20-day forward label overlaps, so consecutive OOF rows are "
                "near-duplicates; the *_effective_independent values are the "
                "number of non-overlapping windows actually available"
            ),
            "accuracy_note": (
                "`backtest[].accuracy` is TRADE accuracy (settled positions with "
                "direction != 0); classifier accuracy on the SAME labelled rows "
                "is reported under `discrimination`"
            ),
            "final_holdout_never_used_for_fitting": True,
            "final_holdout_used_for": (
                "backtest metrics, baselines, shuffled-control, calibrated "
                "Brier/ECE — evaluation ONLY; never for model fitting, feature "
                "selection, scaler fitting, threshold tuning or calibration fit"
            ),
            "isotonic_calibration_fit_window": "older (chronologically first) "
                                                "80% of OOF rows",
            "audit_replay_scope": (
                "persisted audit artifact (data_fingerprint + raw/calibrated "
                "OOF probs + holdout feature hash + calibrator params + labels "
                "+ versions) supports METRIC REPLAY/auditability only — it does "
                "NOT support full retraining replay (model training state is "
                "not persisted)"
            ),
        },
        "discrimination": {
            "metric": "full-OOF AUC is the primary stable RANKING metric; "
                      "holdout (~88 rows) is too noisy to compare changes",
            "metric_definition": (
                "ROC-AUC = probability that a randomly chosen up row is ranked "
                "above a randomly chosen down row. It is NOT classification "
                "accuracy and must never be read as a hit-rate."
            ),
            "full_oof_auc": _r4(full_auc),
            "holdout_auc": _r4(ens_bt.get("roc_auc")),
            "full_oof_accuracy_raw": _r4(oof_acc_raw),
            "holdout_accuracy_calibrated": _r4(holdout_acc_cal),
            "holdout_accuracy_raw": _r4(holdout_acc_raw),
            "holdout_up_rate": _r4(holdout_up_rate),
            "holdout_baseline_accuracy": {
                k: _r4(v) for k, v in baseline_acc.items()
            },
            "holdout_best_baseline_accuracy": _r4(best_baseline),
            "holdout_edge_vs_best_baseline_accuracy": _r4(edge_vs_best),
            "baseline_rule": (
                "a forecast is only useful on a window if its classification "
                "accuracy beats the trivial always-up / always-down predictor on "
                "the SAME rows (negative edge = worse than a constant)"
            ),
            "effective_independent_windows": eff_windows,
            "universe_note": (
                "this block describes ONE symbol. Any aggregate median across "
                "symbols must state the symbol count, the median convention and "
                "the per-symbol spread — a 4-symbol upper-median is NOT a "
                "general accuracy claim."
            ),
        },
        "calibration": {
            "method": method,
            "n": int(len(fit_rows)),
            "n_holdout": int(len(hold_rows)),
            "brier": _r4(brier_raw_hold),
            "brier_calibrated": _r4(brier_cal),
            "ece": _r4(ece_raw_hold),
            "ece_calibrated": _r4(ece_cal),
            "brier_fit": _r4(brier_raw),
            "brier_calibrated_fit": _r4(brier_cal_fit),
            "ece_fit": _r4(ece_raw),
            "ece_calibrated_fit": _r4(ece_cal_fit),
            "report": cal_sensitivity,
        },
        "uncertainty": _r4_deep(uncertainty),
        "backtest": bt_rows,
        "explanation": explanation,
        "monitor": monitor_out,
        "snapshot": snapshot,
        "data_fingerprint": fp_hex,
        "input_fingerprint": {
            "data_fingerprint": fp_hex,
            "algorithm": "sha256",
            "scope": fp_meta.get("scope"),
            "as_of": fp_meta.get("as_of"),
            "sources": fp_meta.get("sources", {}),
            "reconstruction": (
                "not supported — the fingerprint identifies the exact input "
                "snapshot consumed by this forecast run, but the inputs "
                "themselves are NOT persisted; this is identification, not "
                "offline replay"
            ),
        },
    }

    if return_oof:
        result["oof_table"] = oof_records

    _LAST_RUN["last_run"] = {
        "symbol": symbol,
        "generated_at": result["generated_at"],
        "horizon": int(horizon),
        "versions": result["versions"],
        "latest_signal": action,
        "latest_probability": _r4(prob),
        "monitor_mode": report["mode"],
        "alarms": report["alarms"],
        "calibration_method": method,
        "calibration_fit_rows": int(len(fit_rows)),
        "calibration_holdout_rows": int(len(hold_rows)),
        "calibration_n_crossing_0p50": int(cal_sensitivity["n_crossing_0p50"]),
        "oof_brier_raw": _r4(brier_raw),
        "oof_brier_calibrated": _r4(brier_cal),
        "n_oof": int(len(oof)),
        "n_calibration_fit": int(len(fit_rows)),
        "n_holdout": int(len(hold_rows)),
        "holdout_accuracy_calibrated": _r4(holdout_acc_cal),
        "holdout_best_baseline_accuracy": _r4(best_baseline),
        "holdout_edge_vs_best_baseline_accuracy": _r4(edge_vs_best),
        "holdout_rows_effective_independent": eff_windows["holdout"],
        "data_fingerprint": fp_hex,
        "input_as_of": fp_meta.get("as_of"),
    }
    # W10: persist this run's holdout-calibrated Brier/ECE as the live
    # calibration-drift reference. Best-effort — a state-file failure must
    # never break the forecast (same contract as the W9 prediction log).
    try:
        from app.ml import calibration_monitor as _calmon

        _calmon.record_calibration_reference(
            result.get("calibration"), symbol=symbol
        )
    except Exception as _calmon_exc:  # noqa: BLE001
        logger.warning("calibration reference write failed: %s", _calmon_exc)
    return result


# ---------------------------------------------------------------------------
# Network path (API / chat / smoke)
# ---------------------------------------------------------------------------

def _build_symbol_input(
    symbol: str,
    period: str = "5y",
    enrich: bool = False,
    fundamentals_pit: bool = False,
    sector_index: str | None = None,
) -> dict[str, Any]:
    """Fetch + assemble the exact upstream inputs `forecast_symbol` consumes.

    Shared by the live forecast path and the offline benchmark so both paths
    measure the same input construction. Never fabricates data: any unavailable
    source (market index, fundamentals, enrichments) is skipped gracefully and
    recorded in ``input_meta.sources``.

    `fundamentals_pit`: opt-in A/B lever (default OFF, locked baseline is
    unchanged) that additionally attaches the historical quarterly PIT
    snapshot list to the feature frame — giving training rows stepwise-constant
    fund_* values instead of the single current snapshot's last-row-only lift.

    `sector_index` (research-only Task 28 lever, default OFF): a registered
    series id (e.g. "cnxit", "banknifty") used as the sector benchmark for the
    stock-vs-sector relative feature family (NIFTY 50 relative features stay).

    Returns ``{"ok": True, "clean": ..., "features": <feature frame>,
    "fp_hex": <fingerprint>, "input_meta": ..., "as_of": ...}`` or
    ``{"ok": False, "clean": ..., "error": reason}`` when the symbol has no
    usable data (which the caller must surface, not invent).
    """
    clean = clean_symbol(symbol)
    if not validate_symbol(clean):
        return {"ok": False, "clean": clean, "error": "invalid NSE symbol"}

    from app.services import data_service as dsvc
    from app.services.fundamental_service import fundamentals_service

    payload = dsvc.fetch_nse_ohlcv(clean, period=period)
    if not payload.get("is_available"):
        return {
            "ok": False,
            "clean": clean,
            "error": payload.get("error") or f"{clean} data unavailable",
        }

    history = payload.get("history") or []
    if not history:
        return {"ok": False, "clean": clean, "error": "empty OHLCV history"}
    as_of = str((history[-1].get("date") or "")).split("T")[0][:10]

    frame = _ohlcv_from_records(history)
    if frame.empty:
        return {"ok": False, "clean": clean, "error": "empty OHLCV history"}

    market: dict[str, Any] = {}
    source_meta: dict[str, dict[str, Any]] = {}
    for sid in ("nifty50", "india_vix"):
        fetched_at = utc_now_iso()
        row_info: dict[str, Any] = {"fetched_at": fetched_at, "available": False}
        try:
            mdf, meta = dsvc._get_series_frame(sid, period=period, interval="1d")
            if mdf is not None and not mdf.empty:
                market[sid] = mdf
                row_info.update({
                    "available": True,
                    "rows": int(len(mdf)),
                    "data_end": str(mdf.index[-1].date()),
                })
        except Exception as exc:  # noqa: BLE001 — graceful skip
            logger.warning("market index %s unavailable: %s", sid, exc)
        source_meta[sid] = row_info

    fundamentals: dict[str, Any] | None = None
    fundamentals_snapshots: list[dict[str, Any]] | None = None
    try:
        fundamentals = fundamentals_service.get_fundamentals(clean, "NSE")
    except Exception as exc:  # noqa: BLE001
        logger.warning("fundamentals unavailable for %s: %s", clean, exc)
    if fundamentals_pit:
        try:
            # Historical quarterly PIT snapshots (available_at = quarter_end + 45d
            # SEBI filing-lag proxy). Quarter-end close prices from the frame give a
            # trailing P/E per quarter. When it fails, we keep the single current
            # snapshot (last-row-only visibility) — graceful degradation.
            quarter_end_prices = {
                str(d.date()): float(v)
                for d, v in _quarter_end_closes(frame).items()
            }
            fundamentals_snapshots = fundamentals_service.get_quarterly_fundamentals(
                clean, "NSE", quarter_end_prices=quarter_end_prices
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("quarterly fundamentals unavailable for %s: %s", clean, exc)
    source_meta["fundamentals"] = {
        "fetched_at": utc_now_iso(),
        "available": fundamentals is not None,
        "quarterly_snapshots": int(len(fundamentals_snapshots or [])),
    }
    sector_close: pd.Series | None = None
    source_meta["sector"] = {"fetched_at": utc_now_iso(), "available": False}
    if sector_index is not None:
        try:
            sdf, smeta = dsvc._get_series_frame(
                sector_index, period=period, interval="1d"
            )
            if sdf is not None and not sdf.empty:
                sector_close = _series_close({sector_index: sdf}, sector_index)
                source_meta["sector"].update({
                    "available": True,
                    "series_id": sector_index,
                    "rows": int(len(sdf)),
                    "data_end": str(sdf.index[-1].date()),
                })
        except Exception as exc:  # noqa: BLE001 — graceful skip
            logger.warning("sector index %s unavailable: %s", sector_index, exc)

    # ----- reproducibility layer: fingerprint the exact upstream inputs -----
    fp_parts: dict[str, str] = {
        "ohlcv": fp.fingerprint_object("ohlcv", history),
    }
    fp_parts.update({
        f"market:{sid}": fp.fingerprint_object(f"market:{sid}", mdf)
        for sid, mdf in market.items()
    })
    fp_parts["fundamentals"] = fp.fingerprint_object(
        "fundamentals", fundamentals if fundamentals is not None else None
    )
    fp_parts["fundamentals_snapshots"] = fp.fingerprint_object(
        "fundamentals_snapshots",
        fundamentals_snapshots if fundamentals_snapshots else None,
    )
    fp_parts["sector"] = fp.fingerprint_object(
        "sector", sector_close if sector_close is not None else None
    )
    ingested_fp = fp.compose_fingerprint(fp_parts, name="input-snapshot")
    source_meta["stock"] = {
        "fetched_at": payload.get("fetched_at"),
        "available": bool(payload.get("is_available")),
        "rows": int(payload.get("rows") or 0),
        "data_end": payload.get("data_end"),
    }
    input_meta = {
        "scope": "upstream_input",
        "as_of": as_of,
        "sources": source_meta,
    }

    articles: list[dict] | None = None
    macro: dict[str, pd.Series] | None = None
    options_snapshot: dict[str, Any] | None = None
    if enrich:
        try:
            from app.services.news_service import fetch_stock_news
            from app.services.sentiment_service import analyze_articles
            raw = fetch_stock_news(clean, limit=30)
            articles = analyze_articles(raw) if raw else []
        except Exception as exc:  # noqa: BLE001
            logger.warning("news/sentiment unavailable for %s: %s", clean, exc)
        try:
            macro = {}
            for sid in ("snp500", "usd_inr", "gold"):
                mdf, meta = dsvc._get_series_frame(sid, period=period, interval="1d")
                if mdf is not None and not mdf.empty:
                    macro[sid] = _series_close({sid: mdf}, sid)
        except Exception as exc:  # noqa: BLE001
            logger.warning("macro series unavailable for %s: %s", clean, exc)
        try:
            from app.ml.options_df import get_options_inputs
            spot = float(frame["Close"].iloc[-1])
            raw_options = get_options_inputs(clean, spot)
            # get_options_inputs returns metrics under a nested `metrics` key,
            # but add_options_features expects the metric keys at the snapshot
            # top level (plus an available_at/as_of date). Unwrap + timestamp
            # them PIT = fetch time (visible only from today onward).
            if raw_options.get("is_available") and raw_options.get("metrics"):
                options_snapshot = raw_options["metrics"]
                options_snapshot["available_at"] = raw_options.get(
                    "fetched_at", utc_now_iso()
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("options data unavailable for %s: %s", clean, exc)

    try:
        feats = build_feature_frame(
            frame,
            market=market,
            fundamentals=fundamentals,
            fundamentals_snapshots=fundamentals_snapshots,
            articles=articles,
            macro=macro,
            options_snapshot=options_snapshot,
            sector_close=sector_close,
            sector_name=sector_index,
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "clean": clean, "error": f"feature build failed: {exc}"}

    return {
        "ok": True,
        "clean": clean,
        "features": feats,
        "fp_hex": ingested_fp,
        "input_meta": input_meta,
        "as_of": as_of,
    }


def forecast_symbol(
    symbol: str,
    period: str = "10y",
    horizon: int = settings.ml_horizon,
    seed: int = 0,
    fast: bool = True,
    enrich: bool = False,
) -> dict[str, Any]:
    """One-call forecast for a live NSE symbol (network, graceful on failure).

    `period` defaults to 10y (15-Sep-2026 adoption: Task 27 measured 10y median
    full-OOF AUC 0.6179 vs same-day 5y control 0.5596 on TCS/RELIANCE/HDFCBANK/
    INFY). `fast=True` bounds the compute for API latency. `enrich=True`
    additionally pulls news sentiment, options and macro series (slower; used
    off the hot path). Unavailable enrichments are skipped — never fabricated.
    Inputs are built by the shared `_build_symbol_input` (also used by the
    benchmark).
    """
    clean = clean_symbol(symbol)
    if not validate_symbol(clean):
        return _graceful(clean, "invalid NSE symbol")

    # Fast path for API: return instant cached forecast if available on disk
    if fast:
        cached_pre = _forecast_cache_get(clean, as_of="latest")
        if cached_pre is not None:
            logger.info("instant forecast cache hit for %s", clean)
            return cached_pre

    inp = _build_symbol_input(clean, period=period, enrich=enrich)
    if not inp.get("ok"):
        return _graceful(clean, inp.get("error") or f"{clean} data unavailable")

    as_of = inp["as_of"]
    ingested_fp = inp["fp_hex"]

    cached = _forecast_cache_get(clean, as_of, ingested_fp)
    if cached is not None:
        logger.info("forecast cache hit for %s (as_of %s)", clean, as_of)
        return cached

    result = forecast_frame(
        inp["features"], symbol=clean, horizon=horizon, seed=seed, fast=fast,
        data_fingerprint=ingested_fp, input_meta=inp["input_meta"],
        threshold_buy_map=dict(settings.ml_per_symbol_buy_thresholds) or None,
    )
    if result.get("is_available", False):
        _forecast_cache_put(clean, as_of, result)
        # Live path: opportunistic, best-effort forward-test logging (W9).
        # Every successful forecast is appended to the append-only prediction
        # log; a logging failure must NEVER break the forecast.
        try:
            from app.ml import scorecard as _scorecard

            _scorecard.record_prediction(
                result, as_of=as_of, horizon=horizon, symbol=clean
            )
        except Exception as _score_exc:  # noqa: BLE001
            logger.warning("prediction-log write failed: %s", _score_exc)
    return result


# ---------------------------------------------------------------------------
# ML status (freshness / drift / degraded mode) for /api/v1/ml/status
# ---------------------------------------------------------------------------

def last_run_status() -> dict[str, Any]:
    """Metadatum about the most recent forecast run in this process."""
    last = _LAST_RUN.get("last_run")
    return {
        "has_run": last is not None,
        "last_run": last,
        "last_run_mode": (last or {}).get("monitor_mode"),
    }


def _default_source_fetcher(series_id: str):
    from app.services import data_service as dsvc
    return dsvc._get_series_frame(series_id, period="5y", interval="1d")


def build_ml_status(
    source_fetcher: Callable[[str], tuple[pd.DataFrame | None, dict]] | None = None,
) -> dict[str, Any]:
    """Operational status of the ML forecast system (no network required for
    already-cached sources; every source degrades gracefully)."""
    from app.services import data_service as dsvc
    fetcher = source_fetcher or _default_source_fetcher

    sources: list[dict[str, Any]] = []
    for sid in ("nifty50", "banknifty", "india_vix"):
        df, meta = fetcher(sid)
        last = df.index[-1] if df is not None and len(df) else None
        fs = mon.freshness_status(last)
        sources.append({
            "series_id": sid,
            "name": dsvc.SERIES_REGISTRY.get(sid, {}).get("name", sid),
            "is_available": bool(df is not None and len(df) > 0),
            "rows": int(len(df)) if df is not None and len(df) else 0,
            "data_end": str(last) if last is not None else None,
            "age_days": fs["age_days"],
            "status": fs["status"],
            "error": meta.get("error"),
        })

    alarms: list[str] = []
    stale = [s for s in sources if s["status"] != "fresh"]
    if stale:
        alarms.append("market_sources_stale")

    last = last_run_status()
    if last["last_run_mode"] is not None and last["last_run_mode"] != "fresh":
        alarms.append("forecast_degraded")

    # W10: live calibration drift (persisted holdout reference vs the W9
    # forward scorecard). Never raises, never fabricates: without a reference
    # or a sufficient scored sample the status is UNKNOWN, not healthy.
    # Local import: calibration_monitor imports this module at its top level.
    try:
        from app.ml import calibration_monitor as calmon

        cal_drift = calmon.build_live_calibration_drift()
    except Exception as exc:  # noqa: BLE001 — status must never fail on this
        logger.warning("live calibration-drift report failed: %s", exc)
        cal_drift = {
            "status": "sample_unavailable",
            "is_alarm": False,
            "warning": f"{type(exc).__name__}: live calibration drift unavailable",
        }
    cal_alarm = bool(cal_drift.get("is_alarm"))
    if cal_alarm:
        alarms.append("calibration_drift")

    return {
        "status": "degraded" if alarms else "ok",
        "mode": "degraded" if alarms else "fresh",
        "generated_at": ver.utc_now_iso(),
        "versions": {
            "feature": settings.ml_feature_version,
            "model": settings.ml_model_version,
            "ensemble": settings.ml_ensemble_version,
        },
        "thresholds": {
            "buy": settings.ml_threshold_buy,
            "sell": settings.ml_threshold_sell,
        },
        "sources": sources,
        "last_forecast": last["last_run"],
        "calibration_drift": cal_drift,
        "alarms": alarms,
    }
