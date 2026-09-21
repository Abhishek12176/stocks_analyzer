"""Operational monitoring for the forecasting system (Task 9).

`monitor.assess()` pulls together the three classic drift signals plus data
freshness and reports an overall operating mode:
- data freshness  : raw last price date vs now (calendar days).
- feature drift   : population-stability-index (PSI) per feature between a
                    reference distribution and the current distribution.
- prediction drift: PSI between reference and current P(up) distributions.
- calibration drift: Brier deterioration since the last evaluation.

Any violation flips the mode to `degraded` (or `stale` when the data itself
is old). Nothing here calls the network — callers pass in pre-computed arrays.
"""

from __future__ import annotations

from datetime import datetime, date, timezone
from typing import Any, Iterable

import numpy as np

from app.config import settings


def _to_dt(day: datetime | date) -> datetime:
    if isinstance(day, datetime):
        return day.replace(tzinfo=day.tzinfo or timezone.utc)
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc)


def freshness_status(last_date: datetime | date | None,
                     now: datetime | date | None = None,
                     max_days: int | None = None) -> dict:
    """Flag data as `fresh` or `stale` based on calendar age of last row."""
    if last_date is None:
        return {"status": "stale", "age_days": None,
                "last_date": None, "max_days": max_days}
    max_days = settings.ml_freshness_max_days if max_days is None else max_days
    now_dt = _to_dt(now or datetime.now(timezone.utc))
    age = (now_dt - _to_dt(last_date)).days
    return {"status": "fresh" if age <= max_days else "stale",
            "age_days": int(age), "last_date": str(last_date),
            "max_days": max_days}


def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10,
        epsilon: float = 1e-6) -> float:
    """Population stability index between two 1-D sample distributions."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    lo, hi = float(ref.min()), float(ref.max())
    if hi - lo < epsilon:
        lo, hi = lo - 1.0, hi + 1.0
    edges = np.linspace(lo, hi, n_bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    r, _ = np.histogram(ref, bins=edges)
    c, _ = np.histogram(cur, bins=edges)
    rf = r / r.sum()
    cf = c / c.sum()
    rf = np.clip(rf, epsilon, None)
    cf = np.clip(cf, epsilon, None)
    return float(np.sum((cf - rf) * np.log(cf / rf)))


def feature_drift(reference: dict[str, np.ndarray],
                  current: dict[str, np.ndarray],
                  threshold: float | None = None) -> dict:
    """Per-feature PSI + `alarm` flag when any feature exceeds the threshold."""
    threshold = settings.ml_psi_threshold if threshold is None else threshold
    per = {name: psi(ref, current[name]) for name, ref in reference.items()
           if name in current}
    flagged = [name for name, v in per.items() if not np.isnan(v) and v > threshold]
    return {"per_feature": per, "max_psi": float(max(per.values())) if per else 0.0,
            "alarm": bool(flagged), "flagged": flagged, "threshold": threshold}


def prediction_drift(reference: np.ndarray, current: np.ndarray,
                     threshold: float | None = None) -> dict:
    """PSI between two P(up) distributions + alarm flag."""
    threshold = settings.ml_psi_threshold if threshold is None else threshold
    value = psi(reference, current)
    return {"psi": float(value), "alarm": bool(not np.isnan(value) and value > threshold),
            "threshold": threshold}


def calibration_drift(reference_brier: float, current_brier: float,
                      threshold: float | None = None) -> dict:
    """Brier deterioration (higher brier shoots alarms when it grows too much)."""
    threshold = settings.ml_brier_drift_threshold if threshold is None else threshold
    delta = float(current_brier) - float(reference_brier)
    return {"reference_brier": float(reference_brier), "current_brier": float(current_brier),
            "delta": delta, "alarm": bool(delta > threshold), "threshold": threshold}


def assess(
    last_date: datetime | date | None,
    feature_ref: dict[str, np.ndarray],
    feature_cur: dict[str, np.ndarray],
    pred_ref: np.ndarray,
    pred_cur: np.ndarray,
    cal_ref_brier: float,
    cal_cur_brier: float,
    now: datetime | date | None = None,
) -> dict[str, Any]:
    """One-stop monitoring report with an overall operating mode."""
    fresh = freshness_status(last_date, now=now)
    f_drift = feature_drift(feature_ref, feature_cur)
    p_drift = prediction_drift(pred_ref, pred_cur)
    c_drift = calibration_drift(cal_ref_brier, cal_cur_brier)
    alarms: list[str] = []
    if not fresh["status"] == "fresh":
        alarms.append("data_stale")
    if f_drift["alarm"]:
        alarms.append("feature_drift")
    if p_drift["alarm"]:
        alarms.append("prediction_drift")
    if c_drift["alarm"]:
        alarms.append("calibration_drift")
    mode = "fresh" if not alarms else ("stale" if "data_stale" in alarms else "degraded")
    return {
        "mode": mode,
        "freshness": fresh,
        "feature_drift": {k: v for k, v in f_drift.items() if k != "per_feature"},
        "feature_psi": f_drift["per_feature"],
        "prediction_drift": p_drift,
        "calibration_drift": c_drift,
        "alarms": alarms,
    }