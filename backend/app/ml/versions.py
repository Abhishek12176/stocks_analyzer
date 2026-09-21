"""Data + model versioning for the 20-day forecasting system.

Implements the "Data + Model Versioning" requirement: every prediction can
store data timestamp, feature version, model version, training period,
prediction timestamp, feature snapshot, probability, signal and the actual
future outcome — so we can later evaluate whether the model is improving.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.config import settings


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def make_prediction_id(symbol: str, feature_version: str | None = None) -> str:
    """Human-readable, sortable prediction id (symbol + timestamp)."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    return f"{symbol.upper()}_{feature_version or settings.ml_feature_version}_{ts}"


_SNAPSHOT_EXTRA_KEYS = {
    "ensemble_version", "calibration_method", "calibrated_probability",
    "calibrated_signal",
}


def new_snapshot(
    symbol: str,
    *,
    model_version: str | None = None,
    feature_snapshot: dict[str, Any] | None = None,
    model_probability: float | None = None,
    signal: str | None = None,
    prediction_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a versioned prediction snapshot.

    Expected outcome fields:
    - data_timestamp / feature_version / model_version / training_period
    - prediction_timestamp / feature_snapshot
    - model_probability / signal
    - actual_future_outcome (filled later during evaluation)

    Probabilities are rounded to 4 decimal places to avoid displaying fake
    precision (e.g. "68.347829%") that the model cannot justify.
    """
    allowed = {k: v for k, v in extra.items() if k in _SNAPSHOT_EXTRA_KEYS}
    allowed = {k: v for k, v in allowed.items() if v is not None}
    if "calibrated_probability" in allowed:
        cp = allowed["calibrated_probability"]
        if cp is not None and not 0.0 <= cp <= 1.0:
            raise ValueError("calibrated_probability must be in [0, 1]")
        allowed["calibrated_probability"] = round(cp, 4)
    if model_probability is not None:
        if not 0.0 <= model_probability <= 1.0:
            raise ValueError("model_probability must be in [0, 1]")
        model_probability = round(model_probability, 4)
    return {
        "prediction_id": prediction_id or make_prediction_id(symbol),
        "symbol": symbol.upper(),
        "data_timestamp": utc_now_iso(),
        "feature_version": settings.ml_feature_version,
        "model_version": model_version or settings.ml_model_version,
        "training_period": settings.ml_training_period,
        "prediction_timestamp": utc_now_iso(),
        "feature_snapshot": feature_snapshot or {},
        "model_probability": model_probability,
        "signal": signal,
        "actual_future_outcome": None,
    } | allowed


def record_outcome(snapshot: dict[str, Any], actual_outcome: dict[str, Any]) -> dict[str, Any]:
    """Attach the realised future outcome to a snapshot (after the horizon elapses)."""
    snap = dict(snapshot)
    snap["actual_future_outcome"] = actual_outcome
    return snap