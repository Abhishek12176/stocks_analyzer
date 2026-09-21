"""Response schemas for the Task 10 forecast / ML-status APIs.

Mirrors the structure produced by `app.ml.pipeline` (snake_case here, emitted
as camelCase via `BaseSchema`). `ForecastResponse` is also embedded as an
optional `forecast` block inside the full-analysis response.
"""

from __future__ import annotations

from typing import Any, Optional

from app.schemas.base import BaseSchema


class ForecastLatest(BaseSchema):
    as_of: str
    close: Optional[float] = None
    raw_probability: Optional[float] = None
    probability: Optional[float] = None
    signal: Optional[str] = None
    direction: int = 0
    confidence: Optional[float] = None
    reason: Optional[str] = None
    threshold_buy: Optional[float] = None
    threshold_sell: Optional[float] = None


class CalibrationInfo(BaseSchema):
    method: str = "uncalibrated"
    n: int = 0
    n_holdout: Optional[int] = None
    brier: Optional[float] = None
    brier_calibrated: Optional[float] = None
    ece: Optional[float] = None
    ece_calibrated: Optional[float] = None
    brier_fit: Optional[float] = None
    brier_calibrated_fit: Optional[float] = None
    ece_fit: Optional[float] = None
    ece_calibrated_fit: Optional[float] = None
    report: dict[str, Any] = {}


class ForecastExplanation(BaseSchema):
    model: str
    baseline_prob: float
    delta_scale: str
    factors: list[dict[str, Any]]


class ForecastResponse(BaseSchema):
    symbol: str
    is_available: bool
    error: Optional[str] = None
    horizon: int = 20
    generated_at: str
    versions: dict[str, Any] = {}
    latest: Optional[ForecastLatest] = None
    calibration: Optional[CalibrationInfo] = None
    discrimination: dict[str, Any] = {}
    validation: dict[str, Any] = {}
    data_fingerprint: Optional[str] = None
    input_fingerprint: Optional[dict[str, Any]] = None
    backtest: list[dict[str, Any]] = []
    explanation: Optional[ForecastExplanation] = None
    uncertainty: Optional[dict[str, Any]] = None
    monitor: dict[str, Any] = {}
    snapshot: dict[str, Any] = {}


class MlSourceStatus(BaseSchema):
    series_id: str
    name: str
    is_available: bool
    rows: int = 0
    data_end: Optional[str] = None
    age_days: Optional[int] = None
    status: str
    error: Optional[str] = None


class MlStatusResponse(BaseSchema):
    status: str
    mode: str
    generated_at: str
    versions: dict[str, Any] = {}
    thresholds: dict[str, Any] = {}
    sources: list[MlSourceStatus] = []
    last_forecast: Optional[dict[str, Any]] = None
    calibration_drift: Optional[dict[str, Any]] = None
    alarms: list[str] = []


class ScorecardMetrics(BaseSchema):
    n: int = 0
    accuracy: Optional[float] = None
    brier: Optional[float] = None
    up_rate: Optional[float] = None
    always_up_accuracy: Optional[float] = None
    best_baseline_accuracy: Optional[float] = None
    edge_vs_best_baseline_accuracy: Optional[float] = None
    mean_signed_horizon_return: Optional[float] = None
    hit_rate_when_predicting_up: Optional[float] = None


class ScorecardResponse(BaseSchema):
    schema_version: str = "scorecard-v1"
    generated_at: str
    symbol: Optional[str] = None
    n_logged: int = 0
    n_scored: int = 0
    n_waiting: int = 0
    min_scored_for_conclusion: int = 5
    is_sufficient_sample: bool = False
    metrics: Optional[dict[str, Any]] = None
    per_symbol: dict[str, Any] = {}
    note: Optional[str] = None
    warning: Optional[str] = None


class CalibrationDriftResponse(BaseSchema):
    """Live calibration-drift report (W10): persisted holdout reference vs
    the W9 forward scorecard. `is_alarm` fires only with a sufficient scored
    sample; otherwise the status honestly says UNKNOWN (`no_reference`,
    `insufficient_sample`, ...)."""
    schema_version: str = "calibration-drift-v1"
    generated_at: str
    symbol: Optional[str] = None
    min_recent_for_alarm: int = 5
    threshold: Optional[float] = None
    n_recent: int = 0
    reference: Optional[dict[str, Any]] = None
    live: Optional[dict[str, Any]] = None
    delta_brier: Optional[float] = None
    delta_ece: Optional[float] = None
    is_alarm: bool = False
    status: str = "no_scored_predictions"
    note: Optional[str] = None
    warning: Optional[str] = None