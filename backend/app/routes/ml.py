"""ML monitoring route — freshness / drift / degraded-mode status (Task 10)
plus the forward prediction scorecard (W9)."""

import asyncio
import logging

from fastapi import APIRouter, Query

from app.ml.pipeline import build_ml_status
from app.schemas.forecast import (
    CalibrationDriftResponse,
    MlStatusResponse,
    ScorecardResponse,
)

logger = logging.getLogger("equitylens.ml")
_TIMEOUT = 60

router = APIRouter(prefix="/ml", tags=["ml"])


@router.get("/status", response_model=MlStatusResponse)
async def get_ml_status():
    """Operational status of the 20-day forecasting system.

    Reports per-source data freshness, the latest forecast run (when any),
    configured thresholds/versions and any drift alarms. Everything degrades
    gracefully — a source outage never fabricates a clean bill of health.
    """
    try:
        status = await asyncio.wait_for(
            asyncio.to_thread(build_ml_status), timeout=_TIMEOUT
        )
    except Exception as exc:  # noqa: BLE001
        from app.ml import versions as ver
        logger.exception("ml status failed")
        status = {
            "status": "degraded",
            "mode": "degraded",
            "generated_at": ver.utc_now_iso(),
            "versions": {},
            "thresholds": {},
            "sources": [],
            "last_forecast": None,
            "alarms": ["ml_status_error"],
            "error": f"{type(exc).__name__}: {exc}",
        }

    return MlStatusResponse(**status)


@router.get("/scorecard", response_model=ScorecardResponse)
async def get_ml_scorecard(
    symbol: str | None = Query(
        default=None, description="NSE symbol (e.g. RELIANCE); omit for all logged symbols"
    ),
):
    """Forward (in-time) prediction scorecard (W9).

    Scores each LOGGED live prediction against the realized T -> T+horizon
    outcome (20 trading days later) and aggregates real forward-test numbers:
    hit accuracy, Brier, the always-up baseline on the SAME predictions and the
    edge over it. Predictions whose horizon has not elapsed stay `waiting` and
    are NEVER counted. A thin sample (< 5 scored) is labelled insufficient —
    warnings, never conclusions.
    """
    from app.ml import scorecard as sc

    clean = (symbol or "").strip().upper() or None

    def _compute() -> dict:
        try:
            # Fresh prices: score whatever the history now reaches, best-effort.
            sc.score_pending_predictions(symbol=clean, close_lookup=sc.build_live_close_lookup())
        except Exception as exc:  # noqa: BLE001 — scoring must not fail the API
            logger.warning("scorecard pending-scoring failed: %s", exc)
        return sc.scorecard_report(clean)

    try:
        report = await asyncio.wait_for(asyncio.to_thread(_compute), timeout=_TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        logger.exception("ml scorecard failed")
        from datetime import datetime as _dt
        from datetime import timezone as _tz

        return ScorecardResponse(
            generated_at=_dt.now(_tz.utc).isoformat(),
            symbol=clean,
            warning=f"{type(exc).__name__}: scorecard unavailable",
        )
    return ScorecardResponse(**report)


@router.get("/calibration-drift", response_model=CalibrationDriftResponse)
async def get_calibration_drift(
    symbol: str | None = Query(
        default=None, description="NSE symbol (e.g. RELIANCE); omit for all logged symbols"
    ),
):
    """Live calibration-drift report (W10).

    Compares the persisted holdout-calibrated Brier/ECE (recorded at forecast
    time) against the LIVE forward scorecard: the already-scored W9
    predictions are the current sample. The alarm fires only when the Brier
    delta exceeds the configured threshold AND the scored sample is
    sufficient; otherwise the status honestly reports UNKNOWN
    (`no_reference`, `insufficient_sample`, `no_scored_predictions`).
    """
    from app.ml import calibration_monitor as cm

    clean = (symbol or "").strip().upper() or None
    try:
        report = await asyncio.wait_for(
            asyncio.to_thread(cm.build_live_calibration_drift, clean),
            timeout=_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("ml calibration-drift failed")
        from datetime import datetime as _dt
        from datetime import timezone as _tz

        return CalibrationDriftResponse(
            generated_at=_dt.now(_tz.utc).isoformat(),
            symbol=clean,
            status="sample_unavailable",
            warning=f"{type(exc).__name__}: calibration-drift unavailable",
        )
    return CalibrationDriftResponse(**report)