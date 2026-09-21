"""W10 — live calibration-drift monitor (closing the last monitoring gap).

`monitor.assess()` already compares a *reference* Brier against a *current*
one, but both numbers were handed in by the caller — nothing in production
persisted a reference, so the calibration-drift alarm could never fire on
real data. This module makes the loop real and honest end-to-end:

1. `record_calibration_reference()` — after a successful forecast the
   pipeline's holdout-calibrated Brier/ECE is persisted (best-effort, never
   breaks the forecast) as the reference for future live comparisons.
2. `calibration_drift_report()` — compares the reference against the LIVE
   forward scorecard: the already-scored predictions in the W9 prediction
   log are the "current" sample (true out-of-sample, in-time).
3. `build_live_calibration_drift()` — one-call report for the
   `GET /api/v1/ml/calibration-drift` route and `/ml/status`.

Honesty rules (mirroring W1/W9):
- Fewer than ``MIN_RECENT_FOR_ALARM`` scored predictions => status
  ``insufficient_sample`` — NEVER an alarm, NEVER a conclusion.
- No persisted reference => status ``no_reference`` — the report says
  drift is UNKNOWN instead of pretending all is well.
- Missing/NaN/non-finite calibration numbers are skipped, never fabricated.
- The alarm fires on the Brier delta only (pre-registered choice, same
  threshold semantics as `monitor.calibration_drift`); ECE is reported for
  diagnostics and never silently dropped.
- Reference reads never mutate; only `record_calibration_reference` and
  `clear_reference` write.

Nothing here calls the network.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.config import settings
from app.ml import pipeline as pl
from app.ml import versions as ver

logger = logging.getLogger("equitylens.calibration_monitor")

SCHEMA_VERSION = "calibration-drift-v1"

MIN_RECENT_FOR_ALARM = 5  # same sufficiency floor as the W9 scorecard


def default_state_path() -> Path:
    """Persisted reference state (single JSON file for all symbols)."""
    return pl.BACKEND_ROOT / "ml_cache" / "calibration_drift.json"


# ---------------------------------------------------------------------------
# Reference store (persisted)
# ---------------------------------------------------------------------------

def _finite(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _round4(value: Any) -> Any:
    f = _finite(value)
    return None if f is None else round(f, 4)


def load_reference(state_path: Path | None = None) -> dict[str, Any] | None:
    """Load the persisted calibration reference; None when absent/corrupt."""
    path = Path(state_path) if state_path is not None else default_state_path()
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("calibration reference unreadable (%s): %s", path, exc)
        return None
    if not isinstance(raw, dict) or _finite(raw.get("brier")) is None:
        return None
    return raw


def clear_reference(state_path: Path | None = None) -> bool:
    """Remove the persisted reference (True when something was removed)."""
    path = Path(state_path) if state_path is not None else default_state_path()
    try:
        if path.exists():
            path.unlink()
            return True
    except OSError as exc:
        logger.warning("calibration reference clear failed: %s", exc)
    return False


def record_calibration_reference(
    calibration: dict[str, Any] | None,
    *,
    symbol: str | None = None,
    state_path: Path | None = None,
) -> dict[str, Any]:
    """Persist the holdout-calibrated Brier/ECE of a finished forecast run.

    Accepts the ``result["calibration"]`` block produced by
    `pipeline.forecast_frame`. Skips (never fabricates) when the headline
    ``brier_calibrated`` number is missing or non-finite.
    """
    path = Path(state_path) if state_path is not None else default_state_path()
    cal = calibration if isinstance(calibration, dict) else {}
    brier = _finite(cal.get("brier_calibrated"))
    if brier is None:
        return {
            "status": "skipped", "reason": "no finite brier_calibrated",
            "path": str(path),
        }
    ref = {
        "brier": round(brier, 4),
        "ece": _round4(cal.get("ece_calibrated")),
        "brier_raw": _round4(cal.get("brier")),
        "n_holdout": int(cal["n_holdout"])
        if _finite(cal.get("n_holdout")) is not None else None,
        "method": cal.get("method"),
        "symbol": (symbol or "").upper() or None,
        "recorded_at": ver.utc_now_iso(),
        "schema_version": SCHEMA_VERSION,
    }
    existed = path.exists()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(ref, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
    except (OSError, ValueError) as exc:
        return {"status": "skipped", "reason": f"state write failed: {exc}",
                "path": str(path)}
    return {"status": "updated" if existed else "recorded", "path": str(path),
            "reference": ref}


# ---------------------------------------------------------------------------
# Live drift report (reference vs the W9 forward scorecard)
# ---------------------------------------------------------------------------

def _scored_rows(
    symbol: str | None = None, log_dir: Path | None = None
) -> list[dict[str, Any]]:
    """Scored W9 records (the only in-time sample we may judge live by)."""
    from app.ml import scorecard as sc

    recs = sc.load_predictions(symbol, log_dir)
    return [
        r for r in recs
        if r.get("scored") and isinstance(r.get("actual_future_outcome"), dict)
    ]


def calibration_drift_report(
    scored: list[dict[str, Any]],
    *,
    reference: dict[str, Any] | None = None,
    threshold: float | None = None,
    state_path: Path | None = None,
) -> dict[str, Any]:
    """Compare persisted reference calibration vs the live scored sample.

    Pure: reads the reference (argument, else persisted file) and never
    mutates it. Alarm fires only on the pre-registered Brier delta and only
    with a sufficient sample.
    """
    threshold = (
        settings.ml_brier_drift_threshold if threshold is None else float(threshold)
    )
    if reference is None:
        reference = load_reference(state_path)

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "min_recent_for_alarm": MIN_RECENT_FOR_ALARM,
        "threshold": _round4(threshold),
        "n_recent": len(scored),
        "reference": None,
        "live": None,
        "delta_brier": None, "delta_ece": None,
        "is_alarm": False,
        "status": "no_scored_predictions",
        "note": (
            "live Brier/ECE come from the W9 forward scorecard (already-scored "
            "predictions only); the alarm compares them against the persisted "
            "holdout-calibrated reference recorded at forecast time"
        ),
        "warning": None,
    }
    if reference is not None:
        report["reference"] = {
            "brier": _round4(reference.get("brier")),
            "ece": _round4(reference.get("ece")),
            "brier_raw": _round4(reference.get("brier_raw")),
            "n_holdout": reference.get("n_holdout"),
            "method": reference.get("method"),
            "symbol": reference.get("symbol"),
            "recorded_at": reference.get("recorded_at"),
        }
    if not scored:
        report["warning"] = (
            "no scored predictions yet — the forward sample cannot speak"
        )
        return report

    probs = np.array([float(r["probability"]) for r in scored], dtype=float)
    ups = np.array([bool(r["actual_future_outcome"]["label_up"]) for r in scored])
    live_brier = float(np.mean((probs - ups.astype(float)) ** 2))

    from app.ml import calibration as cal

    live_ece = float(cal.ece(probs, ups))
    report["live"] = {"brier": _round4(live_brier), "ece": _round4(live_ece)}

    if reference is None:
        report["status"] = "no_reference"
        report["warning"] = (
            "no persisted calibration reference yet — run a forecast to record "
            "one; until then calibration drift is UNKNOWN, not healthy"
        )
        return report

    ref_brier = _finite(reference.get("brier"))
    if ref_brier is None:
        report["status"] = "reference_unusable"
        report["warning"] = "persisted reference has no finite Brier"
        return report

    delta = live_brier - ref_brier
    report["delta_brier"] = _round4(delta)
    report["delta_ece"] = (
        _round4(live_ece - reference["ece"])
        if _finite(reference.get("ece")) is not None else None
    )
    if len(scored) < MIN_RECENT_FOR_ALARM:
        report["status"] = "insufficient_sample"
        report["warning"] = (
            f"only {len(scored)} scored prediction(s); below the "
            f"{MIN_RECENT_FOR_ALARM}-sample floor — no drift conclusion possible"
        )
        return report

    alarmed = bool(delta > threshold)
    report["is_alarm"] = alarmed
    report["status"] = "drift_detected" if alarmed else "ok"
    if alarmed:
        report["warning"] = (
            f"live Brier {live_brier:.4f} is {delta:.4f} worse than the "
            f"reference {ref_brier:.4f} (threshold {threshold:.4f}) — "
            "calibration degraded in the forward sample"
        )
    return report


def build_live_calibration_drift(
    symbol: str | None = None,
    *,
    log_dir: Path | None = None,
    state_path: Path | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """One-call live report (route entry point). Never raises."""
    clean = (symbol or "").strip().upper() or None
    try:
        scored = _scored_rows(clean, log_dir)
    except Exception as exc:  # noqa: BLE001 — degrade, never fabricate
        logger.warning("calibration-drift sample load failed: %s", exc)
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": ver.utc_now_iso(),
            "symbol": clean,
            "status": "sample_unavailable",
            "is_alarm": False,
            "min_recent_for_alarm": MIN_RECENT_FOR_ALARM,
            "n_recent": 0,
            "warning": f"{type(exc).__name__}: scored sample unavailable",
        }
    report = calibration_drift_report(
        scored, threshold=threshold, state_path=state_path
    )
    report["symbol"] = clean
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Live calibration-drift report: persisted holdout reference "
                    "vs the W9 forward scorecard."
    )
    ap.add_argument("--symbols", default="",
                    help="comma-separated symbols (default: everything logged)")
    args = ap.parse_args()
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    reports = [build_live_calibration_drift(s) for s in syms] or \
        [build_live_calibration_drift(None)]
    print(json.dumps(reports[0] if len(reports) == 1 else reports, indent=2))


if __name__ == "__main__":
    main()
