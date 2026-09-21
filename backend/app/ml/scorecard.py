from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from app.ml import pipeline as pl
from app.ml import versions as ver

logger = logging.getLogger("equitylens.scorecard")

SCHEMA_VERSION = "scorecard-v1"

MIN_SCORED_FOR_CONCLUSION = 5


def default_log_dir() -> Path:
    """Directory holding ``{SYMBOL}.jsonl`` prediction logs."""
    return pl.BACKEND_ROOT / "ml_cache" / "predictions"


def _log_path(symbol: str, log_dir: Path | None = None) -> Path:
    d = Path(log_dir) if log_dir is not None else default_log_dir()
    return d / f"{symbol.upper()}.jsonl"


def required_prediction_keys() -> list[str]:
    return [
        "prediction_id", "symbol", "as_of", "horizon", "generated_at",
        "probability", "raw_probability", "signal", "direction",
        "close", "feature_version", "model_version", "ensemble_version",
        "calibration_method", "data_fingerprint",
    ]


def _resolve_prediction_fields(
    snapshot: dict[str, Any], symbol: str, as_of: str, horizon: int
) -> dict[str, Any] | None:
    """Build the logged record from a forecast result dict."""
    if not isinstance(snapshot, dict) or not bool(snapshot.get("is_available", False)):
        return None
    latest = snapshot.get("latest") or {}
    prob = latest.get("probability")
    if prob is None or (isinstance(prob, float) and np.isnan(prob)):
        return None
    snap = snapshot.get("snapshot") or {}
    versions = snapshot.get("versions") or {}
    rec = {
        "prediction_id": snap.get("prediction_id") or ver.make_prediction_id(symbol),
        "symbol": symbol,
        "as_of": str(as_of),
        "horizon": int(horizon),
        "generated_at": snapshot.get("generated_at") or ver.utc_now_iso(),
        "probability": float(prob),
        "raw_probability": latest.get("raw_probability"),
        "signal": latest.get("signal"),
        "direction": latest.get("direction"),
        "close": latest.get("close"),
        "feature_version": versions.get("feature") or snap.get("feature_version"),
        "model_version": versions.get("model") or snap.get("model_version"),
        "ensemble_version": versions.get("ensemble") or snap.get("ensemble_version"),
        "calibration_method": versions.get("calibration") or snap.get("calibration_method"),
        "data_fingerprint": snapshot.get("data_fingerprint") or snapshot.get(
            "input_fingerprint"
        ),
        "actual_future_outcome": None,
        "scored": False,
        "schema_version": SCHEMA_VERSION,
    }
    try:
        if rec["raw_probability"] is not None:
            rec["raw_probability"] = float(rec["raw_probability"])
        if rec["close"] is not None:
            rec["close"] = float(rec["close"])
    except (TypeError, ValueError):
        return None
    return rec


def load_predictions(
    symbol: str | None = None, log_dir: Path | None = None
) -> list[dict[str, Any]]:
    """Read every logged prediction (one symbol, or all when None)."""
    d = Path(log_dir) if log_dir is not None else default_log_dir()
    if symbol is not None:
        paths = [_log_path(symbol, d)]
    else:
        paths = sorted(d.glob("*.jsonl")) if d.exists() else []
    out: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def record_prediction(
    snapshot: dict[str, Any],
    *,
    as_of: str,
    horizon: int,
    symbol: str | None = None,
    log_dir: Path | None = None,
) -> dict[str, Any]:
    """Append one prediction to the per-symbol JSONL log (idempotent).

    Same ``(as_of, data_fingerprint)`` is never logged twice; the second call
    is a no-op returning ``{"status": "duplicate"}``.
    """
    sym = (symbol or str((snapshot.get("symbol") or "UNKNOWN"))).upper()
    rec = _resolve_prediction_fields(snapshot, sym, as_of, horizon)
    if rec is None:
        return {"status": "skipped", "symbol": sym, "reason": "not a usable forecast"}

    d = Path(log_dir) if log_dir is not None else default_log_dir()
    for prev in load_predictions(sym, d):
        if (prev.get("as_of") == rec["as_of"]
                and prev.get("data_fingerprint") == rec.get("data_fingerprint")):
            return {
                "status": "duplicate", "symbol": sym, "as_of": rec["as_of"],
                "path": str(_log_path(sym, d)),
            }
    path = _log_path(sym, d)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError as exc:
        return {"status": "skipped", "symbol": sym,
                "reason": f"log write failed: {exc}"}
    return {"status": "recorded", "symbol": sym, "as_of": rec["as_of"],
            "path": str(path)}


# scoring -----------------------------------------------------------------------
#
# A prediction for `as_of` is scoreable once the price history reaches at
# least `as_of + horizon` trading days (data[TOTALLY BEFORE] never leaks). The
# comparison is positional (same T -> T+horizon convention as the training
# target), never a forecast of missing prices.

def _parse_day(value: Any) -> Any:
    try:
        return pd.Timestamp(value)
    except (TypeError, ValueError):
        return None


def _normalize_closes(close: pd.Series | pd.DataFrame | None) -> pd.Series | None:
    if close is None:
        return None
    if isinstance(close, pd.DataFrame):
        if "Close" not in close.columns:
            return None
        s = close["Close"].astype(float)
    else:
        s = close.astype(float)
    idx = pd.DatetimeIndex(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s.index = idx
    return s.sort_index()


def evaluate_prediction_against_closes(
    rec: dict[str, Any], close: pd.Series | pd.DataFrame | None
) -> dict[str, Any] | None:
    """Attach the realized outcome to one logged prediction (pure).

    Returns the updated record with ``scored=True`` and
    ``actual_future_outcome`` set, or None when the history does not yet reach
    ``as_of + horizon`` (never fabricates: unscored stays unscored).
    """
    series = _normalize_closes(close)
    if series is None or len(series) == 0:
        return None
    try:
        horizon = int(rec.get("horizon", 20))
    except (TypeError, ValueError):
        return None
    if horizon <= 0:
        return None
    as_of = _parse_day(rec.get("as_of"))
    if as_of is None:
        return None
    idx = pd.DatetimeIndex(series.index)
    entry_close = rec.get("close")
    pos_list = list(idx[idx >= as_of.normalize()])
    if not pos_list:
        return None
    try:
        start_pos = int(idx.get_loc(pos_list[0]))
    except (KeyError, TypeError, ValueError):
        starts = [i for i, d in enumerate(idx) if d >= as_of.normalize()]
        if not starts:
            return None
        start_pos = starts[0]
    exit_pos = start_pos + horizon
    if exit_pos >= len(series):
        return None
    entry = float(entry_close) if entry_close not in (None, "") else float(
        series.iloc[start_pos]
    )
    exit_close = float(series.iloc[exit_pos])
    if not (np.isfinite(entry) and np.isfinite(exit_close) and entry != 0):
        return None
    ret = exit_close / entry - 1.0
    outcome = {
        "entry_close": round(entry, 4),
        "exit_close": round(exit_close, 4),
        "exit_date": idx[exit_pos].isoformat(),
        "return_20d": round(float(ret), 6),
        "label_up": bool(ret > 0),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    return ver.record_outcome(dict(rec), outcome) | {"scored": True}


def score_pending_predictions(
    symbol: str | None = None,
    *,
    close_lookup: Callable[[str], pd.Series | pd.DataFrame | None] | None = None,
    log_dir: Path | None = None,
) -> dict[str, Any]:
    """Score every scoreable logged prediction and rewrite the log files.

    `close_lookup(symbol)` returns the full daily Close history for that
    symbol (injectable for tests; the CLI builds it from data_service).
    """
    d = Path(log_dir) if log_dir is not None else default_log_dir()
    if symbol is not None:
        syms = [symbol.upper()]
    else:
        syms = sorted(p.stem.upper() for p in d.glob("*.jsonl")) if d.exists() else []
    summary = {"symbols": {}, "n_scored": 0, "n_unscoreable": 0}
    for sym in syms:
        recs = load_predictions(sym, d)
        lookup = close_lookup(sym) if close_lookup is not None else None
        updated: list[dict[str, Any]] = []
        n_scored = 0
        n_wait = 0
        for rec in recs:
            if rec.get("scored"):
                updated.append(rec)
                continue
            ev = evaluate_prediction_against_closes(rec, lookup)
            if ev is None:
                updated.append(rec)
                n_wait += 1
            else:
                updated.append(ev)
                n_scored += 1
        if recs:
            path = _log_path(sym, d)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fh:
                for rec in updated:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        summary["symbols"][sym] = {
            "n_total": len(recs), "n_newly_scored": n_scored,
            "n_waiting": n_wait,
        }
        summary["n_scored"] += n_scored
        summary["n_unscoreable"] += n_wait
    return summary


def _round4(value: Any) -> Any:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    if not np.isfinite(f):
        return None
    return round(f, 4)


def scorecard_report(
    symbol: str | None = None, *, log_dir: Path | None = None
) -> dict[str, Any]:
    """Aggregate the SCORED predictions into forward-test numbers.

    Returns warnings (never conclusions) when the scored sample is too small.
    """
    recs = load_predictions(symbol, log_dir)
    scored = [r for r in recs if r.get("scored")
              and isinstance(r.get("actual_future_outcome"), dict)]
    n_total = len(recs)
    n_scored = len(scored)

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol.upper() if symbol else None,
        "n_logged": n_total,
        "n_scored": n_scored,
        "n_waiting": sum(1 for r in recs if not r.get("scored")),
        "min_scored_for_conclusion": MIN_SCORED_FOR_CONCLUSION,
        "is_sufficient_sample": bool(n_scored >= MIN_SCORED_FOR_CONCLUSION),
        "metrics": None,
        "per_symbol": {},
        "note": (
            "forward (in-time) test: each scored prediction compares the P(up) "
            "emitted on as_of against the realized T -> T+horizon label. "
            "Unscored predictions (horizon not yet elapsed) are NEVER counted."
        ),
    }
    if n_scored == 0:
        report["metrics"] = {"status": "no_scored_predictions"}
        return report

    def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        probs = np.array([float(r["probability"]) for r in rows], dtype=float)
        ups = np.array([bool(r["actual_future_outcome"]["label_up"]) for r in rows])
        rets = np.array(
            [float(r["actual_future_outcome"]["return_20d"]) for r in rows], dtype=float
        )
        hit = (probs >= 0.5) == ups
        acc = float(hit.mean())
        up_rate = float(ups.mean())
        best_base = max(up_rate, 1.0 - up_rate)
        return {
            "n": int(len(rows)),
            "accuracy": _round4(acc),
            "brier": _round4(float(np.mean((probs - ups.astype(float)) ** 2))),
            "up_rate": _round4(up_rate),
            "always_up_accuracy": _round4(up_rate),
            "best_baseline_accuracy": _round4(best_base),
            "edge_vs_best_baseline_accuracy": _round4(acc - best_base),
            "mean_signed_horizon_return": _round4(float(rets.mean())),
            "hit_rate_when_predicting_up": _round4(
                float(hit[probs >= 0.5].mean()) if (probs >= 0.5).any() else float("nan")
            ),
        }

    report["metrics"] = _metrics(scored)
    by_sym: dict[str, list[dict[str, Any]]] = {}
    for r in scored:
        by_sym.setdefault(str(r.get("symbol") or "?").upper(), []).append(r)
    for sym, rows in sorted(by_sym.items()):
        report["per_symbol"][sym] = _metrics(rows)
    if not report["is_sufficient_sample"]:
        report["warning"] = (
            f"only {n_scored} scored prediction(s); below the "
            f"{MIN_SCORED_FOR_CONCLUSION}-sample sufficiency floor — "
            "treat these numbers as a ledger preview, NOT evidence"
        )
    return report


def build_live_close_lookup() -> Callable[[str], pd.Series | None]:
    """Close-series provider over the live upstream (network boundary)."""

    def _lookup(symbol: str) -> pd.Series | None:
        try:
            from app.services import data_service as dsvc
        except Exception:
            return None
        try:
            payload = dsvc.fetch_nse_ohlcv(symbol, period="5y", interval="1d")
        except Exception as exc:  # noqa: BLE001
            logger.warning("scorecard price fetch failed for %s: %s", symbol, exc)
            return None
        if not payload.get("is_available"):
            return None
        return pl._ohlcv_from_records(payload.get("history") or [])

    return _lookup


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Forward prediction scorecard: score logged predictions "
                    "against realized prices and print the report."
    )
    ap.add_argument("--symbols", default="",
                    help="comma-separated symbols (default: everything logged)")
    args = ap.parse_args()
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    lookup = build_live_close_lookup()
    scored_any = 0
    if not syms:
        out = score_pending_predictions(close_lookup=lookup)
        scored_any = out.get("n_scored", 0)
    else:
        for sym in syms:
            out = score_pending_predictions(sym, close_lookup=lookup)
            scored_any += out.get("n_scored", 0)
    symbol = syms[0] if len(syms) == 1 else None
    rep = scorecard_report(symbol)
    print(json.dumps(rep, indent=2))
    print(f"\nscored_this_run={scored_any}")


if __name__ == "__main__":
    main()