"""Task 11 — V1 Forecast Benchmark & Failure Analysis (measurement layer).

MULTI-STOCK, REPRODUCIBLE BENCHMARK OF THE *CURRENT* V1 PRODUCTION 20-DAY
FORECAST. This is measurement only: it never changes features, models,
ensemble composition/weights, calibration choice, thresholds, targets,
transaction costs or the backtest accounting. Every symbol is evaluated by
re-running the identical production entry points on freshly built inputs:

- input construction:  ``pipeline._build_symbol_input`` (the exact path the
  live ``forecast_symbol`` uses; the forecast cache is NOT consulted so each
  run measures a fresh snapshot, but every run is tagged with the same input
  fingerprint the live path would use);
- the forecast itself: ``pipeline.forecast_frame`` with the production
  ensemble, walk-forward folds, calibration split (older 80% fit /
  newest 20% holdout), thresholds and baselines.

Per symbol the benchmark records:
  A. Discrimination — holdout ROC-AUC for the ensemble and each installed
     model, plus per-model diagnostics.
  B. Calibration    — method, Brier/ECE on the holdout, distinct calibrated
     levels and boundary-collapse share.
  C. Trading        — holdout backtest via the production equity ledger
     (n_trades, win_rate, profit factor, cum_return, max_dd, sharpe/sortino/
     calmar, no-leverage checks).
  D. Baselines      — always_up / always_down / seeded random / buy-and-hold
     / shuffled-target control on the very same holdout.

Deep, descriptive sections (all OOF-wide, never the final holdout; purely
reporting — nothing is selected or changed):
  - ``model_comparison``: ``backtest.run_backtest`` per installed model on the
    production folds;
  - ``ablation``: ``ablation.run_ablation`` leave-one-group-out on the full OOF;
  - ``failure``: probability collapse, directional asymmetry, trading
    weakness, regime-bucketed OOF AUC (``regime_trend`` / ``regime_vol``) and
    symbol-level trouble.

Cross-stock aggregation (``aggregate``): per-metric median/mean/std + min/max
and pass/fail counts across the universe. Outputs are machine-readable JSON
and a human-readable text report. The module makes NO accuracy claims: every
number is a measured quantity on an explicit out-of-sample window, and its
input snapshot (fingerprint) is recorded alongside.

Reproducibility contract on every record: symbol, as_of, generated_at,
feature/model/ensemble/calibration versions, seed, walk-forward parameters,
data fingerprint and input sources.

CLI (from backend/): ``python -m app.ml.benchmark --smoke``
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from app.config import settings
from app.ml import ablation as abl
from app.ml import backtest as bt
from app.ml import dataset as ds
from app.ml import explain
from app.ml import models as mo
from app.ml import pipeline as pl
from app.ml import versions as ver
from app.utils.validators import clean_symbol

logger = logging.getLogger("equitylens.benchmark")

BENCHMARK_VERSION = "benchmark-v1"
SCHEMA_VERSION = "v1"

# ---------------------------------------------------------------------------
# Universes (drawn from signals_service.ALL_SYMBOLS; diversified liquid NSE)
# ---------------------------------------------------------------------------

DEFAULT_UNIVERSE = [
    "HDFCBANK", "BAJFINANCE",        # Banking / Financials
    "TCS", "INFY",                   # IT
    "RELIANCE", "NTPC",              # Energy
    "ITC", "HINDUNILVR",             # FMCG
    "MARUTI", "TATAMOTORS",          # Autos
    "SUNPHARMA", "DRREDDY",          # Pharma
    "LT", "SIEMENS",                 # Industrials
    "BHARTIARTL",                    # Telecom
    "TATASTEEL", "JSWSTEEL",         # Metals
]

SMOKE_UNIVERSE = ["TCS", "RELIANCE", "HDFCBANK", "INFY"]

ENSEMBLE_COMPONENTS = tuple(pl.ENSEMBLE_MODELS)
BENCH_MODELS = tuple(mo.model_names())

# Metrics captured per hold-out backtest row / model-comparison row (all are
# defined by backtest.strategy_metrics; provenance documented in metric_groups).
METRIC_KEYS = (
    "n_trades", "win_rate", "avg_return", "profit_factor",
    "cum_return", "max_dd", "sharpe", "sortino", "calmar",
    "annualized_return", "annualized_volatility", "avg_daily_return",
    "n_days", "max_concurrent", "no_leverage_ok",
    "accuracy", "precision", "recall", "f1",
    "false_buy_pct", "false_sell_pct",
    "roc_auc", "calibration_brier", "trade_accuracy",
)

ABLATION_METRICS = ("accuracy", "f1", "win_rate", "avg_return", "cum_return",
                    "max_dd", "sharpe", "profit_factor", "n_trades")
ABLATION_DELTA_METRICS = ("accuracy", "f1", "win_rate", "avg_return",
                          "cum_return", "sharpe", "profit_factor")

# Diagnostic thresholds (used ONLY to bucket/report, never to change signals).
COLLAPSE_MAX_DISTINCT_LEVELS = 3     # fewer calibrated levels => near-constant map
COLLAPSE_MAX_PCT_NEAR_0P50 = 0.90    # >=90% calibrated probs within .05 of .50
WEAK_AUC_BOUND = 0.55                # holdout auc < 0.55 => "weak" bucket
MIN_TRADES_FOR_TRADING_EVAL = 5

_TREND_NAMES = {-1.0: "bear", 0.0: "sideways", 1.0: "bull"}
_VOL_NAMES = {-1.0: "low", 0.0: "normal", 1.0: "high"}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    symbols: list[str] = field(default_factory=lambda: list(DEFAULT_UNIVERSE))
    seed: int = 0
    fast: bool = True
    period: str = "5y"
    enrich: bool = False
    horizon: int = settings.ml_horizon
    label: str | None = None
    run_model_comparison: bool = True
    run_ablation: bool = True
    run_failure_diagnostics: bool = True
    ablation_model: str = "rf"
    output_dir: str = "ml_benchmark"
    allow_skipped: bool = True


# ---------------------------------------------------------------------------
# Input provider (offline builds = the same path the live forecast uses)
# ---------------------------------------------------------------------------

def library_input_provider(
    symbol: str, period: str = "5y", enrich: bool = False
) -> dict[str, Any]:
    """Default provider: rebuild the exact upstream inputs the live forecast
    path consumes (`pipeline._build_symbol_input`). The benchmark deliberately
    bypasses the on-disk forecast cache so every run measures a fresh input
    snapshot — but fingerprints the inputs exactly like the live path."""
    return pl._build_symbol_input(symbol, period=period, enrich=enrich)


# ---------------------------------------------------------------------------
# Per-symbol benchmark
# ---------------------------------------------------------------------------

def run_symbol_benchmark(
    symbol: str,
    cfg: BenchmarkConfig,
    input_provider: Callable[[str, str, bool], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One symbol, full benchmark record. Never fabricates: an unavailable
    symbol or an un-runnable forecast is recorded as ``ok: false`` with a
    reason."""
    t0 = time.time()
    clean = clean_symbol(symbol)
    provider = input_provider or library_input_provider
    inp = provider(clean, period=cfg.period, enrich=cfg.enrich)
    if not inp.get("ok"):
        return _fail_record(clean, cfg, inp.get("error") or "input build failed", t0)

    result = pl.forecast_frame(
        inp["features"], symbol=clean, horizon=cfg.horizon, seed=cfg.seed,
        fast=cfg.fast, data_fingerprint=inp.get("fp_hex"),
        input_meta=inp.get("input_meta"),
    )
    if not result.get("is_available", False):
        return _fail_record(clean, cfg, result.get("error") or "forecast unavailable", t0)

    return _build_symbol_record(clean, cfg, result, inp, t0)


def _fail_record(symbol: str, cfg: BenchmarkConfig, error: str, t0: float) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "ok": False,
        "error": error,
        "horizon": int(cfg.horizon),
        "seed": int(cfg.seed),
        "generated_at": ver.utc_now_iso(),
        "elapsed_s": round(time.time() - t0, 2),
    }


def _build_symbol_record(
    symbol: str,
    cfg: BenchmarkConfig,
    result: dict[str, Any],
    inp: dict[str, Any],
    t0: float,
) -> dict[str, Any]:
    hold_rows = result.get("backtest") or []
    by_model = {r.get("model"): r for r in hold_rows if r.get("model")}

    hold = _metric_row(by_model.get("ensemble"))
    per_model = {m: _metric_row(by_model.get(m)) for m in ENSEMBLE_COMPONENTS}
    baselines = {
        m: _metric_row(by_model.get(m))
        for m in ("always_up", "always_down", "random", "buy_hold", "shuffled_control")
    }

    cal = result.get("calibration") or {}
    cal_report = cal.get("report") or {}
    collapse = _collapse_signature(
        cal_report.get("n_distinct_calibrated_levels"),
        cal_report.get("pct_within_0p05_0p50"),
    )

    rec: dict[str, Any] = {
        "symbol": symbol,
        "ok": True,
        "error": None,
        "horizon": int(cfg.horizon),
        "seed": int(cfg.seed),
        "fast": bool(cfg.fast),
        "period": cfg.period,
        "enrich": bool(cfg.enrich),
        "generated_at": result.get("generated_at"),
        "elapsed_s": round(time.time() - t0, 2),
        "as_of": (result.get("latest") or {}).get("as_of"),
        "data_fingerprint": result.get("data_fingerprint"),
        "input_fingerprint": result.get("input_fingerprint"),
        "versions": result.get("versions"),
        "latest": {
            "signal": (result.get("latest") or {}).get("signal"),
            "probability": (result.get("latest") or {}).get("probability"),
            "raw_probability": (result.get("latest") or {}).get("raw_probability"),
            "as_of": (result.get("latest") or {}).get("as_of"),
        },
        "validation": {
            "oof_rows": (result.get("validation") or {}).get("oof_rows"),
            "calibration_fit_rows": (result.get("validation") or {}).get("calibration_fit_rows"),
            "holdout_rows": (result.get("validation") or {}).get("holdout_rows"),
            "oof_rows_effective_independent": (
                (result.get("validation") or {}).get("oof_rows_effective_independent")
            ),
            "holdout_rows_effective_independent": (
                (result.get("validation") or {}).get("holdout_rows_effective_independent")
            ),
            "final_holdout_never_used_for_fitting": (
                (result.get("validation") or {}).get("final_holdout_never_used_for_fitting")
            ),
            "final_holdout_used_for": (result.get("validation") or {}).get("final_holdout_used_for"),
        },
        # Honest measurement block (W1): AUC is a RANKING metric, so the record
        # also carries plain classifier accuracy on the same labelled rows and
        # the trivial always-up / always-down baselines it must beat.
        "discrimination": result.get("discrimination"),
        "calibration": {
            "method": cal.get("method"),
            "n": cal.get("n"),
            "n_holdout": cal.get("n_holdout"),
            "brier": cal.get("brier"),
            "brier_calibrated": cal.get("brier_calibrated"),
            "ece": cal.get("ece"),
            "ece_calibrated": cal.get("ece_calibrated"),
            "report": cal_report,
        },
        "probability_support": collapse,
        "holdout": {
            "ensemble": hold,
            "per_model": per_model,
            "baselines": baselines,
            "evaluation_scope": (
                "newest 20% of out-of-sample walk-forward rows (final holdout only)"
            ),
        },
        "model_comparison": _model_comparison(inp["features"], cfg) if cfg.run_model_comparison else None,
        "ablation": _ablation_section(inp["features"], cfg) if cfg.run_ablation else None,
        "failure": _failure_section(inp["features"], cfg, hold, baselines) if cfg.run_failure_diagnostics else None,
        "feature_groups_present": sorted(abl.default_group_columns(list(inp["features"].columns))),
    }
    return _sanitize(rec)


# ---------------------------------------------------------------------------
# Row / signature helpers
# ---------------------------------------------------------------------------

def _metric_row(row: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in METRIC_KEYS:
        v = row.get(k) if row else None
        out[k] = None if (isinstance(v, float) and np.isnan(v)) else v
    return out


def _collapse_signature(n_levels: Any, pct_near_0p50: Any) -> dict[str, Any]:
    reasons: list[str] = []
    if n_levels is None:
        reasons.append("n_distinct_calibrated_levels unknown")
    elif isinstance(n_levels, (int, float, np.number)) and n_levels <= COLLAPSE_MAX_DISTINCT_LEVELS:
        reasons.append(f"only {int(n_levels)} distinct calibrated levels")
    if pct_near_0p50 is not None and isinstance(pct_near_0p50, (int, float, np.number)) \
            and pct_near_0p50 >= COLLAPSE_MAX_PCT_NEAR_0P50:
        reasons.append(f"{float(pct_near_0p50):.0%} of calibrated probs within +/-0.05 of 0.50")
    return {
        "collapse_flag": bool(reasons),
        "collapse_reason": "; ".join(reasons) if reasons else None,
        "n_distinct_calibrated_levels": int(n_levels) if n_levels is not None else None,
        "pct_within_0p05_0p50": float(pct_near_0p50) if pct_near_0p50 is not None else None,
    }


# ---------------------------------------------------------------------------
# Deep sections (all OOF-wide — never the final holdout)
# ---------------------------------------------------------------------------

def _walk_forward_grid(cfg: BenchmarkConfig) -> dict[str, Any]:
    return {
        "test_size": 40 if cfg.fast else pl.OOF_TEST_SIZE,
        "step": 90 if cfg.fast else pl.OOF_STEP,
        "min_train": pl.OOF_MIN_TRAIN,
        "embargo": int(cfg.horizon),
        "source": "pipeline.forecast_frame walk-forward engine (unchanged)",
    }


def _model_comparison(feats: pd.DataFrame, cfg: BenchmarkConfig) -> dict[str, Any] | None:
    # Sanitise +/-inf exactly like forecast_frame's feature cleaning so the
    # measured matrix matches what the production ensemble actually consumes.
    work = feats.replace([np.inf, -np.inf], np.nan)
    by_model: dict[str, Any] = {}
    for m in mo.model_names():
        res = bt.run_backtest(
            work, model=m, horizon=cfg.horizon,
            test_size=_walk_forward_grid(cfg)["test_size"],
            step=_walk_forward_grid(cfg)["step"],
            min_train=pl.OOF_MIN_TRAIN, embargo=cfg.horizon, seed=cfg.seed,
        )
        by_model[m] = _metric_row(res.get("metrics"))
    return {
        "models": list(mo.model_names()),
        "by_model": by_model,
        "walk_forward": _walk_forward_grid(cfg),
        "evaluation_scope": "full OOF walk-forward with production folds (not the final holdout)",
    }


def _ablation_section(feats: pd.DataFrame, cfg: BenchmarkConfig) -> dict[str, Any] | None:
    grid = _walk_forward_grid(cfg)
    work = feats.replace([np.inf, -np.inf], np.nan)
    try:
        tbl = abl.run_ablation(
            work, model=cfg.ablation_model, horizon=cfg.horizon,
            test_size=grid["test_size"], step=grid["step"],
            min_train=pl.OOF_MIN_TRAIN, embargo=cfg.horizon,
            seed=cfg.seed, close_col="Close",
        )
    except Exception as exc:  # noqa: BLE001 — descriptive failure, never fatal
        return {"error": str(exc), "rows": None, "delta_vs_full": None,
                "evaluation_scope": "full OOF walk-forward"}
    rows: dict[str, Any] = {}
    for variant in tbl.index:
        rows[str(variant)] = {}
        for k in ABLATION_METRICS:
            v = tbl.loc[variant, k]
            rows[str(variant)][k] = None if (isinstance(v, float) and np.isnan(v)) else v
    full = rows.get("full")
    deltas: dict[str, Any] = {}
    for variant, r in rows.items():
        if variant == "full" or full is None:
            continue
        deltas[variant] = {}
        for k in ABLATION_DELTA_METRICS:
            a, b = r.get(k), full.get(k)
            deltas[variant][k] = None if (a is None or b is None) else round(float(a) - float(b), 4)
    return {
        "model": cfg.ablation_model,
        "rows": rows,
        "delta_vs_full": deltas,
        "walk_forward": grid,
        "evaluation_scope": (
            "full OOF walk-forward; ablations are descriptive only and never "
            "used to select/change production features"
        ),
    }


def _failure_section(
    feats: pd.DataFrame, cfg: BenchmarkConfig, hold: dict[str, Any], baselines: dict[str, Any]
) -> dict[str, Any]:
    auc = hold.get("roc_auc")
    auc_bucket: str | None = None
    if auc is not None:
        auc_bucket = (
            "chance_or_worse" if auc <= 0.5
            else ("weak" if auc < WEAK_AUC_BOUND else "useful")
        )
    fb = hold.get("false_buy_pct")
    fs = hold.get("false_sell_pct")
    asymmetry = None
    if fb is not None and fs is not None:
        asymmetry = float(fb) - float(fs)
    cum = hold.get("cum_return")
    bh = (baselines.get("buy_hold") or {}).get("cum_return")
    au = (baselines.get("always_up") or {}).get("cum_return")
    sh = (baselines.get("shuffled_control") or {}).get("roc_auc")
    n_trades = hold.get("n_trades")
    pf = hold.get("profit_factor")
    return {
        "probability_collapse": {
            "flag": _collapse_signature(
                hold.get("n_distinct_calibrated_levels"), hold.get("pct_within_0p05_0p50")
            )["collapse_flag"],
        },
        "directional": {
            "roc_auc": auc,
            "auc_bucket": auc_bucket,
            "recall": hold.get("recall"),
            "precision": hold.get("precision"),
            "false_buy_pct": fb,
            "false_sell_pct": fs,
            "false_buy_vs_false_sell_delta": asymmetry,
        },
        "trading": {
            "n_trades": n_trades,
            "profit_factor": pf,
            "cum_return": cum,
            "max_dd": hold.get("max_dd"),
            "sharpe": hold.get("sharpe"),
            "win_rate": hold.get("win_rate"),
            "no_leverage_ok": hold.get("no_leverage_ok"),
            "flag_few_trades": (n_trades or 0) < MIN_TRADES_FOR_TRADING_EVAL,
            "flag_negative_cum_return": cum is not None and cum < 0,
            "flag_profit_factor_lt_1": pf is not None and pf < 1,
        },
        "vs_baselines": {
            "beats_buy_hold_cum_return": None if (cum is None or bh is None) else bool(cum > bh),
            "beats_always_up_cum_return": None if (cum is None or au is None) else bool(cum > au),
            "shuffled_control_auc": sh,
            "auc_over_shuffled_control": None if (auc is None or sh is None) else float(auc) - float(sh),
        },
        "by_regime": _regime_bucket_diagnostics(feats, cfg),
        "evaluation_scope": "regime buckets measured on the full OOF walk-forward (not the holdout)",
    }


def _regime_bucket_diagnostics(feats: pd.DataFrame, cfg: BenchmarkConfig) -> dict[str, Any]:
    h = cfg.horizon
    out = ds.add_target(feats, horizon=h)
    if len(out) < pl.OOF_MIN_TRAIN + h:
        return {"error": "insufficient rows for regime diagnostics", "buckets": {}}
    if "regime_trend" not in out.columns and "regime_vol" not in out.columns:
        return {"error": "regime features absent (market context not provided)", "buckets": {}}

    ret = out[f"target_ret_{h}d"]
    y = out[f"target_up_{h}d"]
    index = pd.DatetimeIndex(out.index)
    feat_cols = pl._default_forecast_cols(out, h)
    keep = [c for c in feat_cols if out[c].notna().sum() > 0]
    X = out[keep].replace([np.inf, -np.inf], np.nan)

    grid = _walk_forward_grid(cfg)
    folds = bt.purged_walk_forward_splits(
        index, test_size=grid["test_size"], step=grid["step"],
        min_train=pl.OOF_MIN_TRAIN, embargo=h,
    )
    rows: list[dict[str, Any]] = []
    for fi, (tr, te) in enumerate(folds):
        combined, _ = pl._fold_ensemble_probs(
            X, y, tr, te, list(pl.ENSEMBLE_MODELS), seed=cfg.seed + fi
        )
        for i, d in enumerate(te):
            cv = combined[i]
            if np.isnan(cv):
                continue
            rows.append({
                "date": d,
                "prob": float(cv),
                "y": float(y.loc[d]) if pd.notna(y.loc[d]) else None,
                "ret": float(ret.loc[d]) if pd.notna(ret.loc[d]) else None,
            })
    if len(rows) < 20:
        return {"error": "too few OOF rows for regime diagnostics", "buckets": {}}

    rdf = pd.DataFrame(rows).set_index("date").sort_index()
    for rc in ("regime_trend", "regime_vol"):
        if rc in out.columns:
            rdf[rc] = out[rc].reindex(rdf.index)

    buckets: dict[str, Any] = {}
    for rc in ("regime_trend", "regime_vol"):
        if rc not in rdf.columns:
            continue
        bmap: dict[str, Any] = {}
        for b in (-1.0, 0.0, 1.0):
            label = _regime_label(rc, b)
            sub = rdf[rdf[rc].astype(float).eq(b)]
            bmap[label] = _bucket_stats(sub)
        unknown = rdf[rdf[rc].isna() | (~rdf[rc].astype(float).isin((-1.0, 0.0, 1.0)))]
        if len(unknown):
            bmap["unknown"] = _bucket_stats(unknown)
        bucket_meta = {
            "regime_trend": {"levels": "1:bull 0:sideways -1:bear"},
            "regime_vol": {"levels": "1:high 0:normal -1:low"},
        }[rc]
        buckets[rc] = {"semantics": bucket_meta["levels"], "buckets": bmap}
    return {"error": None, "buckets": buckets}


def _regime_label(rc: str, b: float) -> str:
    names = _TREND_NAMES if rc == "regime_trend" else _VOL_NAMES
    return f"{int(b)}:{names[b]}"


def _bucket_stats(sub: pd.DataFrame) -> dict[str, Any]:
    sub = sub[sub["y"].notna() & sub["prob"].notna()]
    if len(sub) < 5:
        return {"n": int(len(sub)), "roc_auc": None, "up_rate": None,
                "prob_mean": None, "prob_std": None}
    yv = sub["y"].to_numpy(dtype=float).astype(int)
    pv = sub["prob"].to_numpy(dtype=float)
    return {
        "n": int(len(sub)),
        "roc_auc": round(float(explain._auc(yv, pv)), 4),
        "up_rate": round(float(yv.mean()), 4),
        "prob_mean": round(float(pv.mean()), 4),
        "prob_std": round(float(pv.std(ddof=0)), 4),
    }


# ---------------------------------------------------------------------------
# Cross-stock aggregation
# ---------------------------------------------------------------------------

def _summarize(values: list[Any]) -> dict[str, Any]:
    vals: list[float] = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, (bool, np.bool_)):
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(f) or math.isinf(f):
            continue
        vals.append(f)
    if not vals:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None, "n": 0}
    a = np.asarray(vals, dtype=float)
    return {
        "mean": round(float(a.mean()), 4),
        "median": round(float(np.median(a)), 4),
        "std": round(float(a.std(ddof=0)), 4),
        "min": round(float(a.min()), 4),
        "max": round(float(a.max()), 4),
        "n": int(len(a)),
    }


def _aggregate(records: list[dict[str, Any]], cfg: BenchmarkConfig) -> dict[str, Any]:
    ok = [r for r in records if r.get("ok")]
    n = len(ok)

    def ens(r: dict[str, Any]) -> dict[str, Any]:
        return ((r.get("holdout") or {}).get("ensemble") or {})

    def per_model(r: dict[str, Any], m: str) -> dict[str, Any]:
        return ((r.get("holdout") or {}).get("per_model") or {}).get(m, {})

    def base(r: dict[str, Any], nm: str) -> dict[str, Any]:
        return ((r.get("holdout") or {}).get("baselines") or {}).get(nm, {})

    counts: dict[str, Any] = {
        "symbols_requested": int(len(records)),
        "symbols_ok": n,
        "symbols_failed": int(len(records) - n),
    }
    metrics: dict[str, Any] = {}
    per_model_agg: dict[str, Any] = {}
    baseline_agg: dict[str, Any] = {}
    model_comparison_agg: dict[str, Any] = {}
    regime_agg: dict[str, Any] = {}
    ablation_agg: dict[str, Any] = {}
    failure_counts: dict[str, Any] = {}
    symbol_level: list[dict[str, Any]] = []

    if n:
        aucs = [ens(r).get("roc_auc") for r in ok]
        cums = [ens(r).get("cum_return") for r in ok]
        pfs = [ens(r).get("profit_factor") for r in ok]
        # Honest accuracy / baseline-relative aggregation (W1).
        accs_cal = [
            (r.get("discrimination") or {}).get("holdout_accuracy_calibrated")
            for r in ok
        ]
        edges = [
            (r.get("discrimination") or {}).get("holdout_edge_vs_best_baseline_accuracy")
            for r in ok
        ]
        always_up_accs = [
            ((r.get("discrimination") or {}).get("holdout_baseline_accuracy") or {})
            .get("always_up")
            for r in ok
        ]
        counts["holdout_auc_gt_0p50"] = sum(1 for a in aucs if a is not None and a > 0.5)
        counts["holdout_auc_gt_0p55"] = sum(1 for a in aucs if a is not None and a > WEAK_AUC_BOUND)
        counts["holdout_auc_le_0p50"] = sum(1 for a in aucs if a is not None and a <= 0.5)
        counts["beats_best_baseline_accuracy"] = sum(
            1 for e in edges if e is not None and e > 0
        )
        counts["worse_than_best_baseline_accuracy"] = sum(
            1 for e in edges if e is not None and e <= 0
        )
        counts["beats_buy_hold_cum_return"] = sum(
            1 for r in ok
            if (lambda c, b: c is not None and b is not None and c > b)(
                ens(r).get("cum_return"), base(r, "buy_hold").get("cum_return"))
        )
        counts["beats_always_up_cum_return"] = sum(
            1 for r in ok
            if (lambda c, a: c is not None and a is not None and c > a)(
                ens(r).get("cum_return"), base(r, "always_up").get("cum_return"))
        )
        counts["profit_factor_ge_1"] = sum(1 for p in pfs if p is not None and p >= 1)
        counts["probability_collapse_flags"] = sum(
            1 for r in ok if (r.get("probability_support") or {}).get("collapse_flag")
        )
        counts["few_trades_lt_5"] = sum(
            1 for r in ok if (ens(r).get("n_trades") or 0) < MIN_TRADES_FOR_TRADING_EVAL
        )
        counts["negative_cum_return"] = sum(1 for c in cums if c is not None and c < 0)

        metrics = {
            "holdout_roc_auc": _summarize(aucs),
            "holdout_classification_accuracy": _summarize(accs_cal),
            "holdout_always_up_baseline_accuracy": _summarize(always_up_accs),
            "holdout_edge_vs_best_baseline_accuracy": _summarize(edges),
            "holdout_calibration_brier_raw": _summarize(
                [(r.get("calibration") or {}).get("brier") for r in ok]),
            "holdout_calibration_brier_calibrated": _summarize(
                [(r.get("calibration") or {}).get("brier_calibrated") for r in ok]),
            "holdout_calibration_ece_raw": _summarize(
                [(r.get("calibration") or {}).get("ece") for r in ok]),
            "holdout_calibration_ece_calibrated": _summarize(
                [(r.get("calibration") or {}).get("ece_calibrated") for r in ok]),
            "n_trades": _summarize([ens(r).get("n_trades") for r in ok]),
            "win_rate": _summarize([ens(r).get("win_rate") for r in ok]),
            "profit_factor": _summarize(pfs),
            "cum_return": _summarize(cums),
            "max_dd": _summarize([ens(r).get("max_dd") for r in ok]),
            "sharpe": _summarize([ens(r).get("sharpe") for r in ok]),
            "sortino": _summarize([ens(r).get("sortino") for r in ok]),
            "calmar": _summarize([ens(r).get("calmar") for r in ok]),
            "n_distinct_calibrated_levels": _summarize(
                [(r.get("probability_support") or {}).get("n_distinct_calibrated_levels") for r in ok]),
            "pct_within_0p05_0p50": _summarize(
                [(r.get("probability_support") or {}).get("pct_within_0p05_0p50") for r in ok]),
        }

        per_model_metrics = ("roc_auc", "cum_return", "win_rate", "profit_factor", "n_trades", "max_dd")
        for m in ENSEMBLE_COMPONENTS:
            per_model_agg[m] = {
                k: _summarize([per_model(r, m).get(k) for r in ok])
                for k in per_model_metrics
            }

        for nm in ("always_up", "always_down", "random", "buy_hold", "shuffled_control"):
            baseline_agg[nm] = {
                k: _summarize([base(r, nm).get(k) for r in ok])
                for k in ("roc_auc", "cum_return", "win_rate", "profit_factor", "n_trades")
            }

        for m in BENCH_MODELS:
            vals = []
            for r in ok:
                mc = r.get("model_comparison") or {}
                row = (mc.get("by_model") or {}).get(m) or {}
                if row:
                    vals.append(row)
            model_comparison_agg[m] = {
                k: _summarize([row.get(k) for row in vals])
                for k in ("roc_auc", "cum_return", "win_rate", "profit_factor", "n_trades")
            }

        regime_agg = _aggregate_regimes(ok)

        abl_rows: dict[str, list[dict[str, Any]]] = {}
        for r in ok:
            ab = r.get("ablation") or {}
            for variant, row in (ab.get("rows") or {}).items():
                abl_rows.setdefault(str(variant), []).append(row)
        for variant, rows in sorted(abl_rows.items()):
            ablation_agg[variant] = {
                k: _summarize([row.get(k) for row in rows])
                for k in ABLATION_DELTA_METRICS
            }

        failure_counts = {
            "chance_or_worse_auc": sum(
                1 for r in ok
                if (lambda s: s == "chance_or_worse")(
                    ((r.get("failure") or {}).get("directional") or {}).get("auc_bucket"))
            ),
            "weak_auc": sum(
                1 for r in ok
                if ((r.get("failure") or {}).get("directional") or {}).get("auc_bucket") == "weak"
            ),
            "useful_auc": sum(
                1 for r in ok
                if ((r.get("failure") or {}).get("directional") or {}).get("auc_bucket") == "useful"
            ),
            "profit_factor_lt_1": sum(
                1 for r in ok
                if ((r.get("failure") or {}).get("trading") or {}).get("flag_profit_factor_lt_1")
            ),
            "negative_cum_return": sum(
                1 for r in ok
                if ((r.get("failure") or {}).get("trading") or {}).get("flag_negative_cum_return")
            ),
            "few_trades": sum(
                1 for r in ok
                if ((r.get("failure") or {}).get("trading") or {}).get("flag_few_trades")
            ),
            "no_leverage_ok": sum(
                1 for r in ok if ens(r).get("no_leverage_ok") == "yes"
            ),
            # Honest accuracy failure counts (W1): how many symbols fail to beat
            # the trivial constant (a forecast with negative edge is worse
            # than a constant on its own holdout).
            "beats_best_baseline_accuracy": sum(
                1 for r in ok
                if ((r.get("discrimination") or {})
                    .get("holdout_edge_vs_best_baseline_accuracy") or 0.0) > 0.0
            ),
            "worse_than_best_baseline_accuracy": sum(
                1 for r in ok
                if ((r.get("discrimination") or {})
                    .get("holdout_edge_vs_best_baseline_accuracy") or 0.0) <= 0.0
            ),
        }

        symbol_level = sorted(
            (
                {
                    "symbol": r["symbol"],
                    "as_of": r.get("as_of"),
                    "holdout_roc_auc": ens(r).get("roc_auc"),
                    "holdout_accuracy_calibrated": (
                        (r.get("discrimination") or {})
                        .get("holdout_accuracy_calibrated")
                    ),
                    "holdout_best_baseline_accuracy": (
                        (r.get("discrimination") or {})
                        .get("holdout_best_baseline_accuracy")
                    ),
                    "holdout_edge_vs_best_baseline_accuracy": (
                        (r.get("discrimination") or {})
                        .get("holdout_edge_vs_best_baseline_accuracy")
                    ),
                    "holdout_effective_independent_windows": (
                        ((r.get("discrimination") or {})
                         .get("effective_independent_windows") or {})
                        .get("holdout")
                    ),
                    "cum_return": ens(r).get("cum_return"),
                    "profit_factor": ens(r).get("profit_factor"),
                    "n_trades": ens(r).get("n_trades"),
                    "collapse_flag": (r.get("probability_support") or {}).get("collapse_flag"),
                }
                for r in ok
            ),
            key=lambda d: (d.get("holdout_roc_auc") is None, d.get("holdout_roc_auc")),
        )

    return {
        "counts": counts,
        "metrics": metrics,
        "per_model_holdout": per_model_agg,
        "baselines": baseline_agg,
        "model_comparison_oof": model_comparison_agg,
        "ablation": ablation_agg,
        "failure": {
            "counts": failure_counts,
            "regime": regime_agg,
            "symbol_level": symbol_level,
        },
    }


def _aggregate_regimes(ok: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for rc in ("regime_trend", "regime_vol"):
        label_to_aucs: dict[str, list[Any]] = {}
        for r in ok:
            buckets = (((r.get("failure") or {}).get("by_regime") or {}).get("buckets") or {}).get(rc) or {}
            for label, stats in (buckets.get("buckets") or {}).items():
                label_to_aucs.setdefault(label, []).append(stats.get("roc_auc"))
        out[rc] = {
            label: _summarize(aucs)
            for label, aucs in label_to_aucs.items()
        }
    return out


# ---------------------------------------------------------------------------
# Issues / warnings
# ---------------------------------------------------------------------------

def _data_quality_issues(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for r in records:
        if not r.get("ok"):
            issues.append({"symbol": r["symbol"], "issue": "unavailable", "reason": r.get("error")})
            continue
        cal = r.get("calibration") or {}
        if cal.get("method") == "uncalibrated":
            issues.append({
                "symbol": r["symbol"],
                "issue": "uncalibrated",
                "reason": (cal.get("report") or {}).get("selection_reason"),
            })
        if (r.get("probability_support") or {}).get("collapse_flag"):
            issues.append({
                "symbol": r["symbol"],
                "issue": "probability_collapse",
                "reason": (r.get("probability_support") or {}).get("collapse_reason"),
            })
        bh = ((r.get("holdout") or {}).get("baselines") or {}).get("buy_hold") or {}
        if bh.get("cum_return") is None:
            issues.append({
                "symbol": r["symbol"],
                "issue": "buy_hold_baseline_missing",
                "reason": "buy-and-hold metrics unavailable on the holdout",
            })
    return issues


def _warnings(cfg: BenchmarkConfig, records: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    if cfg.horizon != settings.ml_horizon:
        warnings.append(
            f"horizon overridden to {cfg.horizon}d (production default is "
            f"{settings.ml_horizon}d) — results are NOT the production contract"
        )
    if cfg.enrich:
        warnings.append(
            "enrich=True pulls news/options/macro inputs the default benchmark "
            "skips; fingerprints still record exactly what was consumed"
        )
    ok = [r for r in records if r.get("ok")]
    if ok and not any((r.get("holdout") or {}).get("ensemble", {}).get("roc_auc") is not None for r in ok):
        warnings.append("no ensemble ROC-AUC produced for any symbol (degraded run)")
    return warnings


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_benchmark(
    cfg: BenchmarkConfig,
    input_provider: Callable[[str, str, bool], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    records: list[dict[str, Any]] = []
    for i, sym in enumerate(cfg.symbols, start=1):
        rec = run_symbol_benchmark(sym, cfg, input_provider=input_provider)
        records.append(rec)
        logger.info(
            "benchmark [%d/%d] %s ok=%s elapsed=%.1fs",
            i, len(cfg.symbols), sym, rec.get("ok"), rec.get("elapsed_s"),
        )

    if not cfg.allow_skipped:
        failed = [r for r in records if not r.get("ok")]
        if failed:
            raise RuntimeError(
                f"benchmark aborted (allow_skipped=False); unavailable symbols: "
                + ", ".join(f"{r['symbol']}: {r.get('error')}" for r in failed)
            )

    aggregate = _aggregate(records, cfg)
    issues = _data_quality_issues(records)
    warnings = _warnings(cfg, records)

    return _sanitize({
        "benchmark_version": BENCHMARK_VERSION,
        "schema_version": SCHEMA_VERSION,
        "generated_at": ver.utc_now_iso(),
        "elapsed_total_s": round(time.time() - t0, 2),
        "run_metadata": {
            "label": cfg.label,
            "seed": int(cfg.seed),
            "fast": bool(cfg.fast),
            "period": cfg.period,
            "enrich": bool(cfg.enrich),
            "horizon": int(cfg.horizon),
            "ablation_model": cfg.ablation_model,
            "deep_sections": {
                "model_comparison": bool(cfg.run_model_comparison),
                "ablation": bool(cfg.run_ablation),
                "failure_diagnostics": bool(cfg.run_failure_diagnostics),
            },
            "production_contract": "production pipeline (forecast_frame with the "
                                   "locked ensemble, weights, calibration split, "
                                   "thresholds, targets, costs and backtest "
                                   "accounting are used unchanged)",
            "versions": {
                "feature": settings.ml_feature_version,
                "model": settings.ml_model_version,
                "ensemble": settings.ml_ensemble_version,
                "benchmark": BENCHMARK_VERSION,
            },
            "walk_forward": _walk_forward_grid(cfg),
            "precision": "floats rounded to 4dp; NaN -> null",
            "no_accuracy_claims": "all numbers are measured quantities on explicit "
                                  "out-of-sample windows, never guarantees",
        },
        "universe": {
            "requested": [clean_symbol(s) for s in cfg.symbols],
            "processed": [r["symbol"] for r in records if r.get("ok")],
            "skipped": [
                {"symbol": r["symbol"], "reason": r.get("error")}
                for r in records if not r.get("ok")
            ],
            "n_requested": len(cfg.symbols),
            "n_ok": aggregate["counts"]["symbols_ok"],
            "n_skipped": aggregate["counts"]["symbols_failed"],
        },
        "aggregate": aggregate,
        "per_symbol": records,
        "data_quality_issues": issues,
        "warnings": warnings,
    })


# ---------------------------------------------------------------------------
# Persistence + human-readable report
# ---------------------------------------------------------------------------

def _label_fragment(result: dict[str, Any]) -> str:
    syms = (result.get("universe") or {}).get("requested") or []
    if set(syms) == set(SMOKE_UNIVERSE):
        return "smoke"
    return f"{len(syms)}syms"


def save_benchmark(result: dict[str, Any], output_dir: str = "ml_benchmark",
                   label: str | None = None) -> dict[str, str]:
    d = Path(output_dir)
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = label or _label_fragment(result)
    stem = f"benchmark_{tag}"
    json_path = d / f"{stem}_{ts}.json"
    txt_path = d / f"{stem}_{ts}.txt"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    txt_path.write_text(render_report(result), encoding="utf-8")
    return {"json": str(json_path), "text": str(txt_path)}


def load_benchmark(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt(v, dflt: str = "n/a") -> str:
    if v is None:
        return dflt
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _pct(v) -> str:
    return "n/a" if v is None else f"{float(v) * 100:.2f}"


def render_report(result: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    meta = result.get("run_metadata") or {}
    uni = result.get("universe") or {}
    agg = result.get("aggregate") or {}
    counts = agg.get("counts") or {}
    metrics = agg.get("metrics") or {}
    issues = result.get("data_quality_issues") or []

    add("=" * 78)
    add("V1 FORECAST BENCHMARK — MEASUREMENT ONLY (no production changes)")
    add("=" * 78)
    add(f"benchmark_version : {result.get('benchmark_version')}")
    add(f"schema_version    : {result.get('schema_version')}")
    add(f"generated_at      : {result.get('generated_at')}")
    add(f"elapsed_total_s   : {result.get('elapsed_total_s')}")
    add(f"label             : {meta.get('label')}")
    add(f"seed / fast       : {meta.get('seed')} / {meta.get('fast')}")
    add(f"horizon / period  : {meta.get('horizon')}d / {meta.get('period')}")
    add(f"versions          : feature={meta.get('versions', {}).get('feature')} "
        f"model={meta.get('versions', {}).get('model')} "
        f"ensemble={meta.get('versions', {}).get('ensemble')}")
    add(f"walk-forward      : test_size={meta.get('walk_forward', {}).get('test_size')} "
        f"step={meta.get('walk_forward', {}).get('step')} "
        f"min_train={meta.get('walk_forward', {}).get('min_train')} "
        f"embargo={meta.get('walk_forward', {}).get('embargo')}")
    add("")

    add(f"UNIVERSE: {uni.get('n_requested')} requested -> {uni.get('n_ok')} ok, "
        f"{uni.get('n_skipped')} skipped")
    if uni.get("skipped"):
        add("  skipped:")
        for s in uni["skipped"]:
            add(f"    - {s['symbol']}: {s['reason']}")
    add("")

    add("PER-SYMBOL HOLD-OUT (ensemble, newest 20% OOS rows)")
    add("-" * 90)
    add(f"{'SYMBOL':<10}{'as_of':<12}{'AUC':>8}{'BrierCal':>11}{'cumRet%':>9}"
        f"{'PF':>8}{'nTrd':>8}{'maxDD%':>9}{'collapse':>9}")
    for r in (result.get("per_symbol") or []):
        if not r.get("ok"):
            add(f"{r['symbol']:<10}  FAILED: {r.get('error')}")
            continue
        ens = ((r.get("holdout") or {}).get("ensemble") or {})
        probsup = r.get("probability_support") or {}
        disc = r.get("discrimination") or {}
        add(f"{r['symbol']:<10}{str(r.get('as_of')):<12}"
            f"{_fmt(ens.get('roc_auc')):>8}"
            f"{_fmt((r.get('calibration') or {}).get('brier_calibrated')):>11}"
            f"{_pct(ens.get('cum_return')):>9}"
            f"{_fmt(ens.get('profit_factor')):>8}"
            f"{_fmt(ens.get('n_trades')):>8}"
            f"{_pct(ens.get('max_dd')):>9}"
            f"{('YES' if probsup.get('collapse_flag') else '-'):>9}")
        # Honest per-symbol accuracy vs the trivial constant (W1). This is the
        # number a user actually cares about; AUC alone must not be read as it.
        add(f"{'':<10}{'acc':<12}"
            f"{_fmt(disc.get('holdout_accuracy_calibrated')):>8}"
            f"{('(baseline ' + _fmt(disc.get('holdout_best_baseline_accuracy')) + ','):>23}"
            f"{('edge ' + _fmt(disc.get('holdout_edge_vs_best_baseline_accuracy')) + ')'):>12}")
    add("")

    add("AGGREGATE COUNTS")
    add("-" * 78)
    for k, v in counts.items():
        add(f"  {k:<34}: {v}")
    add("")

    add("AGGREGATE METRICS (hold-out ensemble, across symbols)")
    add("-" * 78)
    for k, v in metrics.items():
        add(f"  {k:<38} n={v.get('n')}  median={_fmt(v.get('median'))}  "
            f"mean={_fmt(v.get('mean'))}  std={_fmt(v.get('std'))}  "
            f"min={_fmt(v.get('min'))}  max={_fmt(v.get('max'))}")
    add("")

    per_model = agg.get("per_model_holdout") or {}
    add("HOLD-OUT PER-MODEL (median across symbols)")
    add("-" * 78)
    add(f"{'MODEL':<10}{'AUC':>8}{'cumRet%':>9}{'PF':>7}{'winRate':>9}{'nTrd':>6}")
    for m, row in per_model.items():
        add(f"{m:<10}{_fmt(row.get('roc_auc', {}).get('median')):>8}"
            f"{_pct(row.get('cum_return', {}).get('median')):>9}"
            f"{_fmt(row.get('profit_factor', {}).get('median')):>7}"
            f"{_fmt(row.get('win_rate', {}).get('median')):>9}"
            f"{_fmt(row.get('n_trades', {}).get('median')):>6}")
    add("")

    baselines = agg.get("baselines") or {}
    add("BASELINES ON THE SAME HOLD-OUT (median across symbols)")
    add("-" * 78)
    add(f"{'BASELINE':<20}{'cumRet%':>9}{'AUC':>8}{'nTrd':>6}")
    for nm, row in baselines.items():
        add(f"{nm:<20}"
            f"{_pct(row.get('cum_return', {}).get('median')):>9}"
            f"{_fmt(row.get('roc_auc', {}).get('median')):>8}"
            f"{_fmt(row.get('n_trades', {}).get('median')):>6}")
    add("")

    mco = agg.get("model_comparison_oof") or {}
    add("MODEL COMPARISON (full-OOF walk-forward; median across symbols)")
    add("-" * 78)
    add(f"{'MODEL':<10}{'AUC':>8}{'cumRet%':>9}{'PF':>7}{'winRate':>9}{'nTrd':>6}")
    for m, row in mco.items():
        add(f"{m:<10}{_fmt(row.get('roc_auc', {}).get('median')):>8}"
            f"{_pct(row.get('cum_return', {}).get('median')):>9}"
            f"{_fmt(row.get('profit_factor', {}).get('median')):>7}"
            f"{_fmt(row.get('win_rate', {}).get('median')):>9}"
            f"{_fmt(row.get('n_trades', {}).get('median')):>6}")
    add("")

    ablation = agg.get("ablation") or {}
    if ablation:
        add("FEATURE-GROUP ABLATION (rf, full-OOF; median delta vs full across symbols)")
        add("-" * 78)
        for variant, row in sorted(ablation.items()):
            d = ", ".join(
                f"{k}={_fmt(row.get(k, {}).get('median'))}"
                for k in ABLATION_DELTA_METRICS
            )
            add(f"  {variant:<14}: {d}")

    reg = (agg.get("failure") or {}).get("regime") or {}
    if reg:
        add("")
        add("REGIME-BUCKETED OOF AUC (median across symbols)")
        add("-" * 78)
        for rc, labels in reg.items():
            add(f"  {rc}:")
            for label, st in labels.items():
                add(f"    {label:<12} n_syms={st.get('n', 0)}  "
                    f"median_auc={_fmt(st.get('median'))}")

    add("")
    add("FAILURE DIAGNOSTIC COUNTS")
    add("-" * 78)
    fc = (agg.get("failure") or {}).get("counts") or {}
    for k, v in fc.items():
        add(f"  {k:<34}: {v}")
    add("")

    weak = (agg.get("failure") or {}).get("symbol_level") or []
    if weak:
        add("SYMBOL LEVEL (sorted by hold-out AUC, weakest first)")
        add("-" * 78)
        add(f"{'SYMBOL':<10}{'AUC':>8}{'cumRet%':>9}{'PF':>7}{'nTrd':>6}{'collapse':>9}")
        for w in weak:
            add(f"{w.get('symbol'):<10}{_fmt(w.get('holdout_roc_auc')):>8}"
                f"{_pct(w.get('cum_return')):>9}"
                f"{_fmt(w.get('profit_factor')):>7}{_fmt(w.get('n_trades')):>6}"
                f"{('YES' if w.get('collapse_flag') else '-'):>9}")
    add("")

    add("DATA QUALITY ISSUES")
    add("-" * 78)
    if issues:
        for it in issues:
            add(f"  [{it.get('issue')}] {it.get('symbol')}: {it.get('reason')}")
    else:
        add("  none")
    for w in result.get("warnings") or []:
        add(f"  [warning] {w}")
    add("")

    add("PROVENANCE NOTE")
    add("-" * 78)
    add("Every number above is measured on an explicit out-of-sample window;")
    add("the newest 20% of OOF rows are evaluation only. The benchmark never")
    add("changes production features/models/thresholds. Fingerprints tag every")
    add("record, so these numbers are reproducible and attributable to the exact")
    add("input snapshot consumed.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# JSON sanitisation (project convention: 4-dp floats, NaN -> null)
# ---------------------------------------------------------------------------

def _sanitize(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, int) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return round(obj, 4)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(
        description="V1 forecast benchmark (Task 11) — measurement only; "
                    "production behavior is never changed.",
    )
    ap.add_argument("--smoke", action="store_true",
                    help="run over the small SMOKE_UNIVERSE")
    ap.add_argument("--symbols", type=str, default=None,
                    help="comma-separated NSE symbols (overrides the universe)")
    ap.add_argument("--seed", type=int, default=0, help="walk-forward seed (default 0)")
    ap.add_argument("--no-fast", dest="fast", action="store_false",
                    help="use the full-width walk-forward grid (slower)")
    ap.add_argument("--enrich", action="store_true",
                    help="include news/options/macro enrichments (slower)")
    ap.add_argument("--no-model-comparison", dest="model_comparison",
                    action="store_false", help="skip the OOF model-by-model section")
    ap.add_argument("--no-ablation", dest="ablation", action="store_false",
                    help="skip the feature-group ablation section")
    ap.add_argument("--no-failure", dest="failure", action="store_false",
                    help="skip the failure-diagnostic sections")
    ap.add_argument("--label", type=str, default=None,
                    help="label tag embedded in output filenames")
    ap.add_argument("--output-dir", type=str, default="ml_benchmark",
                    help="output directory for JSON + text report")
    args = ap.parse_args(argv)

    if args.symbols:
        symbols = list(dict.fromkeys(
            clean_symbol(s) for s in args.symbols.split(",") if s.strip()))
    elif args.smoke:
        symbols = list(SMOKE_UNIVERSE)
    else:
        symbols = list(DEFAULT_UNIVERSE)

    cfg = BenchmarkConfig(
        symbols=symbols,
        seed=args.seed,
        fast=args.fast,
        enrich=args.enrich,
        label=args.label or ("smoke" if args.smoke else None),
        run_model_comparison=args.model_comparison,
        run_ablation=args.ablation,
        run_failure_diagnostics=args.failure,
        output_dir=args.output_dir,
    )
    logger.info(
        "benchmark starting: universe=%d symbols, seed=%d, fast=%s, "
        "deep={model_comparison:%s, ablation:%s, failure:%s}",
        len(symbols), cfg.seed, cfg.fast,
        cfg.run_model_comparison, cfg.run_ablation, cfg.run_failure_diagnostics,
    )
    result = run_benchmark(cfg)
    paths = save_benchmark(result, output_dir=cfg.output_dir, label=cfg.label)
    print(f"benchmark JSON : {paths['json']}")
    print(f"text report    : {paths['text']}")
    return result


if __name__ == "__main__":
    main()