"""Task 20 — Expected Return + Risk Forecasting Research.

RESEARCH-ONLY module.  Investigates whether forecasting expected return
magnitude and risk contains genuine out-of-sample predictive information
beyond the existing weak V1 directional probability.

Must NOT modify any V1 features, targets, models, ensemble, calibration,
thresholds, backtest, monitor, forecast API, or frontend.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from app.ml import (
    backtest as bt,
    fingerprint as fp,
    models as mo,
    pooled_dataset,
    v2_pooled_models as v2pm,
    v2_selective_forecast as sf,
)

logger = logging.getLogger("equitylens.return_risk")

MODULE_VERSION = "task20-return-risk-v1"
SCHEMA_VERSION = "task20-return-risk-1"
DEFAULT_HORIZON = 20
DEFAULT_SEEDS = (17, 31, 53)
DEFAULT_UNIVERSE = pooled_dataset.DEFAULT_RESEARCH_UNIVERSE
CLASSIFICATIONS = (
    "PROMISING",
    "REJECTED",
    "EVIDENCE STILL INSUFFICIENT",
)
OOF_TEST_SIZE = 60
OOF_STEP = 60
OOF_MIN_TRAIN = 260


def _finite(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    return value


def _json_dump(payload: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_finite(payload), handle, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Regression models for expected return
# ---------------------------------------------------------------------------

def _build_regressor(model_name: str, seed: int = 17) -> Any:
    """Build a regressor for expected return prediction."""
    if model_name == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0)
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=200, max_depth=6, random_state=seed, n_jobs=-1,
        )
    if model_name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=200, max_depth=3, random_state=seed, verbosity=0,
        )
    raise ValueError(f"unknown regressor: {model_name!r}")


def _fit_regressor(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int = 17,
) -> np.ndarray:
    """Train a regressor and predict on test rows."""
    reg = _build_regressor(model_name, seed=seed)
    mask = np.isfinite(y_train)
    if mask.sum() < 10:
        return np.full(len(X_test), np.nan)
    reg.fit(X_train[mask], y_train[mask])
    preds = reg.predict(X_test)
    return np.asarray(preds, dtype=float)


def _fit_direction(
    model_name: str,
    X_train: np.ndarray,
    y_up_train: np.ndarray,
    X_test: np.ndarray,
    seed: int = 17,
) -> np.ndarray:
    """Train a direction model and predict P(up) on test rows."""
    tr = pd.DataFrame(X_train)
    te = pd.DataFrame(X_test)
    try:
        est = mo.fit_model(model_name, tr, pd.Series(y_up_train), seed=seed)
        return np.asarray(mo.predict_up(est, te), dtype=float)
    except Exception as exc:
        logger.debug("direction %s fit failed: %s", model_name, exc)
        return np.full(len(X_test), 0.5)


# ---------------------------------------------------------------------------
# Risk / uncertainty estimation
# ---------------------------------------------------------------------------

def _residual_uncertainty(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute residual-based risk metrics."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    residuals = y_true[mask] - y_pred[mask]
    if len(residuals) < 5:
        return {"cond_vol": None, "residual_mad": None, "rmse": None}
    return {
        "cond_vol": float(np.sqrt(np.mean(residuals ** 2))),
        "residual_mad": float(np.median(np.abs(residuals - np.median(residuals)))),
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
    }


def _quantile_intervals(
    y_pred: np.ndarray,
    residuals_std: float,
    quantiles: tuple[float, float] = (0.05, 0.95),
) -> dict[str, float]:
    """Return quantile-based prediction intervals."""
    z_lo = float(np.percentile(y_pred - 1.96 * residuals_std, quantiles[0] * 100))
    z_hi = float(np.percentile(y_pred + 1.96 * residuals_std, quantiles[1] * 100))
    return {"q_lower": z_lo, "q_upper": z_hi}


def _conditional_volatility_from_residuals(residuals: np.ndarray) -> float:
    """Estimate conditional volatility from regression residuals."""
    mask = np.isfinite(residuals)
    if mask.sum() < 5:
        return float("nan")
    return float(np.sqrt(np.mean(residuals[mask] ** 2)))


# ---------------------------------------------------------------------------
# Entropy interaction diagnostic (one predeclared diagnostic only)
# ---------------------------------------------------------------------------

def _entropy_interaction_diagnostic(oof_records: list[dict[str, Any]]) -> dict[str, Any]:
    """One predeclared diagnostic: correlation between entropy and |ret error|.

    High entropy means P(up) is close to 0.5 (maximal uncertainty).  If high
    uncertainty rows also show larger regression error, entropy-based abstention
    could improve expected-return forecasts.
    """
    entropies: list[float] = []
    errors: list[float] = []
    for rec in oof_records:
        p = rec.get("p_up")
        err = rec.get("ret_error")
        if p is None or not np.isfinite(p) or err is None or not np.isfinite(err):
            continue
        p = float(p)
        if p <= 0 or p >= 1:
            continue
        entropies.append(-(p * math.log(p) + (1 - p) * math.log(1 - p)))
        errors.append(abs(float(err)))

    if len(entropies) < 30:
        return {"n": len(entropies), "corr": None, "interpretation": "insufficient_data"}

    ent_arr = np.asarray(entropies, dtype=float)
    err_arr = np.asarray(errors, dtype=float)
    if np.std(ent_arr) < 1e-10 or np.std(err_arr) < 1e-10:
        return {"n": len(entropies), "corr": 0.0, "interpretation": "zero_variance"}

    corr = float(np.corrcoef(ent_arr, err_arr)[0, 1])
    if abs(corr) > 0.3:
        interpretation = "entropy_reduces_regret"
    elif abs(corr) > 0.15:
        interpretation = "moderate_signal"
    else:
        interpretation = "no_signal"
    return {"n": len(entropies), "corr": corr, "interpretation": interpretation}


# ---------------------------------------------------------------------------
# Main research pipeline
# ---------------------------------------------------------------------------

def run_return_risk_forecast(
    panel: pd.DataFrame,
    *,
    universe: tuple[str, ...] = DEFAULT_UNIVERSE,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    horizon: int = DEFAULT_HORIZON,
    regressors: tuple[str, ...] = ("ridge", "random_forest", "xgboost"),
    direction_model: str = "logistic",
    max_folds: int | None = None,
    final_holdout_fraction: float = 0.20,
) -> dict[str, Any]:
    """Run the full Expected Return + Risk Forecasting research pipeline.

    Returns a results dict suitable for JSON persistence.
    """
    t0 = time.time()

    # --- Feature columns: Config A (V1 baseline) and Config B (plus relative) ---
    matrices = v2pm.build_feature_matrices(panel)
    baseline_cols = list(matrices["baseline"].columns)
    enriched_cols = list(matrices["baseline_plus_relative"].columns)

    # --- Symbol universe ---
    all_symbols = sorted(panel["symbol"].astype(str).unique())
    universe = tuple(s for s in universe if s in all_symbols) or tuple(all_symbols)
    symbols = list(universe)

    # --- Chronological walk-forward folds (same OOF grid as V1 / Task 17) ---
    all_dates = pd.DatetimeIndex(sorted(pd.to_datetime(panel["date"]).dt.normalize().unique()))
    all_folds = bt.purged_walk_forward_splits(
        all_dates, test_size=OOF_TEST_SIZE, step=OOF_STEP,
        min_train=OOF_MIN_TRAIN, embargo=horizon,
    )
    if max_folds is not None and max_folds > 0:
        folds = all_folds[-min(max_folds, len(all_folds)):]
    else:
        folds = all_folds

    # --- Final holdout (newest fraction of panel dates, untouched by fitting) ---
    holdout_n = max(1, int(math.ceil(len(all_dates) * float(final_holdout_fraction))))
    holdout_dates = set(all_dates[-holdout_n:])

    # Pre-indexed targets for O(1) lookups.
    ret_actual = panel["target_ret_20d"]
    up_actual = panel["target_up_20d"]
    date_series = panel["date"]
    symbol_series = panel["symbol"].astype(str)

    oof_records: list[dict[str, Any]] = []
    seed_summary: dict[str, dict[str, Any]] = {}

    for seed in seeds:
        seed_records: list[dict[str, Any]] = []
        for fold_pos, (train_dates, test_dates) in enumerate(folds):
            train_dates = pd.DatetimeIndex(train_dates)
            test_dates = pd.DatetimeIndex(test_dates)
            fno = fold_pos + 1

            tr_mask = pd.to_datetime(date_series).dt.normalize().isin(train_dates)
            te_mask = pd.to_datetime(date_series).dt.normalize().isin(test_dates)

            for symbol in symbols:
                sym_mask = symbol_series == symbol
                tr_idx = panel.index[tr_mask & sym_mask]
                te_idx = panel.index[te_mask & sym_mask]
                if len(tr_idx) < 60 or len(te_idx) == 0:
                    continue

                valid = ret_actual.loc[tr_idx].notna() & up_actual.loc[tr_idx].notna()
                tr_idx = tr_idx[valid.to_numpy()]
                if len(tr_idx) < 30:
                    continue

                y_ret_clean = ret_actual.loc[tr_idx].to_numpy(dtype=float)
                y_up_clean = up_actual.loc[tr_idx].to_numpy(dtype=float)

                configs: list[tuple[str, list[str]]] = [("config_a", baseline_cols)]
                if not set(enriched_cols) == set(baseline_cols):
                    configs.append(("config_b", enriched_cols))

                for config_name, col_list in configs:
                    available = [c for c in col_list if c in panel.columns]
                    if not available:
                        continue

                    Xtr_df = panel.loc[tr_idx, available].replace([np.inf, -np.inf], np.nan)
                    Xte_df = panel.loc[te_idx, available].replace([np.inf, -np.inf], np.nan)

                    imp = bt._MedianImputer(available)
                    Xtr_np = imp.fit_transform(Xtr_df)
                    Xte_np = imp.transform(Xte_df)

                    # Direction model: fit once per (seed, fold, symbol, config).
                    p_up = _fit_direction(direction_model, Xtr_np, y_up_clean, Xte_np, seed=seed)

                    # Two-stage magnitude base: E(return | up) on training rows.
                    up_train = y_ret_clean[y_up_clean == 1]
                    up_mean = float(up_train.mean()) if len(up_train) > 0 else 0.0

                    for reg_name in regressors:
                        try:
                            ret_preds = _fit_regressor(reg_name, Xtr_np, y_ret_clean, Xte_np, seed=seed)
                        except Exception as exc:
                            logger.warning("regressor %s failed for %s: %s", reg_name, symbol, exc)
                            ret_preds = np.full(len(Xte_np), np.nan)

                        for i in range(len(Xte_np)):
                            actual = ret_actual.iat[te_idx[i]]
                            if pd.isna(actual):
                                continue
                            actual_f = float(actual)
                            ret_pred = ret_preds[i]
                            ret_pred_f = float(ret_pred) if np.isfinite(ret_pred) else None
                            p_up_val = float(p_up[i]) if np.isfinite(p_up[i]) else None
                            two_stage = (p_up_val * up_mean) if p_up_val is not None else None

                            rec = {
                                "symbol": symbol,
                                "date": str(pd.Timestamp(date_series.iat[te_idx[i]]).date()),
                                "fold": fno,
                                "seed": seed,
                                "config": config_name,
                                "regressor": reg_name,
                                "ret_predicted": ret_pred_f,
                                "ret_two_stage": float(two_stage) if two_stage is not None else None,
                                "p_up": p_up_val,
                                "up_mean": up_mean,
                                "ret_actual": actual_f,
                                "ret_error": (ret_pred_f - actual_f) if ret_pred_f is not None else None,
                            }
                            seed_records.append(rec)
                            oof_records.append(rec)

        err_sq = [r["ret_error"] ** 2 for r in seed_records if r.get("ret_error") is not None]
        seed_summary[str(seed)] = {
            "rmse": float(np.sqrt(np.mean(err_sq))) if err_sq else None,
            "n_rows": len(seed_records),
        }

    # --- Final holdout evaluation (newest 20% of OOF dates, never fitted) ---
    holdout_oof = [r for r in oof_records if r["date"] in {d.date().isoformat() for d in holdout_dates}]

    holdout_metrics: dict[str, Any] = {}
    if holdout_oof:
        for config_name in sorted({r["config"] for r in holdout_oof}):
            for reg_name in regressors:
                key = f"{config_name}_{reg_name}"
                subset = [r for r in holdout_oof
                          if r["config"] == config_name and r["regressor"] == reg_name
                          and r.get("ret_error") is not None and np.isfinite(r["ret_error"])]
                if len(subset) < 5:
                    continue
                err_arr = np.asarray([r["ret_error"] for r in subset], dtype=float)
                pred_arr = np.asarray([r["ret_predicted"] for r in subset], dtype=float)
                act_arr = pred_arr - err_arr
                m = {
                    "rmse": float(np.sqrt(np.mean(err_arr ** 2))),
                    "mae": float(np.mean(np.abs(err_arr))),
                    "bias": float(np.mean(err_arr)),
                    "corr": float(np.corrcoef(pred_arr, act_arr)[0, 1]) if len(subset) > 5 else None,
                    "n": len(subset),
                }
                holdout_metrics[key] = _finite(m)

    # --- Per-symbol / per-fold / per-seed summary ---
    symbol_metrics: dict[str, Any] = {}
    for symbol in sorted({r["symbol"] for r in oof_records}):
        errs = [r["ret_error"] for r in oof_records
                if r["symbol"] == symbol and r.get("ret_error") is not None]
        if errs:
            arr = np.asarray(errs, dtype=float)
            symbol_metrics[symbol] = {
                "rmse": float(np.sqrt(np.mean(arr ** 2))),
                "mae": float(np.mean(np.abs(arr))),
                "n": len(errs),
            }

    fold_metrics: list[dict[str, Any]] = []
    for fno in sorted({r["fold"] for r in oof_records}):
        errs = [r["ret_error"] for r in oof_records
                if r["fold"] == fno and r.get("ret_error") is not None]
        if errs:
            arr = np.asarray(errs, dtype=float)
            fold_metrics.append({
                "fold": fno,
                "rmse": float(np.sqrt(np.mean(arr ** 2))),
                "mae": float(np.mean(np.abs(arr))),
                "n": len(errs),
            })

    # --- Two-stage vs direct: holdout correlation of direction-aware forecast ---
    two_stage_vs_direct: dict[str, Any] = {}
    if holdout_oof:
        for config_name in sorted({r["config"] for r in holdout_oof}):
            for reg_name in regressors:
                subset = [r for r in holdout_oof
                          if r["config"] == config_name and r["regressor"] == reg_name
                          and None not in (r.get("ret_predicted"), r.get("ret_two_stage"), r.get("ret_actual"))]
                if len(subset) < 5:
                    continue
                pred = np.asarray([r["ret_predicted"] for r in subset], dtype=float)
                two = np.asarray([r["ret_two_stage"] for r in subset], dtype=float)
                act = np.asarray([r["ret_actual"] for r in subset], dtype=float)
                two_stage_vs_direct[f"{config_name}_{reg_name}"] = {
                    "direct_corr": float(np.corrcoef(pred, act)[0, 1]),
                    "two_stage_mae": float(np.mean(np.abs(two - act))),
                    "direct_mae": float(np.mean(np.abs(pred - act))),
                    "n": len(subset),
                }

    # --- Shuffled-target null control (regression on shuffled returns) ---
    null_rmse: float | None = None
    n_null_rows = 0
    try:
        shuffled = ret_actual.sample(frac=1, random_state=42).reset_index(drop=True)
        panel_null = panel.copy()
        panel_null["target_ret_20d"] = shuffled.to_numpy()

        null_errors: list[float] = []
        for tr_dates, te_dates in folds[: min(2, len(folds))]:
            tr_dates = pd.DatetimeIndex(tr_dates)
            te_dates = pd.DatetimeIndex(te_dates)
            tr_mask = pd.to_datetime(date_series).dt.normalize().isin(tr_dates)
            te_mask = pd.to_datetime(date_series).dt.normalize().isin(te_dates)
            for symbol in symbols[:5]:
                sym_mask = symbol_series == symbol
                tr_idx = panel_null.index[tr_mask & sym_mask]
                te_idx = panel_null.index[te_mask & sym_mask]
                if len(tr_idx) < 60 or len(te_idx) == 0:
                    continue
                valid = panel_null.loc[tr_idx, "target_ret_20d"].notna()
                tr_idx = tr_idx[valid.to_numpy()]
                if len(tr_idx) < 30:
                    continue
                y_ret_shuf = panel_null.loc[tr_idx, "target_ret_20d"].to_numpy(dtype=float)
                available = [c for c in baseline_cols if c in panel_null.columns]
                Xtr_df = panel_null.loc[tr_idx, available].replace([np.inf, -np.inf], np.nan)
                Xte_df = panel_null.loc[te_idx, available].replace([np.inf, -np.inf], np.nan)
                imp = bt._MedianImputer(available)
                Xtr_np = imp.fit_transform(Xtr_df)
                Xte_np = imp.transform(Xte_df)
                preds = _fit_regressor("ridge", Xtr_np, y_ret_shuf, Xte_np, seed=42)
                for i in range(len(Xte_np)):
                    act = panel_null.loc[te_idx[i], "target_ret_20d"]
                    if pd.isna(act):
                        continue
                    if np.isfinite(preds[i]):
                        null_errors.append(float(preds[i] - float(act)))
                        n_null_rows += 1
        if null_errors:
            null_rmse = float(np.sqrt(np.mean(np.asarray(null_errors) ** 2)))
    except Exception as exc:
        logger.warning("null control failed: %s", exc)

    # --- Entropy interaction diagnostic ---
    ent_diag = _entropy_interaction_diagnostic(oof_records)

    # --- Bootstrap CI on holdout RMSE (all regressor/config rows pooled) ---
    bootstrap_ci: dict[str, Any] = None
    holdout_errors = [r["ret_error"] for r in oof_records
                      if r["date"] in {d.date().isoformat() for d in holdout_dates}
                      and r.get("ret_error") is not None]
    if len(holdout_errors) > 20:
        err_arr = np.asarray(holdout_errors, dtype=float)
        rng = np.random.default_rng(17)
        boot = [
            float(np.sqrt(np.mean(rng.choice(err_arr, size=len(err_arr), replace=True) ** 2)))
            for _ in range(2000)
        ]
        bootstrap_ci = {
            "rmse_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "n_bootstrap": 2000,
            "n_rows": len(err_arr),
        }

    # --- Classification ---
    classification = _classify_result(
        holdout_metrics=holdout_metrics,
        null_rmse=null_rmse,
        symbol_metrics=symbol_metrics,
        fold_metrics=fold_metrics,
        ent_diag=ent_diag,
    )

    elapsed = time.time() - t0

    result: dict[str, Any] = {
        "module_version": MODULE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "direction_model": direction_model,
            "regressors": list(regressors),
            "seeds": list(seeds),
            "horizon": horizon,
            "max_folds": (None if max_folds is None else max_folds),
            "final_holdout_fraction": final_holdout_fraction,
            "oof_test_size": OOF_TEST_SIZE,
            "oof_step": OOF_STEP,
            "oof_min_train": OOF_MIN_TRAIN,
            "feature_configs": ["config_a"] + ([] if set(enriched_cols) == set(baseline_cols) else ["config_b"]),
            "config_a_cols": len(baseline_cols),
            "config_b_cols": len(enriched_cols),
            "config_b_redundant": set(enriched_cols) == set(baseline_cols),
        },
        "classification": classification,
        "holdout_metrics": holdout_metrics,
        "two_stage_vs_direct": two_stage_vs_direct,
        "per_symbol": symbol_metrics,
        "per_fold": fold_metrics,
        "per_seed": seed_summary,
        "null_control": {"null_rmse": null_rmse, "n_null_rows": n_null_rows},
        "entropy_interaction": ent_diag,
        "bootstrap_ci": bootstrap_ci,
        "symbols": symbols,
        "n_oof_rows": len(oof_records),
        "n_holdout_rows": len(holdout_oof),
        "elapsed_seconds": round(elapsed, 1),
        "generated_at": fp.compose_fingerprint(
            {"n": str(len(oof_records)), "t": str(int(elapsed))},
            name="task20",
        ),
    }
    return result


def _classify_result(
    holdout_metrics: dict[str, Any],
    null_rmse: float | None,
    symbol_metrics: dict[str, Any],
    fold_metrics: list[dict[str, Any]],
    ent_diag: dict[str, Any],
) -> str:
    """Classify Task 20 result: PROMISING / REJECTED / EVIDENCE STILL INSUFFICIENT."""

    if not holdout_metrics:
        return "EVIDENCE STILL INSUFFICIENT"

    best_rmse = None
    for key, m in holdout_metrics.items():
        if m.get("rmse") is not None:
            if best_rmse is None or m["rmse"] < best_rmse:
                best_rmse = m["rmse"]
    if best_rmse is None:
        return "REJECTED"

    if null_rmse is not None and best_rmse >= null_rmse * 0.9:
        return "REJECTED"

    good_symbols = sum(1 for m in symbol_metrics.values()
                       if m.get("rmse") is not None and m["rmse"] < 0.05)
    symbol_ratio = good_symbols / max(len(symbol_metrics), 1)

    fold_rmses = [m.get("rmse") for m in fold_metrics if m.get("rmse") is not None]
    fold_cv = float(np.std(fold_rmses) / max(np.mean(fold_rmses), 1e-10)) if fold_rmses else float("inf")

    ent_signal = ent_diag.get("interpretation", "no_signal")

    if symbol_ratio >= 0.3 and fold_cv < 2.0 and ent_signal in ("entropy_reduces_regret", "moderate_signal"):
        return "PROMISING"
    if symbol_ratio >= 0.3 and fold_cv < 2.0:
        return "EVIDENCE STILL INSUFFICIENT"
    if symbol_ratio < 0.1:
        return "REJECTED"
    return "EVIDENCE STILL INSUFFICIENT"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_results(result: dict[str, Any], output_dir: str = "ml_v2_return_risk") -> Path:
    """Persist Task 20 results."""
    out_dir = Path(__file__).resolve().parents[2] / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "task20_result.json"
    _json_dump(result, result_path)
    logger.info("Task 20 results saved: %s", result_path)
    return result_path


def load_results(path: str | Path) -> dict[str, Any]:
    """Load Task 20 results from JSON."""
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Task 20 — Expected Return + Risk Forecasting Research")
    parser.add_argument("--snapshot", type=str, default=None, help="Path to Task 17 snapshot CSV")
    parser.add_argument("--output-dir", type=str, default="ml_v2_return_risk")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--max-folds", type=int, default=None)
    args = parser.parse_args()

    panel, meta = sf.load_shared_snapshot(args.snapshot)
    logger.info("Loaded snapshot: %d rows, %d symbols", meta["n_rows"], meta["n_symbols"])

    result = run_return_risk_forecast(
        panel, seeds=tuple(args.seeds), max_folds=args.max_folds,
    )
    path = save_results(result, args.output_dir)

    print(f"\n=== Task 20 Classification: {result['classification']} ===")
    print(f"OOF rows: {result['n_oof_rows']}")
    print(f"Holdout rows: {result['n_holdout_rows']}")
    print(f"Entropy interaction: {result['entropy_interaction']}")
    print(f"Results saved to: {path}")


if __name__ == "__main__":
    main()