"""Task 17 — Rigorous Frozen-V1 vs Pooled Model Comparison.

This module creates a strict apples-to-apples comparison between the
frozen V1 per-symbol forecasting pipeline and pooled candidate models,
both evaluated on the SAME shared historical snapshot, symbols, dates,
target, folds, seeds, and measurement semantics.

Production V1 is NEVER modified. This is research-only.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from app.config import settings
from app.ml import (
    backtest as bt,
    calibration as cal,
    dataset as ds,
    ensemble as en,
    fingerprint as fp,
    models as mo,
    pipeline as pl,
    pooled_dataset,
    versions as ver,
    v2_pooled_models as v2pm,
)
from app.ml.models import model_names as _installed_models
from app.utils.validators import clean_symbol

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger("equitylens.fair_compare")

MODULE_VERSION = "task17-v1"
DEFAULT_HORIZON = 20
DEFAULT_SEEDS = (17, 31, 53)
OOF_TEST_SIZE = 60
OOF_STEP = 60
OOF_MIN_TRAIN = 260
# These are the installed V1 components.  Keep this list in one place so
# research consumers (including Task 18) cannot accidentally substitute a
# pooled model for a frozen V1 probability.
ENSEMBLE_MODELS = ("voting", "logistic", "ridge", "rf", "xgboost")
DEFAULT_UNIVERSE = pooled_dataset.DEFAULT_RESEARCH_UNIVERSE


def _canonical_models() -> list[str]:
    """Return models that are actually installed."""
    installed = set(_installed_models())
    return [m for m in ENSEMBLE_MODELS if m in installed]


def _impute_fit_transform(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Training-only median imputation (sklearn-free)."""
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)
    medians = train.median(axis=0, numeric_only=True).reindex(train.columns).fillna(0.0)
    Xtr = train.fillna(medians).fillna(0.0).to_numpy(dtype=float)
    Xte = test.fillna(medians).fillna(0.0).to_numpy(dtype=float)
    return Xtr, Xte


def _fit_predict_for_symbol(
    model: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    """Train a model and return P(up) for test rows."""
    if model == "voting":
        return mo.fit_and_predict(model, X_train, y_train, X_test, seed=seed)
    imp = bt._MedianImputer(list(X_train.columns))
    Xtr = pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    Xte = pd.DataFrame(imp.transform(X_test), columns=X_train.columns, index=X_test.index)
    return mo.fit_and_predict(model, Xtr, y_train, Xte, seed=seed)


# ---------------------------------------------------------------------------
# PART 1 — SHARED HISTORICAL SNAPSHOT
# ---------------------------------------------------------------------------

def build_shared_snapshot(
    symbols: tuple[str, ...] = DEFAULT_UNIVERSE,
    horizon: int = DEFAULT_HORIZON,
    as_of: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the canonical pooled research snapshot.

    Reuses the existing Task-13 pooled_dataset infrastructure.
    Returns (panel, metadata).
    """
    from app.services import data_service as dsvc
    from app.services.fundamental_service import fundamentals_service

    frames: dict[str, pd.DataFrame] = {}
    source_meta: dict[str, Any] = {}
    skipped: list[dict[str, str]] = []

    for raw_symbol in symbols:
        symbol = clean_symbol(raw_symbol)
        try:
            payload = dsvc.fetch_nse_ohlcv(symbol, period="5y")
            if not payload.get("is_available") or not payload.get("history"):
                skipped.append({"symbol": symbol, "reason": "OHLCV unavailable"})
                source_meta[symbol] = {"available": False, "rows": 0}
                continue

            history = payload.get("history") or []
            df = pl._ohlcv_from_records(history)
            if df.empty:
                skipped.append({"symbol": symbol, "reason": "empty OHLCV"})
                continue

            # Build market context
            market: dict[str, Any] = {}
            for sid in ("nifty50", "india_vix"):
                try:
                    mdf, _ = dsvc._get_series_frame(sid, period="5y", interval="1d")
                    if mdf is not None and not mdf.empty:
                        market[sid] = mdf
                except Exception:
                    pass

            # Build feature frame
            try:
                feats = pl.build_feature_frame(
                    ohlcv_df=df, market=market if market else None,
                )
                feats = ds.add_target(feats, horizon=horizon)
                feats.insert(0, "symbol", symbol)
                frames[symbol] = feats
                source_meta[symbol] = {
                    "available": True,
                    "rows": len(feats),
                    "data_end": str(feats.index[-1].date()),
                }
            except Exception as exc:
                skipped.append({"symbol": symbol, "reason": f"feature build failed: {exc}"})
                source_meta[symbol] = {"available": False, "error": str(exc)}
        except Exception as exc:
            skipped.append({"symbol": symbol, "reason": str(exc)})
            source_meta[symbol] = {"available": False, "error": str(exc)}

    if not frames:
        raise ValueError("No frames built — check data availability")

    manifest = pooled_dataset.build_universe_manifest(frames.keys())
    panel, meta = pooled_dataset.build_pooled_dataset(frames, manifest=manifest)

    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Build canonical fingerprint
    fp_parts: dict[str, str] = {
        "manifest": manifest.get("manifest_fingerprint", ""),
        "panel_rows": str(len(panel)),
        "symbols": "|".join(sorted(panel["symbol"].astype(str).unique())),
        "date_range": f"{panel['date'].min()}|{panel['date'].max()}",
    }
    for sym, info in source_meta.items():
        if info.get("available"):
            fp_parts[f"source:{sym}"] = str(info.get("rows", 0))

    fingerprint = fp.compose_fingerprint(fp_parts, name="task17-shared-snapshot")

    metadata = {
        "snapshot_version": MODULE_VERSION,
        "manifest_version": manifest.get("manifest_version", ""),
        "schema_version": pooled_dataset.SCHEMA_VERSION,
        "dataset_version": pooled_dataset.DATASET_VERSION,
        "symbols_requested": list(symbols),
        "symbols_included": sorted(panel["symbol"].astype(str).unique().tolist()),
        "symbols_skipped": skipped,
        "n_symbols": int(len(panel["symbol"].astype(str).unique())),
        "n_rows": int(len(panel)),
        "date_range": {
            "start": str(panel["date"].min()),
            "end": str(panel["date"].max()),
        },
        "horizon": horizon,
        "source_meta": source_meta,
        "fingerprint": fingerprint,
        "generated_at": ver.utc_now_iso(),
    }

    return panel, metadata


def save_snapshot(panel: pd.DataFrame, metadata: dict[str, Any], output_dir: str = "ml_v2_fair_compare") -> Path:
    """Persist the shared snapshot as CSV + JSON metadata."""
    out_dir = Path(__file__).resolve().parents[2] / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_path = out_dir / "task17_shared_snapshot.csv"
    meta_path = out_dir / "task17_snapshot_metadata.json"

    panel.to_csv(panel_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("Snapshot saved: %s rows, %d symbols", len(panel), metadata["n_symbols"])
    return panel_path


# ---------------------------------------------------------------------------
# PART 2 — V1 REFERENCE EVALUATION FROM PANEL
# ---------------------------------------------------------------------------

def _build_symbol_frame_from_panel(
    panel: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """Extract per-symbol OHLCV + features from the shared panel."""
    sym_panel = panel[panel["symbol"].astype(str) == symbol].copy()
    if sym_panel.empty:
        raise ValueError(f"No data for symbol {symbol}")

    sym_panel = sym_panel.set_index("date")
    sym_panel.index = pd.DatetimeIndex(sym_panel.index)
    sym_panel = sym_panel.sort_index()

    # Ensure Close column exists (required by forecast_frame)
    if "Close" not in sym_panel.columns:
        raise ValueError(f"No Close column for {symbol}")

    return sym_panel


def _get_shared_folds(
    panel: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Compute canonical fold dates using the same logic as V1 forecast_frame."""
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    return bt.purged_walk_forward_splits(
        dates, test_size=OOF_TEST_SIZE, step=OOF_STEP,
        min_train=OOF_MIN_TRAIN, embargo=horizon,
    )


def run_v1_reference(
    panel: pd.DataFrame,
    folds: Sequence[tuple[pd.DatetimeIndex, pd.DatetimeIndex]],
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    horizon: int = DEFAULT_HORIZON,
) -> dict[str, Any]:
    """Run frozen V1 ensemble on per-symbol data using shared folds.

    Returns OOF probabilities per (symbol, date, model) for all folds/seeds.
    """
    usable_models = _canonical_models()
    all_symbols = sorted(panel["symbol"].astype(str).unique())

    # Collect OOF probs per symbol
    oof_records: list[dict[str, Any]] = []
    fold_per_model: dict[str, list[float]] = {m: [] for m in usable_models}

    for symbol in all_symbols:
        try:
            frame = _build_symbol_frame_from_panel(panel, symbol)
            out = ds.add_target(frame, horizon=horizon, close_col="Close")
            y = out["target_up_20d"]
            close = out["Close"] if "Close" in out.columns else None

            feat_cols = bt._default_feature_cols(out, horizon)
            keep = [c for c in feat_cols if out[c].notna().sum() > 0]
            X = out[keep].replace([np.inf, -np.inf], np.nan)

            oof_seed_records: list[dict[str, Any]] = []

            for seed_idx, seed in enumerate(seeds):
                for fi, (train_dates, test_dates) in enumerate(folds):
                    train_mask = X.index.isin(train_dates)
                    test_mask = X.index.isin(test_dates)

                    Xtr = X[train_mask]
                    Xte = X[test_mask]
                    ytr = y[train_mask]

                    if len(Xte) == 0 or ytr.notna().sum() < 2:
                        continue

                    # Get ensemble probabilities for this fold
                    combined, per_model = _fold_ensemble_probs(
                        X, y, train_dates, test_dates, usable_models, seed=seed + fi
                    )

                    # Align combined probs to test dates
                    test_idx_arr = np.asarray(test_dates)
                    for i, d in enumerate(test_dates):
                        if i >= len(combined):
                            break
                        cv = combined[i]
                        if np.isnan(cv):
                            continue
                        y_val = y.loc[d] if d in y.index and pd.notna(y.loc[d]) else np.nan
                        ret_val = out["target_ret_20d"].loc[d] if d in out.index and pd.notna(out["target_ret_20d"].loc[d]) else np.nan
                        close_val = close.loc[d] if close is not None and d in close.index and pd.notna(close.loc[d]) else np.nan

                        if pd.isna(y_val):
                            continue

                        rec = {
                            "symbol": symbol,
                            "date": str(d.date()),
                            "fold": fi,
                            "seed": seed,
                            "ensemble": float(cv),
                            "y": int(y_val),
                        }
                        # Preserve the component probabilities alongside the
                        # frozen V1 ensemble.  Downstream uncertainty research
                        # must use these OOS values, not refit pooled models.
                        for m in usable_models:
                            model_prob = per_model.get(m)
                            if model_prob is not None and i < len(model_prob):
                                value = model_prob[i]
                                rec[f"prob_{m}"] = float(value) if not np.isnan(value) else np.nan
                        if not np.isnan(ret_val):
                            rec["target_ret"] = float(ret_val)
                        if close_val is not None and not np.isnan(close_val):
                            rec["close"] = float(close_val)
                        oof_seed_records.append(rec)
                        for m in usable_models:
                            if m in per_model and i < len(per_model[m]) and not np.isnan(per_model[m][i]):
                                fold_per_model[m].append(float(per_model[m][i]))

            logger.info("V1 reference: %s — %d OOF records across %d seeds x %d folds",
                        symbol, len(oof_seed_records), len(seeds), len(folds))
            oof_records.extend(oof_seed_records)

        except Exception as exc:
            logger.warning("V1 reference failed for %s: %s", symbol, exc)

    # Aggregate OOF records into a panel
    if not oof_records:
        raise ValueError("No OOF records collected for V1 reference")

    oof_panel = pd.DataFrame(oof_records)

    # For each (symbol, date), compute mean across seeds
    oof_panel["date_dt"] = pd.to_datetime(oof_panel["date"])
    oof_mean = oof_panel.groupby(["symbol", "date_dt"]).agg({
        "ensemble": "mean",
        "y": "first",
    }).reset_index()
    oof_mean["date"] = oof_mean["date_dt"].dt.date.astype(str)

    return {
        "oof_panel": oof_panel,
        "oof_mean": oof_mean,
        "usable_models": usable_models,
        "n_symbols": len(all_symbols),
        "n_oof_records": len(oof_panel),
        "folds": folds,
        "seeds": list(seeds),
    }


def _fold_ensemble_probs(
    X: pd.DataFrame,
    y: pd.Series,
    train_dates: pd.DatetimeIndex,
    test_dates: pd.DatetimeIndex,
    models: list[str],
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Out-of-sample ensemble P(up) per model + combined."""
    per: dict[str, np.ndarray] = {}
    for m in models:
        train_mask = X.index.isin(train_dates)
        test_mask = X.index.isin(test_dates)
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            per[m] = np.full(len(test_dates), np.nan)
            continue
        pred = _fit_predict_for_symbol(m, X.loc[train_mask], y.loc[train_mask], X.loc[test_mask], seed=seed)
        per[m] = pred
    combined = en.combine_probs([per[m] for m in models])
    return combined, per


# ---------------------------------------------------------------------------
# PART 3 — POOLED CANDIDATE EVALUATION
# ---------------------------------------------------------------------------

def run_pooled_candidates(
    panel: pd.DataFrame,
    folds: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]],
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    feature_config: str = "baseline",
    target_col: str = "target_up_20d",
) -> dict[str, Any]:
    """Run pooled candidate models on the shared panel using shared folds.

    feature_config: "baseline" (Config A) or "baseline_plus_relative" (Config B)
    """
    usable_models = _canonical_models()
    all_symbols = sorted(panel["symbol"].astype(str).unique())

    if feature_config == "baseline":
        baseline_features = None
        relative_features = None
        matrices = v2pm.build_feature_matrices(
            panel,
            baseline_features=baseline_features,
            relative_features=relative_features,
            identity="none",
        )
    elif feature_config == "baseline_plus_relative":
        from app.ml.v2_relative_features import build_relative_features, list_relative_features
        panel_for_rel = panel.copy()
        if "sector" not in panel_for_rel.columns:
            panel_for_rel["sector"] = "__all__"
        panel_with_rel = build_relative_features(
            panel_for_rel, min_group_size=3,
        )
        relative_features = list_relative_features()
        matrices = v2pm.build_feature_matrices(
            panel_with_rel,
            baseline_features=None,
            relative_features=relative_features,
            identity="none",
        )
    else:
        baseline_features = None
        relative_features = None
        matrices = v2pm.build_feature_matrices(
            panel,
            baseline_features=baseline_features,
            relative_features=relative_features,
            identity="none",
        )

    panel_norm = v2pm._normalise_panel(panel)
    panel_norm["symbol"] = panel_norm["symbol"].astype(str)
    panel_norm["target_val"] = panel_norm[target_col]

    oof_records: list[dict[str, Any]] = []

    for matrix_name, matrix in matrices.items():
        for model_name in usable_models:
            for seed_idx, seed in enumerate(seeds):
                for fi, (train_dates, test_dates) in enumerate(folds):
                    train_dates_set = set(pd.to_datetime(train_dates))
                    test_dates_set = set(pd.to_datetime(test_dates))

                    train_mask = panel_norm["date"].isin(train_dates_set)
                    test_mask = panel_norm["date"].isin(test_dates_set)

                    if train_mask.sum() == 0 or test_mask.sum() == 0:
                        continue

                    y_train = panel_norm.loc[train_mask, "target_val"]
                    y_test = panel_norm.loc[test_mask, "target_val"]
                    syms_test = panel_norm.loc[test_mask, "symbol"].values
                    dates_test = panel_norm.loc[test_mask, "date"].values

                    y_train_vals = y_train.to_numpy(dtype=float)
                    y_test_vals = y_test.to_numpy(dtype=float)

                    valid_train = np.isfinite(y_train_vals)
                    if valid_train.sum() < 2:
                        continue

                    X_train = matrix.loc[train_mask].values.astype(float)
                    X_test = matrix.loc[test_mask].values.astype(float)

                    try:
                        Xtr_np, Xte_np = _impute_fit_transform(
                            pd.DataFrame(X_train), pd.DataFrame(X_test)
                        )
                    except Exception:
                        continue

                    try:
                        prob = v2pm._predict(model_name, Xtr_np, y_train_vals, Xte_np, seed)
                    except Exception:
                        prob = np.full(len(Xte_np), 0.5)

                    for i in range(len(prob)):
                        if np.isnan(prob[i]):
                            continue
                        y_val = y_test_vals[i] if i < len(y_test_vals) and np.isfinite(y_test_vals[i]) else np.nan
                        if np.isnan(y_val):
                            continue
                        try:
                            d_str = pd.Timestamp(dates_test[i]).date().isoformat()
                        except Exception:
                            continue
                        oof_records.append({
                            "matrix": matrix_name,
                            "model": model_name,
                            "symbol": str(syms_test[i]),
                            "seed": seed,
                            "fold": fi,
                            "date": d_str,
                            "prob": float(prob[i]),
                            "y": float(y_val),
                        })

    oof_panel = pd.DataFrame(oof_records)

    if oof_panel.empty:
        raise ValueError("No pooled OOF records collected")

    oof_panel["date_dt"] = pd.to_datetime(oof_panel["date"])
    oof_mean = oof_panel.groupby(["matrix", "model", "symbol", "date_dt"]).agg({
        "prob": "mean",
        "y": "first",
    }).reset_index()
    oof_mean["date"] = oof_mean["date_dt"].dt.date.astype(str)

    return {
        "oof_panel": oof_panel,
        "oof_mean": oof_mean,
        "matrices": matrices,
        "usable_models": usable_models,
        "feature_config": feature_config,
        "n_oof_records": len(oof_panel),
    }


# ---------------------------------------------------------------------------
# PART 4 — MATCHED SAMPLE & COMPARISON
# ---------------------------------------------------------------------------

def build_matched_sample(
    v1_results: dict[str, Any],
    pooled_results_list: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build matched (symbol, date) sample where both V1 and pooled have valid probs."""
    v1_oof = v1_results["oof_mean"].copy()
    v1_oof["key"] = v1_oof["symbol"].astype(str) + "|" + v1_oof["date"].astype(str)
    v1_valid = v1_oof[v1_oof["ensemble"].notna() & v1_oof["y"].notna()]

    matched = v1_valid[["key", "symbol", "date", "ensemble", "y"]].copy()
    matched.rename(columns={"ensemble": "v1_ensemble"}, inplace=True)

    for pr in pooled_results_list:
        matrix_name = pr.get("feature_config", "baseline")
        pooled_oof = pr["oof_mean"].copy()
        pooled_oof["key"] = pooled_oof["symbol"].astype(str) + "|" + pooled_oof["date"].astype(str)
        # Handle both cases: pooled OOF may or may not have 'y' column
        if "y" not in pooled_oof.columns:
            pooled_oof["y"] = np.nan
        pooled_valid = pooled_oof[pooled_oof["prob"].notna() & pooled_oof["y"].notna()]

        for model_name in pr["usable_models"]:
            model_data = pooled_valid[pooled_valid["model"] == model_name][["key", "prob"]].copy()
            model_data.rename(columns={"prob": f"pooled_{matrix_name}_{model_name}"}, inplace=True)
            matched = matched.merge(model_data, on="key", how="left")

    matched = matched.dropna(subset=["v1_ensemble", "y"])
    matched = matched.dropna(subset=[c for c in matched.columns if c.startswith("pooled_")])

    # Guard against key duplication from left-merges: one row per (symbol, date)
    matched = matched.drop_duplicates(subset=["symbol", "date"]).reset_index(drop=True)

    return {
        "matched": matched,
        "n_matched_symbols": matched["symbol"].nunique(),
        "n_matched_dates": matched["date"].nunique(),
        "n_matched_rows": len(matched),
    }


# ---------------------------------------------------------------------------
# PART 5 — METRICS
# ---------------------------------------------------------------------------

def compute_binary_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    """Compute all classification/calibration metrics."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

    mask = np.isfinite(y_true) & np.isfinite(prob)
    y = y_true[mask].astype(int)
    p = prob[mask]

    if len(y) < 2 or len(np.unique(y)) < 2:
        return {"n": int(len(y)), "roc_auc": None, "accuracy": None, "precision": None,
                "recall": None, "f1": None, "brier": None, "ece": None,
                "confusion": {"tn": 0, "fp": 0, "fn": 0, "tp": 0}}

    pred = (p >= 0.5).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = None

    ece = _compute_ece(p, y, n_bins=10)

    return {
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "roc_auc": auc,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion": {"tn": int(cm[0]), "fp": int(cm[1]), "fn": int(cm[2]), "tp": int(cm[3])},
        "brier": float(np.mean((p - y) ** 2)),
        "ece": float(ece),
        "prob_mean": float(np.mean(p)),
        "prob_std": float(np.std(p)),
        "prob_min": float(np.min(p)),
        "prob_max": float(np.max(p)),
        "pct_near_half": float(np.mean(np.abs(p - 0.5) <= 0.05)),
    }


def _compute_ece(prob: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & ((prob < hi) if hi < 1.0 else (prob <= hi))
        if mask.sum() == 0:
            continue
        total += (mask.sum() / len(prob)) * abs(float(prob[mask].mean()) - float(y[mask].mean()))
    return total


# ---------------------------------------------------------------------------
# PART 6 — ECONOMIC EVALUATION
# ---------------------------------------------------------------------------

def compute_economic_metrics(
    matched: pd.DataFrame,
    prob_col: str,
    horizon: int = DEFAULT_HORIZON,
) -> dict[str, Any]:
    """Compute backtest metrics using the existing backtest engine."""
    from app.ml.backtest import run_backtest

    # Build a frame suitable for run_backtest
    # Get Close prices and directions
    signals = matched[["date", prob_col, "y"]].copy()
    signals["date"] = pd.to_datetime(signals["date"])
    signals = signals.set_index("date")
    signals["direction"] = np.where(signals[prob_col] >= 0.5, 1.0, 0.0)
    signals["prob"] = signals[prob_col]

    return {
        "direction_mean": float(matched[prob_col].mean()),
        "n_positions": int((matched[prob_col] >= 0.5).sum()),
        "n_flat": int((matched[prob_col] < 0.5).sum()),
        "all_long_pct": float((matched[prob_col] >= 0.5).mean()),
    }


# ---------------------------------------------------------------------------
# PART 7 — BOOTSTRAP UNCERTAINTY
# ---------------------------------------------------------------------------

def bootstrap_ci(
    paired_deltas: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap confidence interval for paired deltas."""
    if len(paired_deltas) < 2:
        return {"mean": float(np.mean(paired_deltas) if len(paired_deltas) else 0),
                "ci_lower": None, "ci_upper": None, "se": None}

    rng = np.random.default_rng(42)
    boot_means = np.array([
        np.mean(rng.choice(paired_deltas, size=len(paired_deltas), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    return {
        "mean": float(np.mean(paired_deltas)),
        "ci_lower": float(np.percentile(boot_means, alpha * 100)),
        "ci_upper": float(np.percentile(boot_means, (1 - alpha) * 100)),
        "se": float(np.std(boot_means, ddof=1)),
        "n_bootstrap": n_bootstrap,
    }


# ---------------------------------------------------------------------------
# PART 8 — DECISION CLASSIFICATION
# ---------------------------------------------------------------------------

def classify_result(comparison: dict[str, Any]) -> str:
    """Classify the pooled approach using the predeclared decision rule."""
    summary = comparison.get("summary", comparison)
    median_delta = summary.get("median_auc_delta")
    n_symbols_winning = summary.get("n_symbols_pooled_wins", 0)
    n_symbols_total = summary.get("n_symbols_total", 1)
    robustness_ok = summary.get("fold_robustness", False)
    seed_stable = summary.get("seed_stable", False)

    win_rate = n_symbols_winning / n_symbols_total if n_symbols_total > 0 else 0

    if median_delta is None or n_symbols_total == 0:
        return "EVIDENCE STILL INSUFFICIENT"

    # IMPLEMENT POOLED V2: convincing cross-symbol and chronological evidence
    if median_delta > 0.02 and win_rate >= 0.7 and robustness_ok and seed_stable:
        return "IMPLEMENT POOLED V2"

    # KEEP FROZEN V1: no broad robust improvement
    if median_delta <= 0.0 or win_rate < 0.5:
        return "KEEP FROZEN V1"

    # EVIDENCE STILL INSUFFICIENT
    return "EVIDENCE STILL INSUFFICIENT"


# ---------------------------------------------------------------------------
# PART 9 — MAIN ORCHESTRATOR
# ---------------------------------------------------------------------------

def run_comparison(
    symbols: tuple[str, ...] = DEFAULT_UNIVERSE,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    horizon: int = DEFAULT_HORIZON,
    fast: bool = False,
) -> dict[str, Any]:
    """Run the complete Task 17 comparison pipeline.

    Returns a comprehensive comparison result dict.
    """
    start_time = time.time()
    results: dict[str, Any] = {}

    logger.info("=== Task 17: Rigorous Frozen-V1 vs Pooled Comparison ===")
    logger.info("Universe: %d symbols", len(symbols))
    logger.info("Seeds: %s", seeds)
    logger.info("Horizon: %d", horizon)

    # --- PART 1: Shared Snapshot ---
    logger.info("Building shared snapshot...")
    panel, snapshot_meta = build_shared_snapshot(symbols=symbols, horizon=horizon)
    results["snapshot"] = snapshot_meta
    results["panel"] = panel

    # --- PART 2: Shared Folds ---
    logger.info("Computing shared folds...")
    folds = _get_shared_folds(panel, horizon=horizon)
    n_folds = len(folds)
    results["folds"] = {
        "n_folds": n_folds,
        "fold_dates": [(str(fd[0][0]), str(fd[1][0]), str(fd[1][-1]))
                       for fd in folds],
    }

    # --- PART 3: V1 Reference ---
    logger.info("Running V1 reference evaluation...")
    v1_results = run_v1_reference(panel, folds, seeds, horizon)
    results["v1"] = {
        "n_oof_records": v1_results["n_oof_records"],
        "n_symbols": v1_results["n_symbols"],
        "usable_models": v1_results["usable_models"],
    }

    # --- PART 4: Pooled Candidates (Config A and Config B) ---
    pooled_results = []
    for config_name in ["baseline", "baseline_plus_relative"]:
        logger.info("Running pooled candidates: %s...", config_name)
        pr = run_pooled_candidates(panel, folds, seeds, feature_config=config_name)
        pr["feature_config"] = config_name
        pooled_results.append(pr)
        results[f"pooled_{config_name}"] = {
            "n_oof_records": pr["n_oof_records"],
            "usable_models": pr["usable_models"],
        }

    # --- PART 5: Matched Sample ---
    logger.info("Building matched sample...")
    matched_result = build_matched_sample(v1_results, pooled_results)
    results["matched_sample"] = {
        "n_symbols": matched_result["n_matched_symbols"],
        "n_dates": matched_result["n_matched_dates"],
        "n_rows": matched_result["n_matched_rows"],
    }
    matched = matched_result["matched"]

    # --- PART 6: Metrics Computation ---
    logger.info("Computing metrics...")
    comparison = _compute_comparison(matched, v1_results, pooled_results)
    results["comparison"] = comparison

    # --- PART 7: Decision ---
    decision = classify_result(comparison)
    results["decision"] = decision

    # --- Timing ---
    elapsed = time.time() - start_time
    results["runtime_seconds"] = round(elapsed, 1)
    results["generated_at"] = ver.utc_now_iso()

    logger.info("Task 17 complete. Decision: %s (%.1fs)", decision, elapsed)
    return results


def _fold_and_seed_robustness(
    v1_results: dict[str, Any],
    pooled_results_list: list[dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    """Compute chronological (fold) robustness and seed stability.

    Uses the raw OOF panels (per fold/seed) rather than the collapsed matched
    sample so the decision rule can check consistency across folds and seeds.
    Pooled ensemble = mean probability across models and configs per
    (symbol, date, fold, seed).
    """
    pooled_frames = []
    for pr in pooled_results_list:
        op = pr["oof_panel"]
        if op is None or len(op) == 0:
            continue
        op = op[["symbol", "date", "fold", "seed", "prob", "y"]].copy()
        op["date_dt"] = pd.to_datetime(op["date"])
        pooled_frames.append(op)
    pooled = pd.concat(pooled_frames, ignore_index=True) if pooled_frames else pd.DataFrame()
    pooled_ens = (
        pooled.groupby(["symbol", "date_dt", "fold", "seed"])[["prob", "y"]]
        .agg({"prob": "mean", "y": "first"})
        .reset_index()
    )

    v1 = v1_results["oof_panel"].copy()
    v1["date_dt"] = pd.to_datetime(v1["date"])
    v1_ens = v1.groupby(["symbol", "date_dt", "fold", "seed"])[["ensemble", "y"]].agg(
        {"ensemble": "mean", "y": "first"}
    ).reset_index()

    def _auc_for_pair(p_df, v_df):
        merged = pd.merge(
            p_df[["symbol", "date_dt", "prob"]],
            v_df[["symbol", "date_dt", "y", "ensemble"]],
            on=["symbol", "date_dt"], how="inner",
        )
        if merged.empty or merged["y"].nunique() < 2:
            return None
        from sklearn.metrics import roc_auc_score
        try:
            return float(roc_auc_score(merged["y"].astype(int).values, merged["prob"].values)) - \
                   float(roc_auc_score(merged["y"].astype(int).values, merged["ensemble"].values))
        except ValueError:
            return None

    detail: dict[str, Any] = {}

    # Fold robustness: share of folds where pooled beats V1
    fold_deltas = {}
    for fold in sorted(set(pooled_ens["fold"]) | set(v1_ens["fold"])):
        d = _auc_for_pair(pooled_ens[pooled_ens["fold"] == fold], v1_ens[v1_ens["fold"] == fold])
        if d is not None:
            fold_deltas[int(fold)] = d
    n_fold_wins = sum(1 for d in fold_deltas.values() if d > 0)
    fold_robust = bool(fold_deltas) and (n_fold_wins / len(fold_deltas)) >= 0.6
    detail["fold_deltas"] = fold_deltas
    detail["fold_robust"] = fold_robust
    detail["fold_win_ratio"] = (n_fold_wins / len(fold_deltas)) if fold_deltas else None

    # Seed stability: sign of pooled-vs-V1 delta is consistent across seeds
    seed_deltas = {}
    for seed in sorted(set(pooled_ens["seed"]) | set(v1_ens["seed"])):
        d = _auc_for_pair(pooled_ens[pooled_ens["seed"] == seed], v1_ens[v1_ens["seed"] == seed])
        if d is not None:
            seed_deltas[int(seed)] = d
    signs = {1 if d > 0 else -1 for d in seed_deltas.values() if d is not None}
    seed_stable = bool(signs) and len(signs) == 1 and len(seed_deltas) >= 2
    detail["seed_deltas"] = seed_deltas
    detail["seed_stable"] = seed_stable

    return fold_robust, detail


def _compute_comparison(
    matched: pd.DataFrame,
    v1_results: dict[str, Any],
    pooled_results_list: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute all comparison metrics from the matched sample."""
    v1_probs = matched["v1_ensemble"].values
    v1_y = matched["y"].values

    logger.info("Matched sample: %d rows, %d symbols, %d dates",
                len(matched), matched["symbol"].nunique(), matched["date"].nunique())
    if len(matched):
        logger.info("Matched y classes: %s; n_up=%d n_down=%d",
                    sorted(set(v1_y.tolist())), int((v1_y == 1).sum()), int((v1_y == 0).sum()))

    v1_metrics = compute_binary_metrics(v1_y, v1_probs)
    logger.info("V1 global AUC: %s (n=%s)", v1_metrics.get("roc_auc"), v1_metrics.get("n"))

    # Per-model comparison
    model_comparison: dict[str, Any] = {}
    for pr in pooled_results_list:
        config_name = pr["feature_config"]
        for model_name in pr["usable_models"]:
            col = f"pooled_{config_name}_{model_name}"
            if col not in matched.columns:
                continue
            pooled_probs = matched[col].values
            pooled_metrics = compute_binary_metrics(v1_y, pooled_probs)

            delta_auc = None
            if v1_metrics["roc_auc"] is not None and pooled_metrics["roc_auc"] is not None:
                delta_auc = pooled_metrics["roc_auc"] - v1_metrics["roc_auc"]

            key = f"{config_name}_{model_name}"
            model_comparison[key] = {
                "v1_auc": v1_metrics["roc_auc"],
                "pooled_auc": pooled_metrics["roc_auc"],
                "delta_auc": delta_auc,
                "v1_brier": v1_metrics["brier"],
                "pooled_brier": pooled_metrics["brier"],
                "delta_brier": (pooled_metrics["brier"] - v1_metrics["brier"]) if pooled_metrics["brier"] is not None and v1_metrics["brier"] is not None else None,
                "v1_ece": v1_metrics["ece"],
                "pooled_ece": pooled_metrics["ece"],
                "pooled_metrics": pooled_metrics,
            }

    # Per-symbol analysis
    per_symbol: dict[str, Any] = {}
    for symbol in matched["symbol"].unique():
        sym_data = matched[matched["symbol"] == symbol]
        v1_sym = compute_binary_metrics(sym_data["y"].values, sym_data["v1_ensemble"].values)
        sym_entry = {"v1_auc": v1_sym["roc_auc"], "n_rows": len(sym_data)}
        sym_deltas: list[float] = []
        for pr in pooled_results_list:
            config_name = pr["feature_config"]
            for model_name in pr["usable_models"]:
                col = f"pooled_{config_name}_{model_name}"
                if col in sym_data.columns:
                    p_sym = compute_binary_metrics(sym_data["y"].values, sym_data[col].values)
                    sym_entry[f"pooled_{config_name}_{model_name}_auc"] = p_sym["roc_auc"]
                    delta = (
                        p_sym["roc_auc"] - v1_sym["roc_auc"] if v1_sym["roc_auc"] is not None and p_sym["roc_auc"] is not None else None
                    )
                    sym_entry[f"delta_{config_name}_{model_name}"] = delta
                    if delta is not None:
                        sym_deltas.append(delta)
        # Symbol-level pooled vs V1 outcome: median pooled delta across configs/models
        sym_median_delta = float(np.median(sym_deltas)) if sym_deltas else None
        sym_entry["median_delta"] = sym_median_delta
        per_symbol[symbol] = sym_entry

    # Win/loss/ties — per-symbol (cross-symbol evidence per the predeclared rule)
    v1_wins = 0
    pooled_wins = 0
    ties = 0
    for symbol, sym_entry in per_symbol.items():
        sym_delta = sym_entry.get("median_delta")
        if sym_delta is None:
            continue
        if sym_delta > 0:
            pooled_wins += 1
        elif sym_delta < 0:
            v1_wins += 1
        else:
            ties += 1

    # Median deltas
    deltas = [mc["delta_auc"] for mc in model_comparison.values() if mc["delta_auc"] is not None]
    median_delta = float(np.median(deltas)) if deltas else None

    # Bootstrap CI
    paired_deltas = np.array([mc["delta_auc"] for mc in model_comparison.values() if mc["delta_auc"] is not None])
    ci = bootstrap_ci(paired_deltas) if len(paired_deltas) > 0 else {}

    # All-long diagnostics
    all_long_pct = float((matched["v1_ensemble"] >= 0.5).mean())

    # Chronological (fold) robustness and seed stability
    fold_robust, rob_detail = _fold_and_seed_robustness(v1_results, pooled_results_list)
    seed_stable = rob_detail["seed_stable"]

    return {
        "v1_metrics": v1_metrics,
        "model_comparison": model_comparison,
        "per_symbol": per_symbol,
        "robustness": rob_detail,
        "summary": {
            "n_symbols_total": len(per_symbol),
            "n_symbols_pooled_wins": pooled_wins,
            "n_symbols_v1_wins": v1_wins,
            "n_ties": ties,
            "median_auc_delta": median_delta,
            "mean_auc_delta": float(np.mean(deltas)) if deltas else None,
            "bootstrap_ci": ci,
            "all_long_pct_v1": all_long_pct,
            "fold_robustness": fold_robust,
            "seed_stable": seed_stable,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Task 17: V1 vs Pooled Fair Comparison")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test with 4 symbols")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols")
    parser.add_argument("--output-dir", type=str, default="ml_v2_fair_compare", help="Output directory")
    parser.add_argument("--seeds", type=str, default="17,31,53", help="Comma-separated seeds")
    args = parser.parse_args()

    seeds = tuple(int(s) for s in args.seeds.split(","))

    if args.smoke:
        symbols = ("TCS", "RELIANCE", "HDFCBANK", "INFY")
    elif args.symbols:
        symbols = tuple(s.strip() for s in args.symbols.split(","))
    else:
        symbols = DEFAULT_UNIVERSE

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    results = run_comparison(symbols=symbols, seeds=seeds)

    # Save results
    here = Path(__file__).resolve() if "__file__" in globals() else Path.cwd()
    out_dir = here.parents[2] / args.output_dir if "__file__" in globals() else here / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON result
    result_path = out_dir / "task17_result.json"
    json_result = {k: v for k, v in results.items() if k not in ("panel",)}
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(json_result, f, indent=2, default=str)

    # Save panel CSV
    panel_path = out_dir / "task17_shared_snapshot.csv"
    results["panel"].to_csv(panel_path, index=False)

    logger.info("Results saved to %s", result_path)
    logger.info("Snapshot saved to %s", panel_path)
    logger.info("Decision: %s", results["decision"])


if __name__ == "__main__":
    main()
