"""Research-only pooled V2 model evaluation.

This module intentionally has no import path from the V1 forecast pipeline.  It
evaluates the frozen ``target_up_20d`` label on a global, chronological panel
using training-fold-only imputation and deterministic purge/embargo gaps.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from app.ml import fingerprint as fp
from app.ml import pooled_dataset
from app.ml.v2_relative_features import build_relative_features, list_relative_features

MODEL_VERSION = "v2-pooled-models-1"
SCHEMA_VERSION = "v2-pooled-model-results-1"
DEFAULT_HORIZON = 20
DEFAULT_SEEDS = (17, 31, 53)
RELATIVE_PREFIXES = ("market_", "sector_", "peer_", "cross_")
MODEL_NAMES = ("logistic", "ridge_probabilistic", "random_forest", "xgboost")


def _dates(panel: pd.DataFrame, date_col: str = "date") -> pd.Series:
    if date_col not in panel:
        raise ValueError(f"panel missing {date_col!r}")
    values = pd.to_datetime(panel[date_col], errors="coerce")
    if values.isna().any():
        raise ValueError("panel contains invalid dates")
    if getattr(values.dt, "tz", None) is not None:
        values = values.dt.tz_localize(None)
    return values.dt.normalize()


def _normalise_panel(
    panel: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    date_col: str = "date",
) -> pd.DataFrame:
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    out = panel.copy()
    dates = _dates(out, date_col)
    if symbol_col not in out:
        out[symbol_col] = "__single_symbol__"
    out[symbol_col] = out[symbol_col].astype(str)
    out[date_col] = dates
    if out[[symbol_col, date_col]].duplicated().any():
        raise ValueError("panel must contain one row per (symbol, date)")
    return out.sort_values([date_col, symbol_col], kind="mergesort").reset_index(drop=True)


def _is_relative(name: str) -> bool:
    return str(name).startswith(RELATIVE_PREFIXES) or "relative" in str(name)


def build_feature_matrices(
    panel: pd.DataFrame,
    *,
    baseline_features: Sequence[str] | None = None,
    relative_features: Sequence[str] | None = None,
    identity: str = "none",
    symbol_col: str = "symbol",
    date_col: str = "date",
) -> dict[str, pd.DataFrame]:
    """Return deterministic baseline and baseline-plus-relative matrices.

    The default identity representation is ``none``.  ``one_hot`` is available
    for a predeclared universe and uses sorted categories from the supplied
    panel, so it cannot silently invent a category at prediction time.
    """
    out = _normalise_panel(panel, symbol_col=symbol_col, date_col=date_col)
    if identity not in {"none", "one_hot"}:
        raise ValueError("identity must be 'none' or 'one_hot'")
    relative = list(relative_features or [
        column for column in out.columns if _is_relative(str(column))
    ])
    missing_relative = [column for column in relative if column not in out]
    if missing_relative:
        raise ValueError(f"missing relative feature columns: {missing_relative}")
    if baseline_features is None:
        candidates = []
        for column in out.columns:
            if column in {symbol_col, date_col} or str(column).startswith("target_"):
                continue
            if not pd.api.types.is_numeric_dtype(out[column]):
                continue
            if column not in relative:
                candidates.append(column)
        baseline = candidates
    else:
        baseline = list(baseline_features)
        missing = [column for column in baseline if column not in out]
        if missing:
            raise ValueError(f"missing baseline feature columns: {missing}")
    if not baseline and not relative:
        baseline = ["__constant__"]

    def matrix(columns: list[str]) -> pd.DataFrame:
        values: dict[str, Any] = {}
        for column in columns:
            if column == "__constant__":
                values[column] = np.zeros(len(out), dtype=float)
            else:
                values[column] = pd.to_numeric(out[column], errors="coerce").to_numpy()
        frame = pd.DataFrame(values, index=out.index)
        if identity == "one_hot":
            symbols = sorted(out[symbol_col].unique())
            encoded = pd.get_dummies(
                pd.Categorical(out[symbol_col], categories=symbols),
                prefix="symbol",
                dtype=float,
            )
            encoded.columns = [str(column) for column in encoded.columns]
            frame = pd.concat([frame, encoded], axis=1)
        return frame.astype(float)

    return {
        "baseline": matrix(list(baseline)),
        "baseline_plus_relative": matrix(list(dict.fromkeys([*baseline, *relative]))),
    }


feature_matrices = build_feature_matrices


def _date_list(values: Iterable[pd.Timestamp]) -> list[str]:
    return [pd.Timestamp(value).date().isoformat() for value in values]


def build_global_panel_splits(
    panel: pd.DataFrame,
    *,
    horizon: int = DEFAULT_HORIZON,
    n_folds: int = 4,
    test_size: int | None = None,
    min_train_dates: int = 60,
    min_train: int | None = None,
    final_holdout_fraction: float = 0.20,
    embargo: int | None = None,
    symbol_col: str = "symbol",
    date_col: str = "date",
) -> dict[str, Any]:
    """Create global-date expanding folds and an untouched newest holdout.

    ``horizon`` and ``embargo`` are counts of global panel dates.  The last
    ``max(horizon, embargo)`` dates before every test window are excluded from
    training.  This is both the forward-label purge and an explicit embargo.
    """
    horizon = int(horizon)
    n_folds = int(n_folds)
    if min_train is not None:
        min_train_dates = int(min_train)
    min_train_dates = int(min_train_dates)
    if horizon <= 0 or n_folds <= 0 or min_train_dates < 1:
        raise ValueError("horizon, n_folds and min_train_dates must be positive")
    fraction = float(final_holdout_fraction)
    if not 0 < fraction < 1:
        raise ValueError("final_holdout_fraction must be between zero and one")
    work = _normalise_panel(panel, symbol_col=symbol_col, date_col=date_col)
    unique = pd.DatetimeIndex(sorted(work[date_col].unique()))
    holdout_n = max(1, int(math.ceil(len(unique) * fraction)))
    if len(unique) <= holdout_n + min_train_dates:
        raise ValueError("panel does not contain enough dates for a final holdout")
    holdout_dates = unique[-holdout_n:]
    pre = unique[:-holdout_n]
    gap = max(horizon, int(embargo if embargo is not None else horizon))
    if len(pre) <= min_train_dates + gap:
        raise ValueError("panel does not contain enough pre-holdout dates")
    available_test_dates = len(pre) - min_train_dates
    if test_size is None:
        test_size = max(1, available_test_dates // n_folds)
    test_size = int(test_size)
    if test_size <= 0:
        raise ValueError("test_size must be positive")
    folds: list[dict[str, Any]] = []
    for fold in range(n_folds):
        test_start_pos = min_train_dates + fold * test_size
        test_end_pos = min(test_start_pos + test_size, len(pre))
        if test_start_pos >= len(pre) or test_end_pos <= test_start_pos:
            break
        candidate_train = pre[:test_start_pos]
        if len(candidate_train) <= gap:
            continue
        train_dates = candidate_train[:-gap]
        purge_dates = candidate_train[-gap:]
        test_dates = pre[test_start_pos:test_end_pos]
        train_mask = work[date_col].isin(train_dates)
        test_mask = work[date_col].isin(test_dates)
        folds.append(
            {
                "fold": fold,
                "train_indices": work.index[train_mask].tolist(),
                "test_indices": work.index[test_mask].tolist(),
                "train_dates": _date_list(train_dates),
                "purge_dates": _date_list(purge_dates),
                "embargo_dates": _date_list(purge_dates),
                "test_dates": _date_list(test_dates),
                "train_end": train_dates[-1].date().isoformat(),
                "test_start": test_dates[0].date().isoformat(),
                "test_end": test_dates[-1].date().isoformat(),
            }
        )
    final_candidate = pre
    final_train = final_candidate[:-gap]
    final_gap = final_candidate[-gap:]
    final_mask = work[date_col].isin(holdout_dates)
    final_train_mask = work[date_col].isin(final_train)
    return {
        "horizon": horizon,
        "embargo": gap,
        "final_holdout_fraction": fraction,
        "folds": folds,
        "final_holdout": {
            "isolated": True,
            "train_indices": work.index[final_train_mask].tolist(),
            "test_indices": work.index[final_mask].tolist(),
            "train_dates": _date_list(final_train),
            "purge_dates": _date_list(final_gap),
            "embargo_dates": _date_list(final_gap),
            "holdout_dates": _date_list(holdout_dates),
            "holdout_start": holdout_dates[0].date().isoformat(),
            "holdout_end": holdout_dates[-1].date().isoformat(),
        },
    }


global_panel_splits = build_global_panel_splits
chronological_global_splits = build_global_panel_splits


def _impute(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)
    medians = train.median(axis=0, numeric_only=True).reindex(train.columns).fillna(0.0)
    return (
        train.fillna(medians).fillna(0.0).to_numpy(dtype=float),
        test.fillna(medians).fillna(0.0).to_numpy(dtype=float),
    )


def _constant_probability(y: np.ndarray, n: int) -> np.ndarray:
    return np.full(n, float(np.mean(y)) if len(y) else 0.5, dtype=float)


def _predict(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    if len(y_train) == 0:
        return np.full(len(x_test), 0.5, dtype=float)
    if len(np.unique(y_train)) < 2:
        return _constant_probability(y_train, len(x_test))
    if model_name == "logistic":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000, random_state=int(seed), solver="lbfgs")
        model.fit(x_train, y_train)
        return np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)
    if model_name == "ridge_probabilistic":
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=1.0)
        model.fit(x_train, y_train)
        return np.clip(np.asarray(model.predict(x_test), dtype=float), 0.0, 1.0)
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=80, max_depth=6, min_samples_leaf=2,
            random_state=int(seed), n_jobs=1,
        )
        model.fit(x_train, y_train.astype(int))
        return np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:
            raise ImportError("xgboost is not installed") from exc
        model = XGBClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, objective="binary:logistic",
            eval_metric="logloss", random_state=int(seed), n_jobs=1,
            verbosity=0,
        )
        model.fit(x_train, y_train.astype(int))
        return np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)
    raise ValueError(f"unknown model {model_name!r}")


def _probability_concentration(probability: np.ndarray) -> dict[str, Any]:
    p = np.asarray(probability, dtype=float)
    if len(p) == 0:
        return {
            "distinct_levels": 0, "near_zero_share": None,
            "near_one_share": None, "near_half_share": None,
        }
    return {
        "distinct_levels": int(np.unique(np.round(p, 12)).size),
        "near_zero_share": float(np.mean(p <= 0.05)),
        "near_one_share": float(np.mean(p >= 0.95)),
        "near_half_share": float(np.mean(np.abs(p - 0.5) <= 0.025)),
    }


def binary_metrics(y_true: Sequence[float], probability: Sequence[float]) -> dict[str, Any]:
    """Compute discrimination, classification, calibration and concentration."""
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, f1_score, precision_score,
        recall_score, roc_auc_score,
    )

    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask].astype(int)
    p = p[mask]
    pred = (p >= 0.5).astype(int)
    if len(y) == 0:
        return {
            "n": 0, "roc_auc": None, "accuracy": None, "precision": None,
            "recall": None, "f1": None, "confusion": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
            "brier": None, "ece": None, "probability_concentration": _probability_concentration(p),
        }
    cm = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    except ValueError:
        auc = None
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    calibration_bins = []
    for left, right in zip(bins[:-1], bins[1:]):
        selected = (p >= left) & ((p < right) if right < 1 else (p <= right))
        count = int(selected.sum())
        if count:
            observed = float(np.mean(y[selected]))
            mean_p = float(np.mean(p[selected]))
            ece += count / len(y) * abs(observed - mean_p)
            calibration_bins.append({
                "left": float(left), "right": float(right),
                "n": count, "mean_probability": mean_p, "observed_rate": observed,
            })
    return {
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "roc_auc": auc,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion": {
            "tn": int(cm[0]), "fp": int(cm[1]), "fn": int(cm[2]), "tp": int(cm[3]),
        },
        "brier": float(np.mean((p - y) ** 2)),
        "ece": float(ece),
        "calibration_bins": calibration_bins,
        "probability_concentration": _probability_concentration(p),
    }


def _evaluate_rows(
    panel: pd.DataFrame,
    feature: pd.DataFrame,
    model_name: str,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    *,
    target_col: str,
    seed: int,
) -> dict[str, Any]:
    train = panel.loc[list(train_indices)]
    test = panel.loc[list(test_indices)]
    train = train.loc[train[target_col].notna()]
    test = test.loc[test[target_col].notna()]
    x_train, x_test = _impute(feature.loc[train.index], feature.loc[test.index])
    y_train = pd.to_numeric(train[target_col], errors="coerce").to_numpy(dtype=float)
    y_test = pd.to_numeric(test[target_col], errors="coerce").to_numpy(dtype=float)
    probability = _predict(model_name, x_train, y_train, x_test, seed)
    per_symbol = {}
    for symbol in sorted(test["symbol"].astype(str).unique()):
        mask = test["symbol"].astype(str).to_numpy() == symbol
        per_symbol[symbol] = binary_metrics(y_test[mask], probability[mask])
    return {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "aggregate": binary_metrics(y_test, probability),
        "per_symbol": per_symbol,
    }


def _baseline_controls(
    panel: pd.DataFrame,
    feature: pd.DataFrame,
    split: Mapping[str, Any],
    *,
    target_col: str,
    seed: int,
) -> dict[str, Any]:
    train = panel.loc[list(split["train_indices"])]
    test = panel.loc[list(split["test_indices"])]
    train = train.loc[train[target_col].notna()]
    test = test.loc[test[target_col].notna()]
    y_train = pd.to_numeric(train[target_col], errors="coerce").to_numpy(dtype=float)
    y_test = pd.to_numeric(test[target_col], errors="coerce").to_numpy(dtype=float)
    prior = float(np.mean(y_train)) if len(y_train) else 0.5
    controls = {
        "train_prior": np.full(len(y_test), prior),
        "always_up": np.ones(len(y_test)),
        "always_down": np.zeros(len(y_test)),
        "shuffled_target_logistic": None,
    }
    if len(y_train) and len(y_test):
        rng = np.random.default_rng(int(seed))
        shuffled = rng.permutation(y_train)
        x_train, x_test = _impute(feature.loc[train.index], feature.loc[test.index])
        try:
            controls["shuffled_target_logistic"] = _predict(
                "logistic", x_train, shuffled, x_test, int(seed)
            )
        except Exception:
            controls["shuffled_target_logistic"] = np.full(len(y_test), prior)
    return {
        name: binary_metrics(y_test, probability)
        for name, probability in controls.items()
        if probability is not None
    }


def evaluate_pooled_models(
    panel: pd.DataFrame,
    *,
    feature_cols: Sequence[str] | None = None,
    baseline_features: Sequence[str] | None = None,
    relative_features: Sequence[str] | None = None,
    target_col: str = "target_up_20d",
    horizon: int = DEFAULT_HORIZON,
    identity: str = "none",
    n_folds: int = 4,
    test_size: int | None = None,
    min_train_dates: int = 60,
    min_train: int | None = None,
    final_holdout_fraction: float = 0.20,
    embargo: int | None = None,
    seed: int = 17,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    models: Sequence[str] = MODEL_NAMES,
) -> dict[str, Any]:
    """Run the predeclared pooled scorecard without model/feature HPO."""
    work = _normalise_panel(panel)
    if target_col not in work:
        raise ValueError(f"panel missing frozen target {target_col!r}")
    if feature_cols is not None:
        baseline_features = list(feature_cols)
    matrices = build_feature_matrices(
        work, baseline_features=baseline_features, relative_features=relative_features,
        identity=identity,
    )
    splits = build_global_panel_splits(
        work, horizon=horizon, n_folds=n_folds, test_size=test_size,
        min_train_dates=min_train_dates, min_train=min_train,
        final_holdout_fraction=final_holdout_fraction,
        embargo=embargo,
    )
    requested = list(models)
    availability: dict[str, Any] = {}
    for model_name in requested:
        if model_name == "xgboost":
            try:
                import xgboost  # noqa: F401
                availability[model_name] = {"installed": True}
            except Exception as exc:
                availability[model_name] = {"installed": False, "reason": str(exc)}
        else:
            availability[model_name] = {"installed": True}
    results: dict[str, Any] = {}
    usable_models = [name for name in requested if availability[name]["installed"]]
    for model_name in usable_models:
        results[model_name] = {}
        for matrix_name, matrix in matrices.items():
            folds = []
            for split in splits["folds"]:
                try:
                    metric = _evaluate_rows(
                        work, matrix, model_name, split["train_indices"], split["test_indices"],
                        target_col=target_col, seed=int(seed) + int(split["fold"]),
                    )
                except Exception as exc:
                    metric = {"error": f"{type(exc).__name__}: {exc}"}
                metric["fold"] = split["fold"]
                metric["test_start"] = split["test_start"]
                metric["test_end"] = split["test_end"]
                folds.append(metric)
            valid_folds = [item for item in folds if "aggregate" in item]
            def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
                if not rows:
                    return {"n": 0}
                total = sum(int(row["aggregate"].get("n", 0)) for row in rows)
                output: dict[str, Any] = {"n": total}
                for key in ("roc_auc", "accuracy", "precision", "recall", "f1", "brier", "ece"):
                    values = [
                        (float(row["aggregate"][key]), int(row["aggregate"]["n"]))
                        for row in rows
                        if row["aggregate"].get(key) is not None
                    ]
                    output[key] = (
                        sum(value * count for value, count in values) / sum(count for _, count in values)
                        if values else None
                    )
                return output
            holdout_split = {
                "train_indices": splits["final_holdout"]["train_indices"],
                "test_indices": splits["final_holdout"]["test_indices"],
            }
            try:
                holdout = _evaluate_rows(
                    work, matrix, model_name, holdout_split["train_indices"],
                    holdout_split["test_indices"], target_col=target_col, seed=int(seed),
                )
            except Exception as exc:
                holdout = {"error": f"{type(exc).__name__}: {exc}"}
            seed_metrics = []
            for run_seed in sorted({int(value) for value in seeds}):
                try:
                    metric = _evaluate_rows(
                        work, matrix, model_name, holdout_split["train_indices"],
                        holdout_split["test_indices"], target_col=target_col, seed=run_seed,
                    )
                    seed_metrics.append({"seed": run_seed, **metric["aggregate"]})
                except Exception as exc:
                    seed_metrics.append({"seed": run_seed, "error": f"{type(exc).__name__}: {exc}"})
            results[model_name][matrix_name] = {
                "folds": folds,
                "aggregate": summarise(valid_folds),
                "final_holdout": {
                    **holdout,
                    "isolated": True,
                    "holdout_start": splits["final_holdout"]["holdout_start"],
                    "holdout_end": splits["final_holdout"]["holdout_end"],
                },
                "seed_sensitivity": seed_metrics,
            }
    controls: dict[str, Any] = {}
    control_feature = matrices["baseline"]
    for split in splits["folds"]:
        controls[str(split["fold"])] = _baseline_controls(
            work, control_feature, split, target_col=target_col,
            seed=int(seed) + int(split["fold"])
        )
    controls["final_holdout"] = _baseline_controls(
        work,
        control_feature,
        {"train_indices": splits["final_holdout"]["train_indices"],
         "test_indices": splits["final_holdout"]["test_indices"]},
        target_col=target_col,
        seed=int(seed),
    )
    feature_comparison = {}
    for model_name, variants in results.items():
        left = variants.get("baseline", {}).get("final_holdout", {}).get("aggregate", {})
        right = variants.get("baseline_plus_relative", {}).get("final_holdout", {}).get("aggregate", {})
        feature_comparison[model_name] = {
            "baseline_holdout": left,
            "baseline_plus_relative_holdout": right,
            "delta_roc_auc": (
                right.get("roc_auc") - left.get("roc_auc")
                if right.get("roc_auc") is not None and left.get("roc_auc") is not None else None
            ),
            "delta_brier": (
                right.get("brier") - left.get("brier")
                if right.get("brier") is not None and left.get("brier") is not None else None
            ),
        }
    config = {
        "target": target_col,
        "target_definition": "frozen V1: Close[T+20]/Close[T]-1 > 0",
        "horizon": int(horizon),
        "identity": identity,
        "identity_policy": "none (no symbol identity; portable pooled model)" if identity == "none"
            else "deterministic sorted one-hot symbol columns",
        "seed": int(seed),
        "seeds": sorted({int(value) for value in seeds}),
        "models_requested": requested,
        "models_available": usable_models,
        "feature_columns": {name: list(matrix.columns) for name, matrix in matrices.items()},
        "no_hpo": True,
        "no_production_integration": True,
    }
    metadata = {
        "model_version": MODEL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fp.fingerprint_object(
            "v2-pooled-model-run",
            {"panel": work, "config": config, "splits": splits},
        ),
        "rows": int(len(work)),
        "symbols": sorted(work["symbol"].astype(str).unique()),
        "date_start": work["date"].min().date().isoformat(),
        "date_end": work["date"].max().date().isoformat(),
    }
    return {
        "metadata": metadata,
        "config": config,
        "availability": availability,
        "splits": splits,
        "results": results,
        "controls": controls,
        "feature_comparison": feature_comparison,
        "limitations": [
            "Metrics are observational research, not a production recommendation.",
            "Global panel rows are dependent across symbols and dates.",
            "No hyperparameter, threshold, calibration, or feature selection search was performed.",
            "V1 production pipeline, API, backtest, calibration, and thresholds are unchanged.",
        ],
    }


run_pooled_model_research = evaluate_pooled_models
run_experiment = evaluate_pooled_models
prepare_feature_matrices = build_feature_matrices
purged_panel_splits = build_global_panel_splits


def run_real_data_experiment(
    *,
    symbols: Iterable[str | Mapping[str, Any]] | None = None,
    output_path: str = "ml_v2_pooled_models/task16_real_data.json",
    period: str = "5y",
    horizon: int = DEFAULT_HORIZON,
    seed: int = 17,
) -> dict[str, Any]:
    """Fetch the Task 13 universe and persist a graceful research report."""
    requested = list(symbols or pooled_dataset.DEFAULT_RESEARCH_UNIVERSE)
    manifest = pooled_dataset.build_universe_manifest(requested)
    try:
        pooled = pooled_dataset.run_pooled_dataset(
            manifest=manifest, period=period, horizon=horizon, seed=seed
        )
        panel = pooled["dataset"]
        if panel.empty:
            raise RuntimeError("upstream returned no usable panel rows")
        try:
            has_sector = any(member.get("sector") for member in manifest.get("members", []))
            relative_input = panel.copy()
            if "sector" not in relative_input.columns:
                # Task 14's no-sector semantics are the whole cross-section.
                relative_input["sector"] = "__all__"
            enriched = build_relative_features(
                relative_input,
                manifest=manifest if has_sector else None,
                horizons=(1, 5, 20),
                min_group_size=3,
            )
        except Exception as exc:
            enriched = panel.copy()
            relative_error = f"{type(exc).__name__}: {exc}"
        else:
            relative_error = None
        result = evaluate_pooled_models(
            enriched, horizon=horizon, min_train_dates=60, n_folds=4,
            seed=seed, relative_features=[
                column for column in list_relative_features((1, 5, 20))
                if column in enriched
            ],
        )
        result["data_source"] = {
            "manifest": manifest,
            "pooled_metadata": pooled["metadata"],
            "pooled_report": pooled["report"],
            "relative_feature_error": relative_error,
        }
    except Exception as exc:
        result = {
            "metadata": {
                "model_version": MODEL_VERSION, "schema_version": SCHEMA_VERSION,
                "fingerprint": fp.fingerprint_object("v2-pooled-model-failure", {
                    "manifest": manifest, "period": period, "horizon": horizon, "seed": seed,
                }),
            },
            "config": {"period": period, "horizon": horizon, "seed": seed},
            "data_source": {
                "manifest": manifest, "symbols_requested": manifest["symbols"],
                "error": f"{type(exc).__name__}: {exc}",
                "graceful_failure": True,
            },
            "results": {}, "controls": {}, "feature_comparison": {},
            "limitations": ["Upstream data was unavailable; no market metrics are claimed."],
        }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    result["result_path"] = str(path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run isolated V2 pooled model research")
    parser.add_argument("--symbols", help="comma-separated Task 13 symbols")
    parser.add_argument("--period", default="5y")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--output-path", default="ml_v2_pooled_models/task16_real_data.json")
    args = parser.parse_args(argv)
    symbols = args.symbols.split(",") if args.symbols else None
    result = run_real_data_experiment(
        symbols=symbols, period=args.period, horizon=args.horizon,
        output_path=args.output_path,
    )
    print(json.dumps({
        "result_path": result["result_path"],
        "fingerprint": result["metadata"].get("fingerprint"),
        "rows": result["metadata"].get("rows"),
        "symbols": len(result["metadata"].get("symbols", [])),
        "error": result.get("data_source", {}).get("error"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
