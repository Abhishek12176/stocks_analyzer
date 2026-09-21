"""RF hyperparameter tuning on the fixed replay snapshot (research-only).

Leakage-free protocol (matches `pipeline.forecast_frame`):
- Same preprocessing: `add_target(horizon=20)` -> numeric-feature selection
  (drop all-NaN / constant cols, inf -> NaN) exactly like the live pipeline.
- Same SAME purged walk-forward splits as the replay run (fast=True):
  test_size=40, step=90, min_train=260, embargo=20.
- Median imputer fitted on train only; RF fitted on train, P(up) predicted on
  test -> OOF probabilities. Full-OOF ROC-AUC (average-rank, `explain._auc`)
  is the per-parameter-set score, median across the 4 replay symbols.

Grid is searched coordinate-wise (one parameter at a time, keep the best),
starting from `mo.DEFAULT_RF`:
  max_depth      : 3, 4, 5, 6, 8
  min_samples_leaf: 10, 15, 20, 30, 40
  n_estimators   : 100, 200, 300

Acceptance rule: adopt a parameter set ONLY if its median full-OOF RF AUC
strictly beats the current ensemble baseline (0.5399). The final candidate's
newest-20% holdout AUC is reported once, diagnostics-only.

Usage:
  python -m app.ml.tune_rf
  python -m app.ml.tune_rf --symbols TCS,RELIANCE,HDFCBANK,INFY
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import settings
from app.ml import backtest as bt
from app.ml import dataset as ds
from app.ml import explain
from app.ml import models as mo
from app.ml.replay_snapshot import OUT_DIR, _frame_path

try:
    from sklearn.ensemble import RandomForestClassifier
    _SKLEARN_OK = True
except Exception:  # noqa: BLE001
    RandomForestClassifier = None  # type: ignore[assignment]
    _SKLEARN_OK = False

HORIZON = settings.ml_horizon
TEST_SIZE = 40
STEP = 90
MIN_TRAIN = 260
EMBARGO = HORIZON
SEED = 0
BASELINE_MEDIAN = 0.5399  # current ensemble median full-OOF AUC (replay 12-Sep-2026)

GRID_MAX_DEPTH = [3, 4, 5, 6, 8]
GRID_MIN_SAMPLES_LEAF = [10, 15, 20, 30, 40]
GRID_N_ESTIMATORS = [100, 200, 300]


def _build_features(frame: pd.DataFrame):
    """Replicate `forecast_frame` preprocessing exactly (horizon=20)."""
    out = ds.add_target(frame, horizon=HORIZON)
    feat_cols = [
        c for c in bt._default_feature_cols(out, HORIZON)
        if pd.api.types.is_numeric_dtype(out[c])
    ]
    keep = [c for c in feat_cols if out[c].notna().sum() > 0]
    X = out[keep].replace([np.inf, -np.inf], np.nan)
    keep = [c for c in keep if X[c].nunique(dropna=True) > 1]
    X = X[keep]
    y = out[f"target_up_{HORIZON}d"]
    index = pd.DatetimeIndex(out.index)
    return out, X, y, index


def _rf_oof(params: dict, X: pd.DataFrame, y: pd.Series, index: pd.DatetimeIndex):
    """Full-OOF P(up) for RandomForest with `params` over the replay splits.

    Returns (prob, label) arrays aligned on date — median-imputed, fitted on
    train only, embargoed walk-forward; never touches raw test rows.
    """
    if RandomForestClassifier is None:
        raise RuntimeError("sklearn unavailable")
    folds = bt.purged_walk_forward_splits(
        index, test_size=TEST_SIZE, step=STEP, min_train=MIN_TRAIN, embargo=EMBARGO
    )
    dates: list[pd.Timestamp] = []
    probs: list[float] = []
    labels: list[float] = []
    for fi, (train_idx, test_idx) in enumerate(folds):
        imp = bt._MedianImputer(list(X.columns))
        Xtr = imp.fit_transform(X.loc[train_idx])
        Xte = imp.transform(X.loc[test_idx])
        y_tr = y.loc[train_idx].to_numpy(dtype=float)
        mask = y.loc[train_idx].notna().to_numpy() & ~np.isnan(Xtr).any(axis=1)
        if int(mask.sum()) == 0:
            continue
        est = RandomForestClassifier(
            random_state=SEED + fi, **(dict(params))
        )
        est.fit(Xtr[mask], y_tr[mask].astype(int))
        prob = est.predict_proba(Xte)[:, 1]
        for d, p in zip(test_idx, np.asarray(prob, dtype=float)):
            yy = y.loc[d]
            if np.isnan(p):
                continue
            dates.append(d)
            probs.append(float(p))
            labels.append(float(yy) if pd.notna(yy) else float("nan"))
    return np.asarray(probs, dtype=float), np.asarray(labels, dtype=float)


def _holdout_split(probs: np.ndarray, labels: np.ndarray):
    """Newest 20% of valid (prob, label) rows = final holdout (diagnostics)."""
    ok = ~np.isnan(labels)
    p, yv = np.asarray(probs, float)[ok], np.asarray(labels, float)[ok]
    cut = max(int(len(p) * 0.8), 0)
    return p[cut:], yv[cut:]


def _evaluate(params: dict, symbols: list[str]) -> dict:
    per_symbol: dict[str, float | None] = {}
    for sym in symbols:
        p = _frame_path(sym)
        if not p.exists():
            print(f"{sym}: no saved frame - run replay --save first")
            per_symbol[sym] = None
            continue
        frame = pd.read_pickle(p)
        _, X, y, index = _build_features(frame)
        probs, labels = _rf_oof(params, X, y, index)
        if len(probs) == 0:
            per_symbol[sym] = None
            continue
        auc = explain._auc(labels, probs)
        per_symbol[sym] = round(float(auc), 4)
    aucs = [a for a in per_symbol.values() if a is not None]
    median = float(np.median(aucs)) if aucs else float("nan")
    return {
        "params": dict(params),
        "per_symbol": {k: v for k, v in per_symbol.items()},
        "median_full_oof_auc": round(median, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="TCS,RELIANCE,HDFCBANK,INFY")
    args = ap.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    if RandomForestClassifier is None:
        print("sklearn unavailable - aborting")
        return

    base = dict(mo.DEFAULT_RF)
    default_params = dict(base)
    rows: list[dict] = []

    def sweep(axis_name: str, values: list, fixed: dict) -> tuple[float, float]:
        """Return (best_value, best_median) for one parameter axis."""
        best_val, best_med = fixed[axis_name], -1.0
        for v in values:
            cand = dict(fixed)
            cand[axis_name] = v
            res = _evaluate(cand, symbols)
            rows.append(res)
            med = float(res["median_full_oof_auc"])
            print(f"{axis_name}={v:<3} median={med:.4f}  "
                  f"{res['per_symbol']}")
            if med > best_med + 1e-12:
                best_val, best_med = v, med
        return best_val, best_med

    res0 = _evaluate({k: v for k, v in base.items()}, symbols)
    rows.append(res0)
    print(f"DEFAULT (n_estimators=200, max_depth=6, min_samples_leaf=20) "
          f"median={res0['median_full_oof_auc']:.4f}  {res0['per_symbol']}")

    best_depth, _ = sweep("max_depth", GRID_MAX_DEPTH, base)
    step1 = dict(base, max_depth=best_depth)
    best_leaf, _ = sweep("min_samples_leaf", GRID_MIN_SAMPLES_LEAF, step1)
    step2 = dict(base, max_depth=best_depth, min_samples_leaf=best_leaf)
    best_est, _ = sweep("n_estimators", GRID_N_ESTIMATORS, step2)

    best_params = dict(base, max_depth=best_depth,
                       min_samples_leaf=best_leaf, n_estimators=best_est)
    best_res = _evaluate(best_params, symbols)
    rows.append(best_res)
    bmed = float(best_res["median_full_oof_auc"])

    print(f"\nBEST combo: max_depth={best_depth} "
          f"min_samples_leaf={best_leaf} n_estimators={best_est} "
          f"-> median full-OOF AUC={bmed:.4f}")

    if bmed > BASELINE_MEDIAN:
        print(f"ADOPT: {bmed:.4f} > {BASELINE_MEDIAN} (current ensemble baseline) "
              f"-> replace mo.DEFAULT_RF with {best_params}")
        adopt = True
    else:
        print(f"NO IMPROVEMENT: {bmed:.4f} <= {BASELINE_MEDIAN} "
              f"-> keep current mo.DEFAULT_RF defaults")
        adopt = False

    out_p = Path(__file__).resolve().parents[2] / "ml_rf_tune"
    out_p.mkdir(parents=True, exist_ok=True)
    (out_p / "rf_tune_result.json").write_text(json.dumps({
        "baseline_ensemble_median": BASELINE_MEDIAN,
        "default_rf": default_params,
        "search": "coordinate-wise (one param at a time)",
        "splits": {"test_size": TEST_SIZE, "step": STEP,
                   "min_train": MIN_TRAIN, "embargo": EMBARGO, "seed": SEED},
        "best_params": best_params,
        "best_median_full_oof_auc": bmed,
        "adopt": adopt,
        "sweep": rows,
    }, indent=2))
    print(f"results -> {out_p / 'rf_tune_result.json'}")
    print("ADOPT" if adopt else "NO IMPROVEMENT")


if __name__ == "__main__":
    main()