"""RF + logistic re-tuning on the new vol-adjusted training target (research-only).

Production state this harness re-tunes against (locked 12-Sep-2026):
  training label : `train_up_20d`   (volatility-adjusted, k=0.5; noise-zone rows
                   are NaN -> dropped from training only)
  evaluation lbl : `target_up_20d`  (full binary) -- full-OOF AUC comparable
  ensemble       : logistic + rf (`pipeline.ENSEMBLE_MODELS`)
  RF             : `mo.DEFAULT_RF` = (n_estimators=300, max_depth=8,
                   min_samples_leaf=30) -- tuned on the OLD pure-binary target
  logistic       : Pipeline(StandardScaler) + LogisticRegression
                   (max_iter=2000, class_weight="balanced"), default C=1.0
                   (C has NEVER been tuned)
  locked median full-OOF ensemble AUC = 0.5703 (full binary)

The old RF optimum was found on the pure-binary training target and logistic C
has never been tuned; the training distribution changed with the vol-adjusted
label, so both are re-tuned here. Protocol is leakage-free and mirrors the live
replay walk-forward exactly (fast=True):
  - same preprocessing + feature selection as `pipeline.forecast_frame`
  - same purged walk-forward splits: test_size=40, step=90, min_train=260,
    embargo=20
  - per fold: median imputer fit on TRAIN only; RF / logistic fit per fold on
    `train_up_20d`; ensemble P(up) = equal-weight combine of both; OOF probs
    scored against the FULL binary `target_up_20d` (`explain._auc`)
  - coordinate-wise search, keep best, starting from the locked params:
      RF        : max_depth {4,6,8,10} -> min_samples_leaf {20,30,40,50}
                  -> n_estimators {200,300,400}
      logistic  : C {0.01,0.1,1.0,10.0} (inside the StandardScaler pipeline)
  - median uses the same convention as `replay_snapshot.summary`
    (sorted(aucs)[len//2]) so it is directly comparable to the locked 0.5703

Acceptance rule: adopt new params ONLY if the median full-OOF ensemble AUC
(full binary) is strictly > 0.5703; otherwise keep the current params and
report "no improvement". This harness only SEARCHES + reports; the production
edit (mo.DEFAULT_RF / mo.DEFAULT_LOGISTIC_C) is applied only when ADOPT is
real, and must then be re-verified with `replay_snapshot --run`.

Usage:
  python -m app.ml.tune_retune_models
  python -m app.ml.tune_retune_models --symbols TCS,RELIANCE,HDFCBANK,INFY
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
from app.ml import ensemble as en
from app.ml import explain
from app.ml import models as mo
from app.ml import pipeline
from app.ml.replay_snapshot import _frame_path

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_OK = True
except Exception:  # noqa: BLE001
    RandomForestClassifier = None  # type: ignore[assignment]
    LogisticRegression = None  # type: ignore[assignment]
    Pipeline = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]
    _SKLEARN_OK = False

HORIZON = settings.ml_horizon
TEST_SIZE = 40
STEP = 90
MIN_TRAIN = 260
EMBARGO = HORIZON
SEED = 0
BASELINE_MEDIAN = 0.5703  # locked replay median full-OOF ensemble AUC (12-Sep-2026)
VOL_ADJ_K = pipeline.VOL_ADJ_K  # 0.5 (locked)

DEFAULT_LOGISTIC_C = 1.0  # sklearn default currently wired in mo._build_estimator

GRID_RF_MAX_DEPTH = [4, 6, 8, 10]
GRID_RF_MIN_SAMPLES_LEAF = [20, 30, 40, 50]
GRID_RF_N_ESTIMATORS = [200, 300, 400]
GRID_LOGISTIC_C = [0.01, 0.1, 1.0, 10.0]


def _median(vals: list[float]) -> float:
    """Replay `replay_snapshot.summary` convention: sorted(aucs)[len//2]."""
    if not vals:
        return float("nan")
    return float(sorted(vals)[len(vals) // 2])


def _build_rf(params: dict, seed: int):
    return RandomForestClassifier(
        random_state=seed, class_weight="balanced", n_jobs=-1, **(dict(params))
    )


def _build_logistic(c: float, seed: int):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=float(c), max_iter=2000, class_weight="balanced", random_state=seed)),
    ])


def _prepare_frame(symbol: str) -> dict | None:
    """Preprocess one replay frame exactly like the live pipeline (k=0.5)."""
    p = _frame_path(symbol)
    if not p.exists():
        return None
    frame = pd.read_pickle(p)
    out = ds.add_target(frame, horizon=HORIZON)
    if VOL_ADJ_K and VOL_ADJ_K > 0:
        out = ds.add_volatility_target(out, horizon=HORIZON, k=VOL_ADJ_K)
    y_eval = out[f"target_up_{HORIZON}d"]
    y_train = out[f"train_up_{HORIZON}d"]
    feat_cols = pipeline._default_forecast_cols(out, HORIZON)
    keep = [c for c in feat_cols if out[c].notna().sum() > 0]
    X = out[keep].replace([np.inf, -np.inf], np.nan)
    keep = [c for c in keep if X[c].nunique(dropna=True) > 1]
    X = X[keep]
    index = pd.DatetimeIndex(out.index)
    folds = bt.purged_walk_forward_splits(
        index, test_size=TEST_SIZE, step=STEP, min_train=MIN_TRAIN, embargo=EMBARGO
    )
    return {"X": X, "y_eval": y_eval, "y_train": y_train, "folds": folds}


def _ensemble_oof(
    prep: dict, rf_params: dict, logistic_c: float
) -> tuple[np.ndarray, np.ndarray]:
    """Full-OOF ensemble P(up) + full-binary labels for one symbol/param set."""
    X, y_train, y_eval = prep["X"], prep["y_train"], prep["y_eval"]
    probs: list[float] = []
    labels: list[float] = []
    for fi, (train_idx, test_idx) in enumerate(prep["folds"]):
        imp = bt._MedianImputer(list(X.columns))
        Xtr = pd.DataFrame(
            imp.fit_transform(X.loc[train_idx]), columns=X.columns,
            index=X.loc[train_idx].index,
        )
        Xte = pd.DataFrame(
            imp.transform(X.loc[test_idx]), columns=X.columns,
            index=X.loc[test_idx].index,
        )
        yv = y_train.loc[train_idx]
        mask = yv.notna().to_numpy() & ~np.isnan(Xtr).any(axis=1)
        per: dict[str, np.ndarray] = {}
        for name, est in (
            ("rf", _build_rf(rf_params, SEED + fi)),
            ("logistic", _build_logistic(logistic_c, SEED + fi)),
        ):
            if int(mask.sum()) == 0:
                per[name] = np.full(len(Xte), np.nan)
            else:
                est.fit(Xtr[mask], yv.to_numpy(dtype=float)[mask].astype(int))
                per[name] = mo.predict_up(est, Xte)
        combined = en.combine_probs([per["rf"], per["logistic"]])
        for d, cv in zip(test_idx, np.asarray(combined, dtype=float)):
            if np.isnan(cv):
                continue
            yy = y_eval.loc[d]
            probs.append(float(cv))
            labels.append(float(yy) if pd.notna(yy) else float("nan"))
    ok = ~np.isnan(np.asarray(labels, dtype=float))
    return np.asarray(probs, dtype=float)[ok], np.asarray(labels, dtype=float)[ok]


def _evaluate(prep: dict[str, dict], rf_params: dict, logistic_c: float) -> dict:
    per_symbol: dict[str, float | None] = {}
    for sym, p in prep.items():
        probs, labels = _ensemble_oof(p, rf_params, logistic_c)
        per_symbol[sym] = (
            round(float(explain._auc(labels, probs)), 4)
            if len(probs) else None
        )
    aucs = [a for a in per_symbol.values() if a is not None]
    return {
        "rf": dict(rf_params),
        "logistic_c": logistic_c,
        "per_symbol": per_symbol,
        "median_full_oof_auc": round(_median(aucs), 4) if aucs else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="TCS,RELIANCE,HDFCBANK,INFY")
    args = ap.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    if not _SKLEARN_OK:
        print("sklearn unavailable - aborting")
        return

    prep: dict[str, dict] = {}
    for sym in symbols:
        p = _prepare_frame(sym)
        if p is None:
            print(f"{sym}: no saved frame - run replay --save first")
        else:
            prep[sym] = p
    if not prep:
        print("no replay frames available - aborting")
        return

    def current_rf() -> dict:
        return {
            k: mo.DEFAULT_RF[k]
            for k in ("n_estimators", "max_depth", "min_samples_leaf")
        }

    best_rf = current_rf()
    best_c = DEFAULT_LOGISTIC_C
    rows: list[dict] = []

    def eval_candidate(rf: dict, c: float) -> dict:
        res = _evaluate(prep, rf, c)
        rows.append(res)
        med = res["median_full_oof_auc"]
        print(f"rf={rf} logistic_c={c}  median={med}  {res['per_symbol']}")
        return res

    res0 = eval_candidate(best_rf, best_c)
    best_med = float(res0["median_full_oof_auc"] or -1.0)
    print(f"LOCKED params (rf=300/8/30, logistic_c=1.0) "
          f"median={res0['median_full_oof_auc']}")

    for axis, values, kind in (
        ("max_depth", GRID_RF_MAX_DEPTH, "rf"),
        ("min_samples_leaf", GRID_RF_MIN_SAMPLES_LEAF, "rf"),
        ("n_estimators", GRID_RF_N_ESTIMATORS, "rf"),
        ("logistic_c", GRID_LOGISTIC_C, "logistic"),
    ):
        if kind == "rf":
            best_val, best_val_med = best_rf[axis], -1.0
            for v in values:
                cand = dict(best_rf)
                cand[axis] = v
                res = eval_candidate(cand, best_c)
                if (res["median_full_oof_auc"] is not None
                        and float(res["median_full_oof_auc"]) > best_val_med + 1e-12):
                    best_val, best_val_med = v, float(res["median_full_oof_auc"])
            best_rf[axis] = best_val
            best_med = best_val_med if best_val_med > 0 else best_med
            print(f"-> best RF {axis}={best_val} (median={best_val_med:.4f})")
        else:
            best_val, best_val_med = best_c, -1.0
            for v in values:
                res = eval_candidate(best_rf, float(v))
                if (res["median_full_oof_auc"] is not None
                        and float(res["median_full_oof_auc"]) > best_val_med + 1e-12):
                    best_val, best_val_med = v, float(res["median_full_oof_auc"])
            best_c = float(best_val)
            if best_val_med > 0:
                best_med = best_val_med
            print(f"-> best logistic C={best_val} (median={best_val_med:.4f})")

    final = eval_candidate(best_rf, best_c)
    bmed = float(final["median_full_oof_auc"] or -1.0)
    print(f"\nBEST combo rf={best_rf} logistic_c={best_c} -> "
          f"median full-OOF AUC={bmed:.4f}")

    adopt = bool(np.isfinite(bmed)) and bmed > BASELINE_MEDIAN
    if adopt:
        print(f"ADOPT: {bmed:.4f} > {BASELINE_MEDIAN}")
        print(f"  -> set mo.DEFAULT_RF = {dict(best_rf, class_weight='balanced', n_jobs=-1)}")
        print(f"  -> set mo.DEFAULT_LOGISTIC_C = {best_c} "
              f"(wire into mo._build_estimator)")
    else:
        print(f"NO IMPROVEMENT: {bmed:.4f} <= {BASELINE_MEDIAN} "
              f"-> keep current params")

    out_p = Path(__file__).resolve().parents[2] / "ml_retune"
    out_p.mkdir(parents=True, exist_ok=True)
    (out_p / "retune_result.json").write_text(json.dumps({
        "baseline_median": BASELINE_MEDIAN,
        "current_rf": current_rf(),
        "current_logistic_c": DEFAULT_LOGISTIC_C,
        "search": ("coordinate-wise: RF max_depth -> min_samples_leaf -> "
                   "n_estimators -> logistic C"),
        "splits": {"test_size": TEST_SIZE, "step": STEP,
                   "min_train": MIN_TRAIN, "embargo": EMBARGO, "seed": SEED},
        "training_label": f"train_up_{HORIZON}d (vol-adjusted k={VOL_ADJ_K}, "
                          "noise rows dropped)",
        "eval_label": f"target_up_{HORIZON}d (full binary)",
        "median_convention": "sorted(aucs)[n//2] (replay summary convention)",
        "best_rf": best_rf,
        "best_logistic_c": best_c,
        "best_median_full_oof_auc": bmed,
        "adopt": adopt,
        "sweep": rows,
    }, indent=2))
    print(f"results -> {out_p / 'retune_result.json'}")
    print("ADOPT" if adopt else "NO IMPROVEMENT")


if __name__ == "__main__":
    main()