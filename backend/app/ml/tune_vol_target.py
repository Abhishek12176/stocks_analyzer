"""Volatility-adjusted TRAINING-target tuning (research-only).

Concept: the full binary label (pure sign of the 20d return) is dominated by
tiny noise-level moves. For TRAINING only, drop the noise band around zero:
    sigma_T = 20d realized vol of daily returns up to T (causal, data <= T)
    theta_T = k * sigma_T * sqrt(20)
    train label = 1 if ret > +theta_T, 0 if ret < -theta_T, NaN in-between
Evaluation ALWAYS stays on the full binary `target_up_20d`, so the full-OOF
AUC here is directly comparable to the 0.5465 baseline.

Protocol (identical to the live pipeline walk-forward):
- same preprocessing + feature selection as `pipeline.forecast_frame`
- same purged walk-forward splits (fast=True): test_size=40, step=90,
  min_train=260, embargo=20
- ensemble = logistic + rf (`pipeline.ENSEMBLE_MODELS`, tuned DEFAULT_RF),
  trained per fold on `train_up_20d` (noise rows dropped), combined P(up)
  scored against the FULL binary label
- the harness reuses `pipeline._fold_ensemble_probs` directly, so at k=0 its
  output must reproduce the replay's per-symbol full-OOF AUCs.

Sweep k in {0.25, 0.5, 0.75, 1.0}. Acceptance: adopt the best k ONLY if its
median full-OOF AUC (full binary) is strictly > 0.5465, else "no improvement".

Usage:
  python -m app.ml.tune_vol_target
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
from app.ml import pipeline
from app.ml.replay_snapshot import _frame_path

HORIZON = settings.ml_horizon
SEED = 0
BASELINE_MEDIAN = 0.5465  # current ensemble median full-OOF AUC (replay 12-Sep-2026)
KS = [0.25, 0.5, 0.75, 1.0]


def _full_oof_ensemble_auc(symbol: str, k: float) -> float | None:
    frame = pd.read_pickle(_frame_path(symbol))
    out = ds.add_target(frame, horizon=HORIZON)
    if k and k > 0:
        out = ds.add_volatility_target(out, horizon=HORIZON, k=k)
    y = out[f"target_up_{HORIZON}d"]                      # full binary (eval)
    train_up_col = f"train_up_{HORIZON}d"
    y_train = out[train_up_col] if k > 0 and train_up_col in out.columns else y

    feat_cols = pipeline._default_forecast_cols(out, HORIZON)
    keep = [c for c in feat_cols if out[c].notna().sum() > 0]
    X = out[keep].replace([np.inf, -np.inf], np.nan)
    keep = [c for c in keep if X[c].nunique(dropna=True) > 1]
    X = X[keep]

    index = pd.DatetimeIndex(out.index)
    folds = bt.purged_walk_forward_splits(
        index, test_size=40, step=90, min_train=260, embargo=HORIZON
    )
    probs: list[float] = []
    labels: list[float] = []
    usable = [m for m in pipeline.ENSEMBLE_MODELS]
    for fi, (train_idx, test_idx) in enumerate(folds):
        combined, _ = pipeline._fold_ensemble_probs(
            X, y_train, train_idx, test_idx, usable, seed=SEED + fi
        )
        for d, cv in zip(test_idx, np.asarray(combined, dtype=float)):
            if np.isnan(cv):
                continue
            yy = y.loc[d]
            probs.append(float(cv))
            labels.append(float(yy) if pd.notna(yy) else float("nan"))
    if not probs:
        return None
    return float(explain._auc(np.asarray(labels, float), np.asarray(probs, float)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="TCS,RELIANCE,HDFCBANK,INFY")
    args = ap.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    rows: list[dict] = []
    results: dict[str, float] = {}
    for k in [0.0] + KS:
        per = {sym: _full_oof_ensemble_auc(sym, k) for sym in symbols}
        aucs = [a for a in per.values() if a is not None]
        med = round(float(np.median(aucs)), 4) if aucs else None
        results[str(k)] = med
        rows.append({
            "k": k, "per_symbol": {s: round(v, 4) if v is not None else None
                                   for s, v in per.items()},
            "median_full_oof_auc": med,
        })
        print(f"k={k:<5} median={med}  "
              f"{ {s: (round(v, 4) if v is not None else None) for s, v in per.items()} }")

    best_k = max(KS, key=lambda k: (results[str(k)] is not None, results[str(k)]))
    best_med = results[str(best_k)]
    print(f"\nBEST k={best_k} -> median full-OOF AUC={best_med}")

    if best_med is not None and best_med > BASELINE_MEDIAN:
        print(f"ADOPT: {best_med} > {BASELINE_MEDIAN} -> set pipeline.VOL_ADJ_K = {best_k}")
        adopt = True
    else:
        print(f"NO IMPROVEMENT: {best_med} <= {BASELINE_MEDIAN} -> keep VOL_ADJ_K = 0.0")
        adopt = False

    out_p = Path(__file__).resolve().parents[2] / "ml_vol_target"
    out_p.mkdir(parents=True, exist_ok=True)
    (out_p / "vol_target_tune_result.json").write_text(json.dumps({
        "baseline_median": BASELINE_MEDIAN,
        "sweep": rows,
        "best_k": best_k,
        "best_median_full_oof_auc": best_med,
        "adopt": adopt,
        "eval_label": "target_up_20d (full binary)",
        "train_label": "train_up_20d (vol-thresholded, noise rows dropped)",
    }, indent=2))
    print(f"results -> {out_p / 'vol_target_tune_result.json'}")
    print("ADOPT" if adopt else "NO IMPROVEMENT")


if __name__ == "__main__":
    main()