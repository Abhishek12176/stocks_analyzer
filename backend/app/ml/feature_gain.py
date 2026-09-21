"""Per-symbol XGBoost feature-gain diagnostic (test-only).

Loads each saved replay frame and runs a walk-forward XGBoost, averaging
feature importance (gain) across folds. Reports the top features and the
share of the newly-added momentum features, so we can judge whether they
genuinely contribute (and which features to keep/drop for selection).

Usage:
  python -m app.ml.feature_gain --symbols TCS,RELIANCE,HDFCBANK,INFY
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml import backtest as bt
from app.ml import models as mo
from app.ml import dataset as ds

OUT_DIR = Path(__file__).resolve().parents[2] / "ml_replay"
DEFAULT_SYMBOLS = ["TCS", "RELIANCE", "HDFCBANK", "INFY"]
NEW_FEATURES = {
    "dist_52w_high",
    "vol_ratio_20_60",
    "rel_rs_slope_nifty50_20d",
}


def _frame_path(symbol: str) -> Path:
    return OUT_DIR / f"{symbol.lower()}_features.pkl"


def analyze(symbol: str) -> None:
    frame = pd.read_pickle(_frame_path(symbol))
    out = ds.add_target(frame, horizon=20, close_col="Close")
    y = out["target_up_20d"]
    index = pd.DatetimeIndex(out.index)

    feat_cols = bt._default_feature_cols(out, 20)
    keep = [c for c in feat_cols if out[c].notna().sum() > 0]
    X = out[keep].replace([np.inf, -np.inf], np.nan)

    folds = bt.purged_walk_forward_splits(
        index, test_size=40, step=90, min_train=260, embargo=20
    )
    gain: dict[str, float] = {}
    for fi, (tr, te) in enumerate(folds):
        imp = bt._MedianImputer(list(X.columns))
        Xtr = pd.DataFrame(
            imp.fit_transform(X.loc[tr]), columns=X.columns, index=X.loc[tr].index
        )
        ytr = y.loc[tr]
        mask = ytr.notna().to_numpy() & ~np.isnan(Xtr.to_numpy()).any(axis=1)
        est = mo._build_estimator("xgboost", seed=fi)
        est.fit(Xtr[mask], ytr[mask].astype(int))
        for c, v in zip(X.columns, est.feature_importances_):
            gain[c] = gain.get(c, 0.0) + float(v)

    total = sum(gain.values()) or 1.0
    ranked = sorted(gain.items(), key=lambda kv: -kv[1])
    print(f"\n=== {symbol} ===")
    for c, v in ranked[:15]:
        tag = "  <-- NEW" if c in NEW_FEATURES else ""
        print(f"  {c:45s} {v/total:.4f}{tag}")
    new_share = sum(v for c, v in gain.items() if c in NEW_FEATURES) / total
    print(f"  NEW-features total gain share: {new_share:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    args = ap.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    for sym in symbols:
        analyze(sym)


if __name__ == "__main__":
    main()