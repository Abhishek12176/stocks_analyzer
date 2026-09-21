"""Research-only: per-symbol + pooled probability BLEND probe (Task 22).

Follow-up to Task 21 (vol-adjusted pooled LOO, NO ADOPT).  Here we blend the two
full-OOF P(up) vectors that already exist on the Task 17 20-symbol panel:

    P_blend = w * P_pooled + (1 - w) * P_per_symbol

- per-symbol vector : locked ensemble (logistic + rf) full-OOF P(up) per test date.
- pooled vector     : leave-one-symbol-out rf (locked 300/8/30) on the SAME fold
                      windows, pool = other symbols' rows up to each fold's embargo
                      cut (held-out symbol never trained).
- both vectors are re-derived per symbol, aligned by DATE (same purged splits,
  so test dates coincide by construction), then blended on the inner join.
- variant ``drop_levels`` = raw OHLC levels removed (Task 21 winner on core);
  ``keep_levels`` reported for robustness.

Acceptance (report-only; no production change):
  median full-OOF AUC over NEW3 (ICICIBANK/SBIN/ITC) > 0.52   AND
  median full-OOF AUC over CORE4 (TCS/RELIANCE/HDFCBANK/INFY) >= 0.55.

Usage:
  python -m app.ml.v2_pooled_blend --symbols TCS,RELIANCE,HDFCBANK,INFY,ICICIBANK,SBIN,ITC
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from app.ml import backtest as bt
from app.ml import explain
from app.ml import pipeline
from app.ml import v2_pooled_vol_adj as v2

HORIZON = v2.HORIZON
SEED_BASE = v2.SEED_BASE
WEIGHTS = (0.3, 0.5, 0.7)
CORE4 = ("TCS", "RELIANCE", "HDFCBANK", "INFY")
NEW3 = ("ICICIBANK", "SBIN", "ITC")
OUT_DIR = Path(__file__).resolve().parents[2] / "ml_v2_pooled_vol_adj"


def _per_symbol_oof_probs(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    models: tuple[str, ...] = ("logistic", "rf"),
) -> pd.Series:
    """Per-symbol full-OOF P(up) per test date (same protocol as Task 21)."""
    out = frame.copy()
    if "date" in out.columns:
        out = out.set_index("date").sort_index()
    X = v2._slim_numeric(out, feature_cols)
    y_train = out[f"train_up_{HORIZON}d"]
    folds = bt.purged_walk_forward_splits(
        pd.DatetimeIndex(out.index), v2.TEST_SIZE, v2.STEP,
        min_train=v2.MIN_TRAIN, embargo=v2.EMBARGO,
    )
    values: dict[pd.Timestamp, float] = {}
    for fi, (train_idx, test_idx) in enumerate(folds):
        combined, _ = pipeline._fold_ensemble_probs(
            X, y_train, train_idx, test_idx, list(models), seed=SEED_BASE + fi
        )
        for date, cv in zip(test_idx, np.asarray(combined, dtype=float)):
            if not np.isnan(cv):
                values[pd.Timestamp(date)] = float(cv)
    if not values:
        return pd.Series(dtype=float)
    series = pd.Series(values)
    series.index = pd.DatetimeIndex(list(values))
    return series.sort_index()


def _pooled_loo_probs(
    panel: pd.DataFrame,
    held_out: str,
    feature_cols: Sequence[str],
    model: str = "rf",
) -> pd.Series:
    """Held-out symbol's pooled full-OOF P(up) per test date (same as Task 21)."""
    X_all = v2._slim_numeric(panel, feature_cols)
    y_all = panel[f"train_up_{HORIZON}d"]
    dates = pd.DatetimeIndex(panel["date"])
    symbols = panel["symbol"].to_numpy()
    held = panel["symbol"] == held_out
    held_dates = pd.DatetimeIndex(panel.loc[held, "date"].to_numpy())
    folds = bt.purged_walk_forward_splits(
        held_dates, v2.TEST_SIZE, v2.STEP, min_train=v2.MIN_TRAIN, embargo=v2.EMBARGO,
    )
    pos = np.arange(len(panel))
    values: dict[pd.Timestamp, float] = {}
    for fi, (train_idx, test_idx) in enumerate(folds):
        cutoff = pd.Timestamp(max(train_idx))
        pool_mask = (~held) & (dates <= cutoff)
        test_dates = pd.DatetimeIndex(test_idx)
        test_positions = np.where(
            (symbols == held_out)
            & np.isin(dates.to_numpy().astype("datetime64[ns]"),
                      test_dates.to_numpy().astype("datetime64[ns]"))
        )[0]
        if len(test_positions) == 0:
            continue
        cv = bt._fit_predict_with_median(
            model, pos[pool_mask.to_numpy()], pos[test_positions],
            X_all, y_all, seed=SEED_BASE + fi, model_kwargs={},
        )
        for p, cvv in zip(test_positions, cv):
            if not np.isnan(cvv):
                values[pd.Timestamp(dates[p])] = float(cvv)
    if not values:
        return pd.Series(dtype=float)
    series = pd.Series(values)
    series.index = pd.DatetimeIndex(list(values))
    return series.sort_index()


def _blend_auc(
    p_per: pd.Series,
    p_pooled: pd.Series,
    label: pd.Series,
    w: float,
) -> float:
    idx = p_per.index.intersection(p_pooled.index)
    if len(idx) == 0:
        return float("nan")
    blended = w * p_pooled.loc[idx] + (1.0 - w) * p_per.loc[idx]
    values = blended.to_numpy(dtype=float)
    labels = label.reindex(idx).to_numpy(dtype=float)
    valid = ~np.isnan(labels)
    return float(explain._auc(labels[valid], values[valid]))


def _median_of(values: dict[str, float | None], subset: Sequence[str]) -> float:
    present = [v for s, v in values.items()
               if s in subset and v is not None and not np.isnan(v)]
    return round(float(np.median(present)), 4) if present else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="per-symbol + pooled P(up) blend")
    ap.add_argument("--symbols", default=",".join([*CORE4, *NEW3]))
    ap.add_argument("--weights", default="0.3,0.5,0.7",
                    help="comma-separated blend weights on the pooled model")
    args = ap.parse_args()

    symbols = [s.upper() for s in args.symbols.split(",") if s.strip()]
    weights = tuple(float(w) for w in args.weights.split(",") if w.strip())

    panel = v2._add_targets(v2._load_panel(v2.PANEL_CSV))
    feature_cols = v2._feature_columns(panel, drop_levels=False)
    feature_cols_drop = v2._feature_columns(panel, drop_levels=True)

    result: dict[str, Any] = {
        "task": "v2-pooled-blend-1",
        "schema_version": "v2-pooled-blend-result-1",
        "weights": list(weights),
        "blend_formula": "P_blend = w*P_pooled + (1-w)*P_per_symbol",
        "per_symbol_model": "ensemble logistic+rf, locked splits",
        "pooled_model": "leave-one-symbol-out rf (mo.DEFAULT_RF)",
        "eval_label": f"target_up_{HORIZON}d (full binary)",
        "results": {},
    }

    per_auc: dict[str, float] = {}
    pooled_auc: dict[str, dict[str, float]] = {"keep_levels": {}, "drop_levels": {}}
    blend: dict[str, dict[str, dict[str, float]]] = {
        "keep_levels": {}, "drop_levels": {},
    }
    for symbol in symbols:
        frame = panel.loc[panel["symbol"] == symbol].sort_values("date").reset_index(drop=True)
        label_rows = panel["symbol"] == symbol
        label = pd.Series(
            panel.loc[label_rows, f"target_up_{HORIZON}d"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(panel.loc[label_rows, "date"].to_numpy()),
        )
        p_per = _per_symbol_oof_probs(frame, feature_cols)
        per_auc[symbol] = round(
            float(explain._auc(
                label.reindex(p_per.index).to_numpy(dtype=float),
                p_per.to_numpy(dtype=float),
            )), 4)
        for tag, cols in (("keep_levels", feature_cols), ("drop_levels", feature_cols_drop)):
            p_pooled = _pooled_loo_probs(panel, symbol, cols)
            pooled_auc[tag][symbol] = round(float(explain._auc(
                label.reindex(p_pooled.index).to_numpy(dtype=float),
                p_pooled.to_numpy(dtype=float),
            )), 4)
            for w in weights:
                auc = _blend_auc(p_per, p_pooled, label, w)
                blend[tag].setdefault(str(w), {})[symbol] = round(auc, 4) if not np.isnan(auc) else None
        print(f"{symbol}: per={per_auc[symbol]} "
              f"pooled_keep={pooled_auc['keep_levels'][symbol]} "
              f"pooled_drop={pooled_auc['drop_levels'][symbol]}", flush=True)
        result["results"]["per_symbol_full_oof_auc"] = dict(per_auc)
        result["results"]["pooled_full_oof_auc"] = pooled_auc
        result["results"]["blend_full_oof_auc"] = blend
        (OUT_DIR / "task22_blend_checkpoint.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")

    result["results"]["per_symbol_full_oof_auc"] = per_auc
    result["results"]["pooled_full_oof_auc"] = pooled_auc
    result["results"]["blend_full_oof_auc"] = blend
    result["results"]["blend_curve"] = {w: {
        "new3_median": _median_of(blend["drop_levels"][w], NEW3),
        "core4_median": _median_of(blend["drop_levels"][w], CORE4),
    } for w in blend["drop_levels"]}

    acceptance: dict[str, Any] = {}
    for tag in ("drop_levels", "keep_levels"):
        ok_rows = {}
        for w in blend[tag]:
            new3 = _median_of(blend[tag][w], NEW3)
            core4 = _median_of(blend[tag][w], CORE4)
            ok_rows[w] = {"new3_median": new3, "core4_median": core4,
                          "pass": bool(new3 > 0.52 and core4 >= 0.55)}
        acceptance[tag] = ok_rows
        winner = next((w for w in map(str, sort_weights(weights)) if ok_rows[w]["pass"]), None)
        acceptance[tag + "_winner_w"] = winner
    result["acceptance"] = acceptance

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "task22_pooled_blend_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print("\nblend table (drop_levels pooled):")
    for w in blend["drop_levels"]:
        print(f"  w={w}: new3={result['results']['blend_curve'][w]['new3_median']} "
              f"core4={result['results']['blend_curve'][w]['core4_median']}")
    print("acceptance:", json.dumps(acceptance, indent=2))
    print(f"results -> {OUT_DIR / 'task22_pooled_blend_result.json'}")


def sort_weights(weights: Sequence[float]) -> Sequence[float]:
    return tuple(sorted(weights))


if __name__ == "__main__":
    main()