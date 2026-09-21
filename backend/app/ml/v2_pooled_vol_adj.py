"""Research-only: vol-adjusted pooled cross-symbol generalization probe (Task 21).

Hypothesis (from the sizeable part of the map): the vol-adjusted TRAINING target
(``train_up_20d``, k=0.5) normalises each symbol by its own volatility, so labels
become cross-symbol comparable and a SINGLE pooled model may generalise to held-out
symbols that per-symbol models currently fail on (ICICIBANK/SBIN/ITC are ~chance
under the locked config).

This module is deliberately research-only with no import path from the live forecast
pipeline.  It reuses the exact locked measurement machinery:

- panel source       : ``ml_v2_fair_compare/task17_shared_snapshot.csv``
                       (Task 17 shared 20-symbol snapshot, as_of 2026-09-09,
                        matched 960 common dates, ALL with the same upstream features;
                        identical panel for per-symbol AND pooled evaluation).
- splits             : ``purged_walk_forward_splits`` test_size=40, step=90,
                       min_train=260, embargo=20  (identical to the locked replay).
- train label        : ``train_up_20d`` (vol-adj, k=0.5, noise rows dropped).
- eval label         : ``target_up_20d`` FULL binary  (AUC directly comparable).
- per-symbol model   : locked ensemble logistic + rf on the per-symbol frame.
- pooled model       : ONE rf (locked ``mo.DEFAULT_RF`` 300/8/30) trained on every
                       other symbol's rows with ``date <= max(purged train dates)``
                       of the held-out fold window (embargo-faithful -> no leakage).
- leakage check      : held-out symbol NEVER contributes training rows; median
                       imputer fit on pooled-train rows only.

Answers produced:
  D1  coverage at k=0.5 per symbol  = rows OUTSIDE the noise band / rows with a
      valid forward target  (fraction of labels that survive the threshold).
  D2  per-symbol full-OOF AUC with raw OHLC levels kept vs dropped  (is the new-
      symbol failure caused by symbol-specific raw levels?  run on all 20, heavy
      column reported).
  P1  pooled leave-one-symbol-out full-OOF AUC (rf, keep levels).
  P2  pooled leave-one-symbol-out full-OOF AUC (rf, drop raw OHLC levels).

Acceptance (research only; adopt ONLY if it clears both):
  median full-OOF AUC over NEW symbols (ICICIBANK/SBIN/ITC) > 0.52   AND
  median full-OOF AUC over CORE symbols (TCS/RELIANCE/HDFCBANK/INFY) >= 0.55.

No production file is modified; nothing is imported from the live path.

Usage:
  python -m app.ml.v2_pooled_vol_adj                  # diagnostics + pooled (20 LOO)
  python -m app.ml.v2_pooled_vol_adj --skip-pooled    # diagnostics + baselines only
  python -m app.ml.v2_pooled_vol_adj --held-out ICICIBANK,SBIN,ITC,TCS,RELIANCE,HDFCBANK,INFY
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.ml import backtest as bt
from app.ml import dataset as ds
from app.ml import explain
from app.ml import models as mo
from app.ml import pipeline

HORIZON = 20
K = 0.5
SEED_BASE = 0
TEST_SIZE = 40
STEP = 90
MIN_TRAIN = 260
EMBARGO = HORIZON

RESTRICTED = {"date", "symbol"}
LEVEL_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
CORE_SYMBOLS = ("TCS", "RELIANCE", "HDFCBANK", "INFY")
NEW_SYMBOLS = ("ICICIBANK", "SBIN", "ITC")
FOCUS_SYMBOLS = tuple([*CORE_SYMBOLS, *NEW_SYMBOLS])

PANEL_CSV = (
    Path(__file__).resolve().parents[2] / "ml_v2_fair_compare" / "task17_shared_snapshot.csv"
)
OUT_DIR = Path(__file__).resolve().parents[2] / "ml_v2_pooled_vol_adj"


def _load_panel(path: Path | str) -> pd.DataFrame:
    panel = pd.read_csv(path)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel["symbol"] = panel["symbol"].astype(str).str.upper()
    if panel["date"].isna().any():
        raise ValueError("panel contains invalid dates")
    return panel.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)


def _add_targets(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol chronological target computation (add_target + vol-adj k)."""
    parts = []
    for symbol, group in panel.groupby("symbol", sort=True):
        frame = group.sort_values("date").copy()
        out = ds.add_target(frame, horizon=HORIZON)
        out = ds.add_volatility_target(out, horizon=HORIZON, k=K)
        parts.append(out.reset_index(drop=True))
    return pd.concat(parts, ignore_index=True).reset_index(drop=True)


def _feature_columns(panel: pd.DataFrame, drop_levels: bool) -> list[str]:
    excluded = RESTRICTED | {
        f"target_ret_{HORIZON}d", f"target_up_{HORIZON}d",
        f"train_up_{HORIZON}d", f"vol_theta_{HORIZON}d",
    }
    out = []
    for column in panel.columns:
        if column in excluded or not pd.api.types.is_numeric_dtype(panel[column]):
            continue
        if drop_levels and column in LEVEL_COLUMNS:
            continue
        out.append(column)
    return out


def _slim_numeric(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = frame[feature_cols].replace([np.inf, -np.inf], np.nan)
    keep = [c for c in feature_cols if X[c].notna().sum() > 0]
    keep = [c for c in keep if X[c].nunique(dropna=True) > 1]
    return X[keep]


def _coverage(frame: pd.DataFrame) -> dict[str, Any]:
    target = frame[f"target_up_{HORIZON}d"].notna()
    train = frame[f"train_up_{HORIZON}d"].notna()
    total = int(target.sum())
    covered = int((train & target).sum())
    return {
        "target_rows": total,
        "outside_noise_rows": covered,
        "coverage_k05": round(covered / total, 4) if total else None,
    }


def _full_oof_auc_single(
    frame: pd.DataFrame,
    feature_cols: list[str],
    models: tuple[str, ...] = ("logistic", "rf"),
) -> float | None:
    """Per-symbol full-OOF ensemble AUC on the full binary target (locked protocol)."""
    out = frame.copy()
    if "date" in out.columns:
        out = out.set_index("date").sort_index()
    X = _slim_numeric(out, feature_cols)
    y = out[f"target_up_{HORIZON}d"]
    y_train = out[f"train_up_{HORIZON}d"]
    folds = bt.purged_walk_forward_splits(
        pd.DatetimeIndex(out.index), TEST_SIZE, STEP, min_train=MIN_TRAIN,
        embargo=EMBARGO,
    )
    probs: list[float] = []
    labels: list[float] = []
    for fi, (train_idx, test_idx) in enumerate(folds):
        combined, _ = pipeline._fold_ensemble_probs(
            X, y_train, train_idx, test_idx, list(models), seed=SEED_BASE + fi
        )
        for date, cv in zip(test_idx, np.asarray(combined, dtype=float)):
            if np.isnan(cv):
                continue
            yy = y.loc[date]
            probs.append(float(cv))
            labels.append(float(yy) if pd.notna(yy) else float("nan"))
    if not probs:
        return None
    return float(explain._auc(np.asarray(labels, float), np.asarray(probs, float)))


def _pooled_loo_auc(
    panel: pd.DataFrame,
    held_out: str,
    feature_cols: list[str],
    model: str = "rf",
) -> float | None:
    """Full-OOF AUC for one held-out symbol under leave-one-symbol-out pooling.

    Fold boundaries are the held-out symbol's own purged walk-forward splits;
    at each fold the pool = every OTHER symbol's rows with ``date <= max(purged
    train dates)`` (exactly the same 20d embargo semantics as the per-symbol
    path, so no future label can cross the test window).
    """
    X_all = _slim_numeric(panel, feature_cols)
    y_all = panel[f"train_up_{HORIZON}d"]
    y_eval = panel[f"target_up_{HORIZON}d"]
    dates = pd.DatetimeIndex(panel["date"])
    symbols = panel["symbol"].to_numpy()
    held = panel["symbol"] == held_out
    held_dates = pd.DatetimeIndex(panel.loc[held, "date"].to_numpy())
    folds = bt.purged_walk_forward_splits(
        held_dates, TEST_SIZE, STEP, min_train=MIN_TRAIN, embargo=EMBARGO,
    )
    pos = np.arange(len(panel))
    probs: list[tuple[pd.Timestamp, float]] = []
    for fi, (train_idx, test_idx) in enumerate(folds):
        cutoff = pd.Timestamp(max(train_idx))
        pool_mask = (~held) & (dates <= cutoff)
        test_dates = pd.DatetimeIndex(test_idx)
        test_mask_arr = held.to_numpy().copy()
        s_dates = dates.to_numpy()
        test_positions = np.where(
            (symbols == held_out) & np.isin(s_dates.astype("datetime64[ns]"),
                                            test_dates.to_numpy().astype("datetime64[ns]"))
        )[0]
        if len(test_positions) == 0:
            continue
        cv = bt._fit_predict_with_median(
            model, pos[pool_mask.to_numpy()], pos[test_positions],
            X_all, y_all, seed=SEED_BASE + fi, model_kwargs={},
        )
        for p, cvv in zip(test_positions, cv):
            if np.isnan(cvv):
                continue
            probs.append((dates[p], float(cvv)))
    if not probs:
        return None
    p_dates = pd.DatetimeIndex([pair[0] for pair in probs])
    p_vals = np.asarray([pair[1] for pair in probs], dtype=float)
    held_rows = symbols == held_out
    lookup = pd.Series(
        y_eval[held_rows].to_numpy(dtype=float),
        index=pd.DatetimeIndex(dates[held_rows]),
    )
    labels = lookup.loc[p_dates].to_numpy(dtype=float)
    valid = ~np.isnan(labels)
    return float(explain._auc(labels[valid], p_vals[valid]))


def _median_of(values: dict[str, float | None]) -> float | None:
    present = [v for v in values.values() if v is not None]
    return round(float(np.median(present)), 4) if present else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Vol-adjusted pooled cross-symbol probe")
    ap.add_argument("--skip-pooled", action="store_true",
                    help="run diagnostics + per-symbol baselines only")
    ap.add_argument("--held-out", default="",
                    help="comma-separated symbols to leave out for pooled AUC "
                         "(default: all panel symbols)")
    ap.add_argument("--model", default="rf")
    args = ap.parse_args()

    if not PANEL_CSV.exists():
        raise FileNotFoundError(PANEL_CSV)
    panel = _load_panel(PANEL_CSV)
    panel = _add_targets(panel)
    symbols = sorted(panel["symbol"].unique())
    feature_cols = _feature_columns(panel, drop_levels=False)
    feature_cols_drop = _feature_columns(panel, drop_levels=True)

    rows0 = len(panel)
    result: dict[str, Any] = {
        "task": "v2-pooled-vol-adj-1",
        "schema_version": "v2-pooled-vol-adj-result-1",
        "source_panel": {
            "path": str(PANEL_CSV),
            "rows": rows0,
            "symbols": symbols,
            "date_start": panel["date"].min().date().isoformat(),
            "date_end": panel["date"].max().date().isoformat(),
        },
        "split_config": {"test_size": TEST_SIZE, "step": STEP,
                         "min_train": MIN_TRAIN, "embargo": EMBARGO},
        "train_label": f"train_up_{HORIZON}d (vol-adj k={K}, noise dropped)",
        "eval_label": f"target_up_{HORIZON}d (full binary)",
        "levels_dropped": LEVEL_COLUMNS,
        "pooled_model": args.model,
        "results": {},
    }

    # D1 coverage -------------------------------------------------------------
    coverage: dict[str, Any] = {}
    for symbol in symbols:
        coverage[symbol] = _coverage(panel.loc[panel["symbol"] == symbol])
    result["results"]["coverage_k05"] = coverage
    print(f"panel rows={rows0} symbols={len(symbols)}")
    print("D1 coverage(k=0.5): "
          + json.dumps({s: c["coverage_k05"] for s, c in coverage.items()}, sort_keys=True))

    # D2 per-symbol baselines -------------------------------------------------
    base_full: dict[str, float | None] = {}
    base_drop: dict[str, float | None] = {}
    for symbol in symbols:
        frame = panel.loc[panel["symbol"] == symbol].sort_values("date").reset_index(drop=True)
        base_full[symbol] = _full_oof_auc_single(frame, feature_cols)
        if symbol in FOCUS_SYMBOLS:
            base_drop[symbol] = _full_oof_auc_single(frame, feature_cols_drop)
    result["results"]["per_symbol_baseline_full_features"] = base_full
    result["results"]["per_symbol_baseline_drop_levels"] = base_drop
    print("D2 baseline(full): " + json.dumps(
        {s: (round(v, 4) if v is not None else None) for s, v in base_full.items()},
        sort_keys=True))
    print("D2 baseline(drop levels, focus): " + json.dumps(
        {s: (round(v, 4) if v is not None else None) for s, v in base_drop.items()},
        sort_keys=True))

    # P pooled ----------------------------------------------------------------
    pooled_keep: dict[str, float | None] = {}
    pooled_drop: dict[str, float | None] = {}
    if not args.skip_pooled:
        held_out = [s.upper() for s in args.held_out.split(",") if s.strip()] or symbols
        held_out = [s for s in held_out if s in panel["symbol"].unique()]
        for i, symbol in enumerate(held_out, 1):
            pooled_keep[symbol] = _pooled_loo_auc(panel, symbol, feature_cols, model=args.model)
            pooled_drop[symbol] = _pooled_loo_auc(panel, symbol, feature_cols_drop, model=args.model)
            print(f"P[{i}/{len(held_out)}] {symbol}: "
                  f"pooled_keep={pooled_keep[symbol]} pooled_drop={pooled_drop[symbol]}",
                  flush=True)
            result["results"]["pooled_loo_keep_levels"] = dict(pooled_keep)
            result["results"]["pooled_loo_drop_levels"] = dict(pooled_drop)
            _write_checkpoint(result)

    # Summary + acceptance ----------------------------------------------------
    def med(values: dict[str, float | None], subset: tuple[str, ...]) -> float | None:
        return _median_of({s: values.get(s) for s in subset})

    groups: dict[str, dict[str, Any]] = {}
    for name, subset in (("all", tuple(symbols)), ("core4", CORE_SYMBOLS),
                         ("new3", NEW_SYMBOLS)):
        groups[name] = {
            "baseline_full": med(base_full, subset),
            "baseline_drop_levels": med(base_drop, subset),
            "pooled_keep_levels": med(pooled_keep, subset),
            "pooled_drop_levels": med(pooled_drop, subset),
        }
    result["summary_median_by_group"] = groups

    if not args.skip_pooled:
        new_pooled_drop = groups["new3"]["pooled_drop_levels"]
        core_pooled_drop = groups["core4"]["pooled_drop_levels"]
        adopt = bool(
            new_pooled_drop is not None and new_pooled_drop > 0.52
            and core_pooled_drop is not None and core_pooled_drop >= 0.55
        )
        result["acceptance"] = {
            "rule": "new3 pooled median > 0.52 AND core4 pooled median >= 0.55",
            "new3_pooled_drop_median": new_pooled_drop,
            "core4_pooled_drop_median": core_pooled_drop,
            "adopt_pooled_training": adopt,
        }
        print(f"ACCEPTANCE -> new3={new_pooled_drop} core4={core_pooled_drop} "
              f"ADOPT={adopt}")

    _write_checkpoint(result, final=True)
    print(f"results -> {OUT_DIR / 'task21_pooled_voladj_result.json'}")
    print("ADOPT POOLED" if (not args.skip_pooled and result.get("acceptance", {}).get("adopt_pooled_training"))
          else "NO ADOPT / partial" if result.get("summary_median_by_group")
          else "no pooled run")


def _write_checkpoint(result: dict[str, Any], final: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = "task21_pooled_voladj_result.json" if final else "task21_pooled_voladj_checkpoint.json"
    (OUT_DIR / name).write_text(json.dumps(result, indent=2, sort_keys=True, default=str),
                                encoding="utf-8")


if __name__ == "__main__":
    main()