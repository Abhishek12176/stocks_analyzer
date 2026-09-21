"""Research-only probe: capture per-row OOF detail on the same-day 10y control.

Runs `pipeline.forecast_frame(..., return_oof=True)` on the exact
`ml_replay_control10y/` snapshots (as_of 2026-09-18, same-day as the 15y A/B)
and persists the per-row OOF table + headline metrics to `ml_oof_probes/`.
The `return_oof=True` path changes NOTHING in the metrics — this script also
re-derives the holdout accuracy from the captured rows and checks it equals
the reported `discrimination.holdout_accuracy_calibrated` (parity guard).

Run:  python ml_oof_prober.py --symbols TCS,RELIANCE,HDFCBANK,INFY
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from app.ml import pipeline, replay_snapshot

OUT_DIR = Path(__file__).resolve().parent / "ml_oof_probes"
DEFAULT_SYMBOLS = ["TCS", "RELIANCE", "HDFCBANK", "INFY"]


def _r(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not _finite(f) else round(f, 4)


def _finite(f: float) -> bool:
    import math
    return math.isfinite(f)


def _holdout_accuracy(recs: list[dict]) -> float | None:
    hold = [r for r in recs if r.get("is_holdout")]
    pairs = [r for r in hold if r.get("prob_cal") is not None and r.get("y") is not None]
    if not pairs:
        return None
    hit = sum(1 for r in pairs if (r["prob_cal"] >= 0.5) == (r["y"] >= 0.5))
    return round(hit / len(pairs), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--label", default="control10y",
                    help="snapshot label sub-dir to read (default control10y)")
    args = ap.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for sym in symbols:
        p = replay_snapshot._frame_path(sym, replay_snapshot._out_dir(args.label))
        if not p.exists():
            print(f"{sym}: no frame in {args.label} - run replay --save first")
            sys.exit(1)
        frame = pd.read_pickle(p)
        res = pipeline.forecast_frame(frame, symbol=sym, fast=True, return_oof=True)
        if not res.get("is_available"):
            print(f"{sym}: UNAVAILABLE ({res.get('error')})")
            continue
        recs = res["oof_table"]
        recon = _holdout_accuracy(recs)
        reported = res["discrimination"]["holdout_accuracy_calibrated"]
        parity = "OK" if (recon is not None and reported is not None
                          and abs(recon - reported) < 1e-9) else "MISMATCH"
        out = {
            "symbol": sym,
            "horizon": res["horizon"],
            "full_oof_auc": res["discrimination"]["full_oof_auc"],
            "holdout_accuracy_reported": reported,
            "holdout_accuracy_recon": recon,
            "parity": parity,
            "edge_vs_best_baseline": res["discrimination"][
                "holdout_edge_vs_best_baseline_accuracy"],
            "n_oof": len(recs),
            "n_holdout": sum(1 for r in recs if r.get("is_holdout")),
            "backtest": {
                k: res["backtest"][0].get(k) for k in (
                    "n_trades", "cum_return", "avg_return", "win_rate", "accuracy")
            } if res["backtest"] else None,
            "latest": res["latest"],
            "oof": recs,
        }
        (OUT_DIR / f"{sym.lower()}.json").write_text(
            json.dumps(out, indent=1, default=str)
        )
        print(
            f"{sym}: parity={parity} recon_acc={recon} reported_acc={reported} "
            f"full_oof_auc={res['discrimination']['full_oof_auc']} "
            f"edge={res['discrimination']['holdout_edge_vs_best_baseline_accuracy']} "
            f"n_oof={len(recs)} n_hold={sum(1 for r in recs if r.get('is_holdout'))}"
        )


if __name__ == "__main__":
    main()