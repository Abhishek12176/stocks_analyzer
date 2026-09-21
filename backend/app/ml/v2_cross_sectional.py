"""Cross-sectional relative-strength research (V2, research-only).

Adds cross-sectional features from a peer universe to a symbol's frame and
measures whether they improve the full-OOF discrimination vs the V1 baseline
(median full-OOF AUC 0.5351). Production V1 is untouched.

Features (all causal, data <= T only):
- cs_mom_rank_{h}d            : momentum percentile rank of the symbol within
                                the peer universe (0-1), per horizon h
- cs_rel_ret_vs_median_{h}d   : symbol h-day return minus universe-median
- cs_rs_ratio_20d             : symbol close / equal-weight universe index - 1

Usage:
  python -m app.ml.v2_cross_sectional --save --symbols TCS,RELIANCE,HDFCBANK,INFY
  python -m app.ml.v2_cross_sectional --run  --symbols TCS,RELIANCE,HDFCBANK,INFY
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml import pipeline

OUT_DIR = Path(__file__).resolve().parents[2] / "ml_cross_sectional"
DEFAULT_SYMBOLS = ["TCS", "RELIANCE", "HDFCBANK", "INFY"]

# Diversified NIFTY50 large-cap universe (20 names, 9 sectors).
PEER_UNIVERSE = [
    "TCS", "INFY", "RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK",
    "AXISBANK", "HINDUNILVR", "ITC", "MARUTI", "TATAMOTORS", "SUNPHARMA",
    "DRREDDY", "TATASTEEL", "JSWSTEEL", "LT", "BHARTIARTL", "NTPC", "BAJFINANCE",
]
HORIZONS = (10, 20, 60)


def _frame_path(symbol):
    return OUT_DIR / f"{symbol.lower()}_features.pkl"


def _meta_path(symbol):
    return OUT_DIR / f"{symbol.lower()}_meta.json"


def fetch_universe_panel(period="5y"):
    """Fetch aligned Close panel for the peer universe (graceful)."""
    from app.services.data_service import fetch_nse_ohlcv
    closes = {}
    for sym in PEER_UNIVERSE:
        try:
            payload = fetch_nse_ohlcv(sym, period=period)
            if payload.get("is_available"):
                df = pd.DataFrame(payload["history"])
                df.index = pd.to_datetime(df["date"])
                closes[sym] = df["close"].astype(float)
        except Exception:
            continue
    return pd.DataFrame(closes).sort_index()


def add_cross_sectional_features(frame, panel):
    """Attach cross-sectional features to the symbol frame (causal)."""
    out = frame.copy()
    close = out["Close"]
    panel = panel.reindex(out.index)
    uni_idx = panel.mean(axis=1)  # equal-weight universe index
    out["cs_rs_ratio_20d"] = close / uni_idx.replace(0, np.nan) - 1
    for h in HORIZONS:
        t_ret = close.pct_change(h, fill_method=None)
        u_ret = panel.pct_change(h, fill_method=None)
        med = u_ret.median(axis=1)
        out[f"cs_rel_ret_vs_median_{h}d"] = t_ret - med
        combined = u_ret.copy()
        combined["__target"] = t_ret
        out[f"cs_mom_rank_{h}d"] = combined.rank(axis=1, pct=True)["__target"]
    return out


def save(symbols):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = fetch_universe_panel()
    for sym in symbols:
        inp = pipeline._build_symbol_input(sym, period="5y", enrich=False)
        if not inp.get("ok"):
            print(f"{sym}: SKIP ({inp.get('error')})")
            continue
        frame = add_cross_sectional_features(inp["features"], panel)
        frame.to_pickle(_frame_path(sym))
        _meta_path(sym).write_text(
            json.dumps({"as_of": inp["as_of"], "fp": inp["fp_hex"]})
        )
        print(f"{sym}: saved {len(frame)} rows, as_of={inp['as_of']}")


def run(symbols):
    aucs = []
    for sym in symbols:
        p = _frame_path(sym)
        if not p.exists():
            print(f"{sym}: no saved frame - run --save first")
            continue
        frame = pd.read_pickle(p)
        res = pipeline.forecast_frame(frame, symbol=sym, fast=True)
        if not res.get("is_available"):
            print(f"{sym}: UNAVAILABLE ({res.get('error')})")
            continue
        auc = res.get("discrimination", {}).get("full_oof_auc")
        aucs.append(auc)
        print(f"{sym}: full-OOF AUC={auc}")
    if aucs:
        med = sorted(aucs)[len(aucs) // 2]
        print(f"\nAGGREGATE median full-OOF AUC={med:.4f} (baseline 0.5351)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    args = ap.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.save:
        save(symbols)
    if args.run:
        run(symbols)
    if not args.save and not args.run:
        ap.print_help()


if __name__ == "__main__":
    main()