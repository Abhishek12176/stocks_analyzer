"""Snapshot replay harness - clean A/B for the 20-day forecast (test-only).

Saves the exact feature frames for a symbol universe ONCE, then replays
`pipeline.forecast_frame` on those identical frames. This lets you compare
different code versions (git stash A/B) on the SAME input snapshot, removing
the upstream-data confound entirely.

Usage:
  python -m app.ml.replay_snapshot --save --symbols TCS,RELIANCE,HDFCBANK,INFY
  python -m app.ml.replay_snapshot --run  --symbols TCS,RELIANCE,HDFCBANK,INFY
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from app.ml import pipeline

OUT_DIR = Path(__file__).resolve().parents[2] / "ml_replay"
DEFAULT_SYMBOLS = ["TCS", "RELIANCE", "HDFCBANK", "INFY"]


def _out_dir(label: str | None = None) -> Path:
    """Snapshot dir: baseline `ml_replay/` or a labelling subdirectory
    (e.g. `ml_replay_pit_fund/`) so A/B variants never clobber the locked
    baseline snapshots."""
    if label:
        return OUT_DIR.parent / f"ml_replay_{label}"
    return OUT_DIR


def _frame_path(symbol: str, out_dir: Path | None = None) -> Path:
    return (out_dir or OUT_DIR) / f"{symbol.lower()}_features.pkl"


def _meta_path(symbol: str, out_dir: Path | None = None) -> Path:
    return (out_dir or OUT_DIR) / f"{symbol.lower()}_meta.json"


# Research A/B levers (all default OFF; the locked ml_replay/ baseline is never touched):
#   fundamentals_pit  -> historical quarterly PIT fund_* snapshots
#   sector            -> stock-vs-SECTOR relative features (per-symbol sector map)
SECTOR_MAP: dict[str, str] = {
    "TCS": "cnxit",
    "INFY": "cnxit",
    "HDFCBANK": "banknifty",
    "RELIANCE": "nifty50",  # no tight sector fit; keep nifty as the bench
}


def _symbol_sector(sym: str, sector: bool) -> str | None:
    return SECTOR_MAP.get(sym.upper()) if sector else None


def save(symbols: list[str], enrich: bool = False, fundamentals_pit: bool = False,
         sector: bool = False, label: str | None = None, period: str = "5y") -> None:
    out_dir = _out_dir(label)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sym in symbols:
        inp = pipeline._build_symbol_input(
            sym, period=period, enrich=enrich, fundamentals_pit=fundamentals_pit,
            sector_index=_symbol_sector(sym, sector),
        )
        if not inp.get("ok"):
            print(f"{sym}: SKIP ({inp.get('error')})")
            continue
        inp["features"].to_pickle(_frame_path(sym, out_dir))
        _meta_path(sym, out_dir).write_text(
            json.dumps({"as_of": inp["as_of"], "fp": inp["fp_hex"]})
        )
        print(f"{sym}: saved {len(inp['features'])} rows, as_of={inp['as_of']}")


def _fmt(v) -> str:
    """Compact numeric formatting for the console summary (None -> 'n/a')."""
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


def _true_median(values: list[float]) -> float:
    """Proper median: mean of the two middle values for an even count.

    The legacy aggregate used ``sorted(aucs)[len(aucs) // 2]`` — for n=4 that
    picks the 3rd-smallest value (an UPPER median), which systematically
    flatters an even-sized universe. Reported alongside the legacy value so the
    difference is visible instead of hidden.
    """
    s = sorted(float(v) for v in values)
    n = len(s)
    if n == 0:
        return float("nan")
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def _upper_median(values: list[float]) -> float:
    """Legacy convention kept for continuity (upper-middle for even counts)."""
    s = sorted(float(v) for v in values)
    return s[len(s) // 2] if s else float("nan")


def _summarize(res: dict) -> dict:
    b = next((r for r in res.get("backtest", []) if r.get("model") == "ensemble"), {})
    latest = res.get("latest", {})
    disc = res.get("discrimination", {}) or {}
    base = disc.get("holdout_baseline_accuracy") or {}
    eff = disc.get("effective_independent_windows") or {}
    return {
        "auc": b.get("roc_auc"),
        "full_oof_auc": disc.get("full_oof_auc"),
        # Honest accuracy / baseline-relative fields (W1). AUC is a ranking
        # metric; `holdout_accuracy_calibrated` is plain classifier accuracy on
        # the same labelled rows and is what must beat the baselines.
        "holdout_accuracy_calibrated": disc.get("holdout_accuracy_calibrated"),
        "full_oof_accuracy_raw": disc.get("full_oof_accuracy_raw"),
        "holdout_best_baseline_accuracy": disc.get("holdout_best_baseline_accuracy"),
        "holdout_edge_vs_best_baseline_accuracy": disc.get(
            "holdout_edge_vs_best_baseline_accuracy"
        ),
        "holdout_always_up_accuracy": base.get("always_up"),
        "holdout_effective_independent_windows": eff.get("holdout"),
        "oof_effective_independent_windows": eff.get("oof"),
        "cum_ret_pct": round((b.get("cum_return") or 0) * 100, 2),
        "n_trades": b.get("n_trades"),
        "p_up": latest.get("probability"),
        "signal": latest.get("signal"),
    }


def run(symbols: list[str], label: str | None = None,
        threshold_buy_map: dict[str, float] | None = None) -> dict:
    out_dir = _out_dir(label)
    results = {}
    for sym in symbols:
        p = _frame_path(sym, out_dir)
        if not p.exists():
            print(f"{sym}: no saved frame in {out_dir} - run --save first")
            continue
        frame = pd.read_pickle(p)
        res = pipeline.forecast_frame(
            frame, symbol=sym, fast=True, threshold_buy_map=threshold_buy_map,
        )
        if not res.get("is_available"):
            print(f"{sym}: UNAVAILABLE ({res.get('error')})")
            results[sym] = {"error": res.get("error")}
            continue
        s = _summarize(res)
        results[sym] = s
        print(
            f"{sym}: OOF_AUC={_fmt(s['full_oof_auc'])} "
            f"acc={_fmt(s['holdout_accuracy_calibrated'])} "
            f"(always_up={_fmt(s['holdout_always_up_accuracy'])}, "
            f"edge={_fmt(s['holdout_edge_vs_best_baseline_accuracy'])}) "
            f"cumRet%={s['cum_ret_pct']} nTrd={s['n_trades']} "
            f"P(up)={s['p_up']} {s['signal']}"
        )
    return results


def summary(results: dict) -> None:
    """Honest cross-symbol aggregate.

    States the universe size, BOTH median conventions, the per-symbol spread and
    the accuracy-vs-baseline edge — so a "median full-OOF AUC" line can never be
    mistaken for a general accuracy claim (W1).
    """
    aucs = [r.get("full_oof_auc") for r in results.values()
            if r.get("full_oof_auc") is not None]
    key = "full-OOF AUC"
    if not aucs:
        aucs = [r["auc"] for r in results.values() if r.get("auc") is not None]
        key = "holdout AUC"
    if not aucs:
        print("\nAGGREGATE: no symbol produced a usable AUC")
        return

    n = len(aucs)
    print("")
    print(f"AGGREGATE over n={n} SYMBOL(S) — universe size matters, state it always")
    print(f"  median {key} (true)             = {_fmt(_true_median(aucs))}")
    print(f"  median {key} (upper-mid legacy) = {_fmt(_upper_median(aucs))}"
          "   <- `sorted(aucs)[n//2]`; flatters even n")
    print(f"  min / max {key:<21} = {_fmt(min(aucs))} / {_fmt(max(aucs))}")
    print("  NOTE: this is a RANKING metric (AUC), NOT classification accuracy,")
    print("        and OOF rows overlap (20d labels) -> far fewer independent")
    print("        windows than rows; small-n spread is noise-dominated.")

    accs = [r.get("holdout_accuracy_calibrated") for r in results.values()
            if r.get("holdout_accuracy_calibrated") is not None]
    ups = [r.get("holdout_always_up_accuracy") for r in results.values()
           if r.get("holdout_always_up_accuracy") is not None]
    edges = [r.get("holdout_edge_vs_best_baseline_accuracy") for r in results.values()
             if r.get("holdout_edge_vs_best_baseline_accuracy") is not None]
    if accs:
        print("")
        print(f"  median hold-out classifier accuracy (true)     = "
              f"{_fmt(_true_median(accs))}")
        if ups:
            print(f"  median hold-out always-up baseline accuracy    = "
                  f"{_fmt(_true_median(ups))}")
        if edges:
            beats = sum(1 for e in edges if e > 0)
            print(f"  symbols whose accuracy BEATS the best baseline = {beats}/{len(edges)}")
            print(f"  median edge vs best baseline                   = "
                  f"{_fmt(_true_median(edges))}  (negative = worse than a constant)")
    eff = [r.get("holdout_effective_independent_windows") for r in results.values()
           if r.get("holdout_effective_independent_windows") is not None]
    if eff:
        print(f"  hold-out effective independent windows (min..max) = {min(eff)}..{max(eff)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--enrich", action="store_true",
                    help="Include macro/news/options enrichments in saved snapshot")
    ap.add_argument("--fundamentals-pit", action="store_true",
                    help="Attach historical quarterly PIT fundamentals to the saved "
                         "snapshot (research A/B lever; snapshots go to "
                         "ml_replay_pit_fund/ so the locked baseline is untouched)")
    ap.add_argument("--sector", action="store_true",
                    help="Add stock-vs-sector relative features (research A/B lever, "
                         "Task 28; per-symbol sector map: TCS/INFY->cnxit, "
                         "HDFCBANK->banknifty; snapshots go to ml_replay_sector/)")
    ap.add_argument("--period", default="5y",
                    help="OHLCV history window (default 5y; e.g. 10y for the longer-"
                         "history A/B; snapshots go to ml_replay_YYYY/ by period)")
    ap.add_argument("--label", default=None,
                    help="Explicit sub-directory label (overrides period/lever "
                         "auto-label); never use a name that collides with the "
                         "locked baseline ml_replay/")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument(
        "--threshold-map", default=None,
        help="per-symbol BUY-threshold overrides as SYM=T,SYM=T (, separated); "
             "applied in backtest + live probe (Task 31 lever)")
    args = ap.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    tmap: dict[str, float] | None = None
    if args.threshold_map:
        tmap = {}
        for kv in args.threshold_map.split(","):
            if not kv.strip():
                continue
            k, v = kv.strip().split("=")
            tmap[k.strip().upper()] = float(v.strip())
    parts = []
    if args.period != "5y":
        parts.append(args.period)
    if args.fundamentals_pit and not args.enrich:
        parts.append("pit_fund")
    if args.sector and not args.enrich:  # --enrich label is a separate convention
        parts.append("sector")
    label = args.label or ("_".join(parts) or None)
    if args.enrich:
        label = args.label or None
    if args.save:
        save(symbols, enrich=args.enrich, fundamentals_pit=args.fundamentals_pit,
             sector=args.sector, label=label, period=args.period)
    if args.run:
        results = run(symbols, label=label, threshold_buy_map=tmap)
        summary(results)
    if not args.save and not args.run:
        ap.print_help()


if __name__ == "__main__":
    main()