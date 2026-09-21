"""Clean A/B for GDELT news-sentiment features (RESEARCH ONLY — Task 24).

Baseline = the LOCKED 09-09 replay snapshot + ``pipeline.forecast_frame``
(median full-OOF AUC 0.5703). Variant = the SAME locked frame with ``sent_*``
columns attached from GDELT daily AvgTone (built ON TOP of the locked snapshot,
no fresh ``--save`` pull → no as_of confound, mirrors the cross-sectional
protocol of FILE 11).

Usage:
  python -m app.ml.gdelt_ab --symbols TCS,RELIANCE,HDFCBANK,INFY        # fetch + run
  python -m app.ml.gdelt_ab --symbols TCS,RELIANCE,HDFCBANK,INFY --no-fetch  # cached
  python -m app.ml.gdelt_ab --symbols TCS --coverage                    # coverage only

Strict acceptance: adopt ONLY if the variant median full-OOF AUC is strictly
> 0.5703. Otherwise keep the locked config and report honestly.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.ml import pipeline
from app.ml.gdelt_sentiment import (
    _SAFE_QUERIES,
    add_gdelt_sentiment_features,
    fetch_gdelt_timeline,
)

LOCKED_BASELINE = 0.5703
OUT_DIR = Path(__file__).resolve().parents[2] / "ml_gdelt"
REPLAY_DIR = Path(__file__).resolve().parents[2] / "ml_replay"
SNP_START = datetime(2021, 9, 9, tzinfo=timezone.utc)
SNP_END = datetime(2026, 9, 9, tzinfo=timezone.utc)
DEFAULT_SYMBOLS = ["TCS", "RELIANCE", "HDFCBANK", "INFY"]


def _tone_cache_path(symbol: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR / f"{symbol.lower()}_tone.pkl"


def get_or_fetch_tone(symbol: str, no_fetch: bool) -> pd.Series:
    path = _tone_cache_path(symbol)
    if path.exists():
        tone = pd.read_pickle(path)
        print(f"{symbol}: loaded cached tone ({len(tone)} days)")
        return tone
    if no_fetch:
        print(f"{symbol}: no cached tone and --no-fetch -> SKIP")
        return pd.Series(dtype=float)
    query = _SAFE_QUERIES.get(symbol, symbol)
    tone = fetch_gdelt_timeline(query, SNP_START, SNP_END, probe=True)
    if tone.empty:
        print(f"{symbol}: GDELT returned NO tone (blocked or no matches)")
        return pd.Series(dtype=float)
    tone.to_pickle(path)
    print(f"{symbol}: fetched {len(tone)} days of GDELT tone (query {query})")
    return tone


def load_or_fetch_all(symbols: list[str], no_fetch: bool, retry_minutes: float = 8.0,
                      max_rounds: int = 6) -> dict[str, pd.Series]:
    """Load cached / fetch GDELT tone per symbol, re-probing around the temp IP block.

    GDELT hard-blocks an IP for a while after a burst. Iterate rounds: fetch all
    still-missing symbols; if anything was blocked, cool down and retry. Stops
    early once everything is cached or all rounds are exhausted.
    """
    tones: dict[str, pd.Series] = {}
    for round_no in range(1, max_rounds + 1):
        missing = [s for s in symbols if s not in tones]
        if not missing:
            break
        blocked = False
        for sym in missing:
            tone = get_or_fetch_tone(sym, no_fetch)
            if tone.empty:
                if not no_fetch:
                    blocked = True
                continue
            tones[sym] = tone
        if not blocked or no_fetch:
            break
        print(f"GDELT still throttled after round {round_no} -> cooling down "
              f"{retry_minutes:.0f} min")
        time.sleep(retry_minutes * 60)
    return tones


def _frame_path(symbol: str) -> Path:
    return REPLAY_DIR / f"{symbol.lower()}_features.pkl"


def run(symbols: list[str], no_fetch: bool, window: int,
        retry_minutes: float = 8.0, max_rounds: int = 6) -> dict:
    tones = load_or_fetch_all(symbols, no_fetch=no_fetch,
                              retry_minutes=retry_minutes, max_rounds=max_rounds)
    results: dict[str, dict] = {}
    for sym in symbols:
        p = _frame_path(sym)
        if not p.exists():
            print(f"{sym}: no locked snapshot frame -> SKIP")
            continue
        frame = pd.read_pickle(p)
        tone = tones.get(sym)
        if tone is None or tone.empty:
            results[sym] = {"error": "no_gdelt_tone"}
            print(f"{sym}: no GDELT tone -> SKIP")
            continue

        trading = pd.DatetimeIndex(frame.index)
        present = tone.reindex(trading).dropna()
        cov = len(present) / len(trading) * 100
        print(f"{sym}: news coverage {cov:.1f}% ({len(present)}/{len(trading)} "
              f"trading days), tone range [{present.min():.3f}, {present.max():.3f}], "
              f"n_distinct={int(present.nunique())}")

        base = pipeline.forecast_frame(frame, symbol=sym, fast=True)
        variant = pipeline.forecast_frame(
            add_gdelt_sentiment_features(frame, tone, sym, window=window),
            symbol=sym, fast=True,
        )
        b_auc = base.get("discrimination", {}).get("full_oof_auc")
        v_auc = variant.get("discrimination", {}).get("full_oof_auc")
        results[sym] = {
            "base_full_oof_auc": b_auc,
            "variant_full_oof_auc": v_auc,
            "delta": round(v_auc - b_auc, 4) if v_auc is not None and b_auc is not None else None,
            "coverage_pct": round(cov, 2),
            "sent_cols_kept": [c for c in variant.columns if c.startswith("sent_")],
        }
        print(f"{sym}: base={b_auc} variant={v_auc} "
              f"delta={results[sym]['delta']} coverage={cov:.1f}%")
    return results


def summary(results: dict) -> None:
    deltas = [(r.get("base_full_oof_auc"), r.get("variant_full_oof_auc"))
              for r in results.values()
              if r.get("base_full_oof_auc") is not None and r.get("variant_full_oof_auc") is not None]
    if not deltas:
        print("\nNo comparable runs.")
        return
    base = sorted(a for a, _ in deltas)[len(deltas) // 2]
    var = sorted(b for _, b in deltas)[len(deltas) // 2]
    print(f"\nAGGREGATE (n={len(deltas)}): baseline median full-OOF AUC = {base:.4f} | "
          f"variant median = {var:.4f} | locked baseline = {LOCKED_BASELINE}")
    verdict = "ADOPT" if var > LOCKED_BASELINE else "REJECTED (keep locked config)"
    print(f"ACCEPTANCE (strictly > {LOCKED_BASELINE}): {verdict}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="GDELT %(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--no-fetch", action="store_true", help="use cached GDELT tone only")
    ap.add_argument("--coverage", action="store_true", help="print coverage and exit")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--cooldown-min", type=float, default=8.0,
                    help="minutes to wait between GDELT retry rounds (temp IP block)")
    ap.add_argument("--max-rounds", type=int, default=6)
    args = ap.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    if args.coverage:
        tones = load_or_fetch_all(symbols, args.no_fetch,
                                  retry_minutes=args.cooldown_min,
                                  max_rounds=args.max_rounds)
        for sym in symbols:
            tone = tones.get(sym)
            if tone is None or tone.empty:
                continue
            p = _frame_path(sym)
            if not p.exists():
                continue
            trading = pd.DatetimeIndex(pd.read_pickle(p).index)
            present = tone.reindex(trading).dropna()
            print(f"{sym}: coverage {len(present)}/{len(trading)} = "
                  f"{round(len(present)/len(trading)*100, 2)}%")
        return

    results = run(symbols, args.no_fetch, args.window,
                  retry_minutes=args.cooldown_min, max_rounds=args.max_rounds)
    summary(results)
    var_median = _median_variant(results)
    (OUT_DIR / "gdelt_ab_result.json").write_text(
        json.dumps({"as_of_snapshot": "2026-09-09", "locked_baseline": LOCKED_BASELINE,
                    "results": results,
                    "verdict": "ADOPT" if var_median > LOCKED_BASELINE else "REJECTED"},
                   indent=2), encoding="utf-8")


def _median_variant(results: dict) -> float:
    vals = [r.get("variant_full_oof_auc") for r in results.values()
            if r.get("variant_full_oof_auc") is not None]
    return sorted(vals)[len(vals) // 2] if vals else -1.0


if __name__ == "__main__":
    main()