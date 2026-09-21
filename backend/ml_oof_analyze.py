"""Research-only analysis of per-row OOF (ml_oof_probes/*.json).

Experiments (all honest: thresholds are tuned ONLY on the older OOF-fit 80%,
then evaluated on the newest-20% holdout rows, which never touched fitting):

  CONTROL  - current production policy: decide_signal(p, tb=0.60, ts=0.40,
             confidence_floor=0.08, allow_short=False, regime-aware).
  EXP1     - per-symbol BUY threshold tb* selected on fit rows (min coverage
             guard), long-only. Evaluated on hold rows.
  EXP2     - regime-conditional thresholds: separate tb* for BEAR
             (regime_trend==-1 or regime_risk==-1) and NON-BEAR, both selected
             on fit rows, evaluated on hold rows.

Metrics (reported per symbol and as a cross-symbol median, stating universe
and median convention):
  n_long   - number of BUY rows in the holdout
  acc      - fraction of BUY rows whose label y==1 (direction accuracy of BUYs)
  sum_ret  - equal-notional sum of 20d returns over the BUY rows (realised PnL
             proxy; the formal ledger cum_return from the replay is listed too)
  edge     - acc minus the holdout always-up base rate (up_rate)
"""

from __future__ import annotations

import json
from pathlib import Path

from app.ml import signal as sig

PROBES = Path(__file__).resolve().parent / "ml_oof_probes"
SYMBOLS = ["TCS", "RELIANCE", "HDFCBANK", "INFY"]

CONTROL_TB = 0.60
CONTROL_TS = 0.40
FLOOR = 0.08
GRID = [round(0.50 + 0.02 * i, 2) for i in range(12)]  # 0.50..0.72
MIN_COVERAGE_FIT = 50  # preferred trade count on fit rows before a threshold counts


def _policy_dirs(recs: list[dict]) -> tuple[dict, dict]:
    fit = [r for r in recs if not r.get("is_holdout")]
    hold = [r for r in recs if r.get("is_holdout")]
    return dict(fit=fit, hold=hold)


def _regime(r: dict) -> tuple[float | None, bool]:
    t = r.get("regime_trend")
    k = r.get("regime_risk")
    bear = (t == -1) or (k == -1)
    return t, bear


def _dir(p: float | None, tb: float, ts: float, r: dict) -> float:
    if p is None:
        return 0.0
    t, bear = _regime(r)
    res = sig.decide_signal(
        float(p),
        threshold_buy=tb,
        threshold_sell=ts,
        confidence_floor=FLOOR,
        allow_short=False,
        regime_trend=(float(t) if t is not None else None),
        regime_risk=(float(r.get("regime_risk")) if r.get("regime_risk") is not None else None),
    )
    return float(res["direction"])


def _stats(recs: list[dict], direction: list[float], up_rate_all: float | None = None):
    longs = [r for r, d in zip(recs, direction) if d == 1.0]
    n = len(longs)
    if n == 0:
        return dict(n_long=0, acc=None, sum_ret=None, up_rate=None, edge=None)
    ok = [r for r in longs if r.get("y") is not None]
    acc = round(sum(1 for r in ok if r["y"] >= 0.5) / len(ok), 4) if ok else None
    sret = round(sum(float(r["ret"]) for r in longs if r.get("ret") is not None), 4)
    up = round(sum(1 for r in ok if r["y"] >= 0.5) / len(ok), 4) if ok else None
    edge = None
    if acc is not None and up_rate_all is not None:
        edge = round(acc - up_rate_all, 4)
    return dict(n_long=n, acc=acc, sum_ret=sret, up_rate=up, edge=edge)


def _eval(recs: list[dict], tb: float, ts: float, regimemap=None) -> dict:
    """Evaluate a policy on `recs`; `regimemap` = optional {bear: tb, nonbear: tb}."""
    up_rate_all = round(
        sum(1 for r in recs if r.get("y") is not None and r["y"] >= 0.5)
        / sum(1 for r in recs if r.get("y") is not None), 4
    ) if any(r.get("y") is not None for r in recs) else None
    directions = []
    for r in recs:
        if regimemap is not None:
            _, bear = _regime(r)
            tb = regimemap["bear"] if bear else regimemap["nonbear"]
            directions.append(_dir(r.get("prob_cal"), tb, ts, r))
        else:
            directions.append(_dir(r.get("prob_cal"), tb, ts, r))
    s = _stats(recs, directions, up_rate_all)
    s["tb"] = tb
    return s


def _pick(recs: list[dict], is_bear_filter, ts: float) -> float | None:
    """Pick tb* on fit rows: max long-accuracy among grid points with >= 50 longs."""
    rows = [r for r in recs if (is_bear_filter(r) if is_bear_filter else True)]
    best = None
    best_score = -1.0
    for tb in GRID:
        s = _eval(rows, tb, ts)
        if s["n_long"] < MIN_COVERAGE_FIT or s["acc"] is None:
            continue
        # prefer accuracy; break ties toward more trades (better coverage)
        score = (s["acc"], s["n_long"])
        if best is None or score > best_score:
            best = tb
            best_score = score
    return best


def main() -> None:
    print(f"universe n={len(SYMBOLS)} (median = true median), holdout=200/symbol\n")
    header = f"{'SYM':<9}{'policy':<16}{'tb':>6}{'n_long':>7}{'acc%':>7}{'sum_ret%':>10}{'edge%':>7}{'ctrl_sum_ret%':>13}"
    print(header)
    print("-" * len(header))

    rows_summary: list[dict] = []
    for sym in SYMBOLS:
        d = json.loads((PROBES / f"{sym.lower()}.json").read_text())
        recs = d["oof"]
        parts = _policy_dirs(recs)
        fit_rows, hold_rows = parts["fit"], parts["hold"]

        ctrl = _eval(hold_rows, CONTROL_TB, CONTROL_TS)
        ctrl_fit = _eval(fit_rows, CONTROL_TB, CONTROL_TS)

        # EXP1: per-symbol threshold from fit rows
        tb_exp1 = _pick(fit_rows, None, CONTROL_TS)
        exp1 = _eval(hold_rows, tb_exp1, CONTROL_TS) if tb_exp1 else {"n_long": 0}

        # EXP2: regime-conditional thresholds from fit rows
        tb_bear = _pick(fit_rows, lambda r: _regime(r)[1], CONTROL_TS)
        tb_non = _pick(fit_rows, lambda r: not _regime(r)[1], CONTROL_TS)
        exp2 = (
            _eval(hold_rows, CONTROL_TB, CONTROL_TS,
                  regimemap={"bear": tb_bear, "nonbear": tb_non})
            if tb_bear and tb_non else {"n_long": 0}
        )

        ctrl_ret = d["backtest"]["cum_return"]
        ctrl_ret_pct = round(ctrl_ret * 100, 2) if ctrl_ret is not None else None

        def row(label: str, s: dict, tb: object) -> None:
            acc = s.get("acc")
            sr = s.get("sum_ret")
            eg = s.get("edge")
            print(
                f"{sym:<9}{label:<16}{str(tb):>6}"
                f"{s.get('n_long', 0):>7}"
                f"{(acc * 100 if acc is not None else float('nan')):>7.1f}"
                f"{(sr * 100 if sr is not None else float('nan')):>10.1f}"
                f"{(eg * 100 if eg is not None else float('nan')):>7.1f}"
                f"{str(ctrl_ret_pct):>13}"
            )

        row("CONTROL", ctrl, CONTROL_TB)
        if exp1.get("n_long"):
            row("EXP1 per-sym", exp1, tb_exp1)
        if exp2.get("n_long"):
            row("EXP2 regime", exp2, f"{tb_bear}/{tb_non}")
        print("-" * len(header))

        rows_summary.append({
            "symbol": sym, "control": ctrl, "exp1": exp1, "exp2": exp2,
            "ctrl_fit": ctrl_fit, "tb_exp1": tb_exp1,
            "tb_bear": tb_bear, "tb_non": tb_non,
            "ctrl_replay_cum_ret_pct": ctrl_ret_pct,
        })

    # cross-symbol medians (true median on the pooled policy stats)
    def med(key, policy="control"):
        vals = []
        for e in rows_summary:
            s = e[policy]
            v = s.get(key)
            if v is None and key == "n_long":
                v = 0.0
            if v is not None:
                vals.append(float(v))
        if not vals:
            return float("nan")
        vals.sort()
        n = len(vals)
        return 0.5 * (vals[n // 2 - 1] + vals[n // 2]) if n % 2 == 0 else vals[n // 2]

    print("\nCROSS-SYMBOL MEDIANS (n=4, true median, same 200-row holdout/symbol)")
    print(f"  {'metric':<22}{'CONTROL':>12}{'EXP1':>12}{'EXP2':>12}")
    for key, label in (("n_long", "n_long (hold)"),
                       ("acc", "BUY hit-rate"),
                       ("sum_ret", "sum_ret% (hold)")):
        print(f"  {label:<22}{med(key):>12.3f}{med(key, 'exp1'):>12.3f}{med(key, 'exp2'):>12.3f}")
    print(f"  {'edge vs base-rate':<22}{med('edge'):>12.3f}{med('edge', 'exp1'):>12.3f}{med('edge', 'exp2'):>12.3f}")


if __name__ == "__main__":
    main()