"""Walk-forward backtest + purged CV + costs + metrics (Task 8).

Core loop (per fold, strictly chronological):
    train -> purge last `embargo` rows (their 20d labels would overlap the
             test window) -> fit -> predict P(up) on test -> bet +1 long when
             P(up) >= threshold (or -1 short with `allow_short`) -> P&L =
             direction * 20d-forward-return - realistic round-trip cost.

Metrics compare each model against the existing rule-based voting engine and
buy-and-hold, in the SAME trade-evaluation frame (every row T is a 20d bet).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.config import settings
from app.ml import dataset as ds
from app.ml import explain
from app.ml import models as mo

ANNUAL_DAYS = 252

# Nominal starting capital for the no-leverage mark-to-market ledger. Returns
# and drawdowns are scale-invariant; the level only sets the per-position size
# = capital / horizon (max gross exposure = 1.0x capital, no hidden leverage).
BACKTEST_CAPITAL = 1_000_000.0


@dataclass(frozen=True)
class CostModel:
    brokerage_inr: float = 20.0
    fee_percent: float = 0.01
    slippage_bps: float = 10.0

    @classmethod
    def from_settings(cls) -> "CostModel":
        return cls(
            brokerage_inr=settings.ml_cost_brokerage_inr,
            fee_percent=settings.ml_cost_fee_percent,
            slippage_bps=settings.ml_cost_slippage_bps,
        )

    def per_side_cost(self, price: float) -> float:
        if price is None or price <= 0:
            return np.nan
        one_side = (
            self.brokerage_inr
            + (self.fee_percent / 100.0) * price
            + (self.slippage_bps / 10000.0) * price
        ) / price
        return float(one_side)

    def round_trip(self, entry: float, exit_price: float | None = None) -> float:
        exit_price = entry if exit_price is None else exit_price
        one = self.per_side_cost(entry)
        two = self.per_side_cost(exit_price)
        if np.isnan(one) or np.isnan(two):
            return np.nan
        return one + two


# ---------------------------------------------------------------------------
# Purging / embargo
# ---------------------------------------------------------------------------

def purge_last(train_idx: pd.DatetimeIndex, embargo: int) -> pd.DatetimeIndex:
    """Drop the last `embargo` training rows so their 20d labels (which would
    poke into the test window) never leak into the fit."""
    if embargo <= 0 or len(train_idx) <= embargo:
        return train_idx
    return train_idx[:-embargo]


def purged_walk_forward_splits(
    index: pd.DatetimeIndex,
    test_size: int,
    step: int,
    min_train: int,
    embargo: int = 0,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    folds = ds.walk_forward_splits(index, test_size, step, min_train=min_train)
    return [(purge_last(train, embargo), test) for train, test in folds]


def purged_chrono_splits(
    index: pd.DatetimeIndex,
    n_splits: int = 5,
    embargo: int = 20,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Expanding-window purged CV folds (chronological, embargoed)."""
    index = pd.DatetimeIndex(sorted(index))
    out = []
    n = len(index)
    for i in range(1, n_splits + 1):
        cut = int(round(n * i / n_splits))
        if cut >= n:
            continue
        train = purge_last(index[:cut], embargo)
        test = index[cut:]
        if len(train) > 0 and len(test) > 0:
            out.append((train, test))
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def classification_metrics(y_true: np.ndarray, y_pred_buy: np.ndarray) -> dict:
    """Accuracy / precision / recall / F1 / confusion (+ false-buy/sell %).

    Positive class = 1 (up); `y_pred_buy` = 1 when a BUY signal was fired.
    """
    y_true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred_buy, dtype=int)
    n = len(y_true)
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tp = int(((pred == 1) & (y_true == 1)).sum())
    acc = (tp + tn) / n if n else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "n": n},
        "false_buy_pct": (fp / (tp + fp)) if (tp + fp) else float("nan"),
        "false_sell_pct": (fn / (tn + fn)) if (tn + fn) else float("nan"),
    }


def trade_metrics(pl: np.ndarray, horizon: int = 20) -> dict:
    """Per-trade stats over the vector of per-20d-bet P&L fractions.

    This describes the *bets* (win rate, average per-bet return, profit
    factor). Equity-level stats (cum_return / max_dd / sharpe / sortino /
    calmar) must never be compounded from these overlapping bets — that is
    what the old code did here and it produced fake -99% / < -100% drawdowns.
    Real equity-level stats come from `equity_ledger` + `equity_metrics`.
    """
    del horizon  # per-bet stats are horizon-independent
    pl = np.asarray(pl, dtype=float)
    pl = pl[~np.isnan(pl)]
    if len(pl) == 0:
        return {k: float("nan") for k in
                ("n_trades", "win_rate", "avg_return", "profit_factor")}
    gross_profit = float(np.sum(pl[pl > 0]))
    gross_loss = float(-np.sum(pl[pl < 0]))
    pf = (gross_profit / gross_loss if gross_loss > 0
          else (float("inf") if gross_profit > 0 else float("nan")))
    return {
        "n_trades": int(len(pl)),
        "win_rate": float((pl > 0).mean()),
        "avg_return": float(np.mean(pl)),
        "profit_factor": pf,
    }


def calibration_bins(prob: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> dict:
    """Mean predicted vs mean actual, per probability bin + Brier score."""
    prob = np.asarray(prob, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    valid = ~np.isnan(prob)
    prob, y_true = prob[valid], y_true[valid]
    bins = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob >= lo) & (prob < hi) if hi < 1.0 else (prob >= lo) & (prob <= hi)
        if m.sum() == 0:
            continue
        bins.append({
            "bin_lo": float(lo), "bin_hi": float(hi), "count": int(m.sum()),
            "predicted": float(prob[m].mean()), "actual": float(y_true[m].mean()),
        })
    brier = float(np.mean((prob - y_true) ** 2)) if len(prob) else float("nan")
    return {"bins": bins, "brier": brier, "n": int(len(prob))}


# ---------------------------------------------------------------------------
# Mark-to-market equity ledger + honest equity metrics
# ---------------------------------------------------------------------------

def equity_ledger(
    close: pd.Series,
    signals: pd.DataFrame,
    horizon: int,
    capital: float,
    cost: CostModel | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Daily MTM equity for a staggered long/flat (or short) book.

    Every non-zero signal (index = date, column ``direction`` in {+1, -1})
    opens a position of notional ``C = capital / horizon`` at ``close[date]``
    and settles it at ``close[date + horizon]`` — exactly the forward-return
    window of the prediction target ``target_ret_{horizon}d``, so each bet's
    P&L ties to the model's label. Cost is charged exactly once, at the open
    (never inside the daily mark, so it can't be double-counted).

    Sizing is no-leverage by construction: a position lives exactly ``horizon``
    bars and at most one opens per bar, so committed position notional
    ``<= horizon * C == capital`` (enforced with a hard guard).

    Returns ``(equity, trades)``:
    - equity : per-bar ``cash + sum(active V_j)`` on the close timeline, where
      ``V_j = direction * C * close(t) / P_j``;
    - trades : settled positions with entry/exit date, direction, prices,
      cost, ``ret = exit/entry - 1`` and net ``pl``.
    Equity satisfies ``equity(t) = capital + realized P&L + unrealized P&L
    - cumulated costs`` at every bar (the two ledger invariants).
    """
    if cost is None:
        cost = CostModel.from_settings()
    close = close.dropna().sort_index()
    timeline = list(close.index)
    if len(timeline) == 0 or capital <= 0:
        raise ValueError("equity_ledger needs a non-empty close series and capital > 0")
    horizon = max(int(horizon), 1)
    C = float(capital) / horizon

    if "direction" not in signals.columns:
        raise ValueError("signals needs a 'direction' column")
    sig = signals.copy()
    sig = sig[sig["direction"].notna() & (sig["direction"] != 0)]
    if not sig.index.is_unique:
        sig = sig[~sig.index.duplicated(keep="first")]
    locs = close.index.get_indexer(sig.index)
    sig = sig.assign(_pos=locs)
    sig = sig[(sig["_pos"] >= 0) & (sig["_pos"] + horizon < len(close))]
    if sig.empty:
        empty_cols = ["entry_date", "exit_date", "direction", "entry", "exit",
                      "notional", "cost", "ret", "pl"]
        return pd.Series(dtype=float), pd.DataFrame(columns=empty_cols)

    open_at: dict[int, list[dict[str, Any]]] = {}
    n_open = 0
    for _, row in sig.iterrows():
        p = int(row["_pos"])
        entry = float(close.iloc[p])
        if not np.isfinite(entry) or entry <= 0:
            continue
        direction = float(row["direction"])
        frac = cost.round_trip(entry)
        cost_notional = C * frac if np.isfinite(frac) else 0.0
        rec = {
            "entry_pos": p, "exit_pos": p + horizon,
            "direction": direction, "shares": direction * C / entry,
            "notional": C, "entry_price": entry, "cost": cost_notional,
            "entry_date": row.name,
        }
        open_at.setdefault(p, []).append(rec)
        n_open += 1
    if n_open == 0:
        empty_cols = ["entry_date", "exit_date", "direction", "entry", "exit",
                      "notional", "cost", "ret", "pl"]
        return pd.Series(dtype=float), pd.DataFrame(columns=empty_cols)

    cash = float(capital)
    active: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    equity_vals: list[float] = []
    max_concurrent = 0
    for t in range(len(close)):
        price = float(close.iloc[t])
        stayed: list[dict[str, Any]] = []
        for rec in active:
            if rec["exit_pos"] == t:
                exit_price = price
                cash += rec["shares"] * exit_price
                trades.append({
                    "entry_date": rec["entry_date"],
                    "exit_date": timeline[t],
                    "direction": rec["direction"],
                    "entry": rec["entry_price"],
                    "exit": exit_price,
                    "notional": rec["notional"],
                    "cost": rec["cost"],
                    "ret": exit_price / rec["entry_price"] - 1.0,
                    "pl": (rec["direction"] * rec["notional"]
                           * (exit_price / rec["entry_price"] - 1.0)) - rec["cost"],
                })
            else:
                stayed.append(rec)
        active = stayed
        for rec in open_at.get(t, []):
            committed = abs(rec["shares"]) * rec["entry_price"] + sum(
                abs(r["shares"]) * r["entry_price"] for r in active
            )
            if committed > capital + 1e-6:
                raise RuntimeError(
                    f"no-hidden-leverage violated at {timeline[t]}: "
                    f"committed {committed:.4f} > capital {capital:.4f}")
            cash -= rec["direction"] * rec["notional"] + rec["cost"]
            active.append(rec)
        max_concurrent = max(max_concurrent, len(active))
        mark = cash + sum(r["shares"] * price for r in active)
        equity_vals.append(mark)

    equity = pd.Series(equity_vals, index=timeline, dtype=float)
    trades_df = pd.DataFrame(
        trades,
        columns=["entry_date", "exit_date", "direction", "entry", "exit",
                 "notional", "cost", "ret", "pl"],
    )
    equity.attrs["max_concurrent"] = int(max_concurrent)
    equity.attrs["capital"] = float(capital)
    return equity, trades_df


def equity_metrics(equity: pd.Series) -> dict:
    """True portfolio stats from a daily marked-to-market equity curve."""
    eq = equity.dropna()
    n = int(len(eq))
    if n < 2 or float(eq.iloc[0]) <= 0:
        base = {k: float("nan") for k in (
            "cum_return", "max_dd", "sharpe", "sortino", "calmar",
            "annualized_return", "annualized_volatility", "avg_daily_return")}
        base["n_days"] = n
        return base
    vals = eq.to_numpy(dtype=float)
    cum = float(vals[-1] / vals[0] - 1.0)
    peak = np.maximum.accumulate(vals)
    max_dd = float(np.min(vals / peak - 1.0))
    daily = vals[1:] / vals[:-1] - 1.0
    mean_r = float(np.mean(daily))
    std_r = float(np.std(daily, ddof=0))
    sharpe = (mean_r / std_r) * math.sqrt(ANNUAL_DAYS) if std_r > 0 else float("nan")
    down = daily[daily < 0]
    dsig = float(np.std(down, ddof=0)) if len(down) else 0.0
    sortino = (mean_r / dsig) * math.sqrt(ANNUAL_DAYS) if dsig > 0 else float("nan")
    ann = (1.0 + cum) ** (ANNUAL_DAYS / max(n - 1, 1)) - 1.0 if cum > -1.0 else float("nan")
    vol = std_r * math.sqrt(ANNUAL_DAYS)
    calmar = ann / abs(max_dd) if max_dd < 0 else float("nan")
    return {
        "cum_return": cum, "max_dd": max_dd, "sharpe": sharpe,
        "sortino": sortino, "calmar": calmar,
        "annualized_return": ann, "annualized_volatility": vol,
        "avg_daily_return": mean_r, "n_days": n,
    }


def strategy_metrics(
    close: pd.Series,
    signals: pd.DataFrame,
    horizon: int,
    capital: float,
    cost: CostModel | None = None,
) -> dict | None:
    """Uniform performance metrics for any set of direction signals.

    ``signals``: index = dates, columns ``direction`` (+1 long / -1 short /
    0 flat) plus optional ``prob`` (P(up)), ``y`` (0/1 label) and ``ret``
    (the target forward return at that date).

    - per-trade stats come from the settled ledger trades;
    - equity stats come from the daily MTM curve (``equity_metrics``);
    - when ``prob`` and ``y`` are supplied, classification metrics + ROC-AUC
      + Brier are computed from the flagged long decisions (P(up) >= 0.5).
    """
    real = close.dropna().sort_index()
    if len(real) == 0:
        return None
    sig = signals.copy()
    if "prob" in sig.columns:
        sig = sig[sig["prob"].notna()]
    sig = sig[sig["direction"].notna()]
    if sig.empty:
        return None
    start = sig.index.min()
    win = real.loc[start:]
    if len(win) < horizon + 1:
        return None

    equity, tdf = equity_ledger(win, sig, horizon, capital, cost)
    metrics = equity_metrics(equity)
    max_concurrent = float(equity.attrs.get("max_concurrent", float("nan"))) if len(equity) else float("nan")
    metrics["max_concurrent"] = max_concurrent
    metrics["no_leverage_ok"] = (
        ("yes" if len(equity) == 0 else "no")
        if not np.isfinite(max_concurrent)
        else (
            "yes" if max_concurrent <= max(int(horizon), 1) else "no"
        )
    )
    metrics["trade_accuracy"] = float("nan")

    if len(tdf):
        pl = tdf["pl"].to_numpy(dtype=float)
        metrics["n_trades"] = int(len(pl))
        metrics["win_rate"] = float((pl > 0).mean())
        metrics["avg_return"] = float(np.mean(pl))
        gp = float(np.sum(pl[pl > 0]))
        gl = float(-np.sum(pl[pl < 0]))
        metrics["profit_factor"] = gp / gl if gl > 0 else (float("inf") if gp > 0 else float("nan"))
    else:
        metrics["n_trades"] = 0
        metrics["win_rate"] = float("nan")
        metrics["avg_return"] = float("nan")
        metrics["profit_factor"] = float("nan")

    if "ret" in sig.columns:
        rr = sig["ret"].to_numpy(dtype=float)
        dd = sig["direction"].to_numpy(dtype=float)
        active = (dd != 0) & ~np.isnan(rr)
        if active.any():
            metrics["trade_accuracy"] = float((rr[active] * dd[active] > 0).mean())

    metrics.setdefault("accuracy", metrics["trade_accuracy"]
                       if np.isfinite(metrics["trade_accuracy"]) else float("nan"))
    for k in ("precision", "recall", "f1", "false_buy_pct", "false_sell_pct",
              "roc_auc", "calibration_brier"):
        metrics.setdefault(k, float("nan"))
    metrics.setdefault("confusion", None)

    if "y" in sig.columns and "prob" in sig.columns:
        yy = sig["y"].to_numpy(dtype=float)
        pp = sig["prob"].to_numpy(dtype=float)
        dd = sig["direction"].to_numpy(dtype=float)
        both = ~np.isnan(yy) & ~np.isnan(pp) & ~np.isnan(dd)
        if both.any():
            yv = yy[both].astype(int)
            pv = pp[both]
            dv = dd[both]
            cl = classification_metrics(yv, (dv > 0).astype(int))
            cl["roc_auc"] = explain._auc(yv, pv)
            cl["calibration_brier"] = float(np.mean((pv - yv) ** 2))
            for key, val in cl.items():
                metrics[key] = val

    # Provenance: which metrics describe one completed 20-day bet and which
    # describe the daily marked-to-market equity curve. This keeps the final
    # report honest about what each number measures.
    _PER_BET = ("n_trades", "win_rate", "avg_return", "profit_factor")
    _EQUITY_CURVE = ("cum_return", "max_dd", "sharpe", "sortino", "calmar",
                     "annualized_return", "annualized_volatility",
                     "avg_daily_return", "n_days", "max_concurrent",
                     "no_leverage_ok")
    _CLASSIFICATION = ("accuracy", "precision", "recall", "f1", "confusion",
                       "false_buy_pct", "false_sell_pct", "roc_auc",
                       "calibration_brier", "trade_accuracy")
    metrics["metric_groups"] = {
        "per_bet": {"metrics": list(_PER_BET),
                    "meaning": "per completed horizon-day bet (one settled "
                               "position; entry Close[T], exit Close[T+horizon])"},
        "daily_equity_curve": {"metrics": list(_EQUITY_CURVE),
                               "meaning": "daily marked-to-market portfolio/"
                                          "equity curve over the tested window"},
        "classification": {"metrics": list(_CLASSIFICATION),
                           "meaning": "P(up)>=0.5 decision vs actual direction "
                                      "on the tested rows"},
    }
    return metrics


# ---------------------------------------------------------------------------
# Imputation (sklearn-free median fill, fit-on-train only)
# ---------------------------------------------------------------------------

class _MedianImputer:
    def __init__(self, columns: list[str]):
        self.columns = columns
        self.median: dict[str, float] = {}

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        arr = X[self.columns].to_numpy(dtype=float)
        medians = np.nanmedian(arr, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        self.median = {c: float(m) for c, m in zip(self.columns, medians)}
        return self._fill(arr)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        arr = X[self.columns].to_numpy(dtype=float)
        return self._fill(arr)

    def _fill(self, arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        for j, m in enumerate(self.median.values()):
            col = out[:, j]
            col[np.isnan(col)] = m
            out[:, j] = col
        return out


# ---------------------------------------------------------------------------
# Core run
# ---------------------------------------------------------------------------

def _default_feature_cols(frame: pd.DataFrame, horizon: int) -> list[str]:
    excluded = {"date"} | {f"target_ret_{horizon}d", f"target_up_{horizon}d"}
    return [c for c in frame.columns if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])]


def _fit_predict_with_median(
    model: str, train_idx, test_idx, X: pd.DataFrame, y: pd.Series,
    seed: int = 0, model_kwargs: dict[str, Any] | None = None,
) -> np.ndarray:
    if model == "voting":
        # raw values go straight to the rule engine (never impute a vote).
        return mo.fit_and_predict(model, X.loc[train_idx], y.loc[train_idx],
                                  X.loc[test_idx], seed=seed,
                                  **(model_kwargs or {}))
    imp = _MedianImputer(list(X.columns))
    Xtr = pd.DataFrame(imp.fit_transform(X.loc[train_idx]),
                       columns=X.columns, index=X.loc[train_idx].index)
    Xte = pd.DataFrame(imp.transform(X.loc[test_idx]),
                       columns=X.columns, index=X.loc[test_idx].index)
    return mo.fit_and_predict(model, Xtr, y.loc[train_idx], Xte, seed=seed,
                              **(model_kwargs or {}))


def run_backtest(
    frame: pd.DataFrame,
    model: str = "rf",
    horizon: int = 20,
    close_col: str = "Close",
    feature_cols: list[str] | None = None,
    test_size: int = 60,
    step: int = 60,
    min_train: int = 260,
    embargo: int | None = None,
    threshold: float = 0.5,
    cost: CostModel | None = None,
    allow_short: bool = False,
    seed: int = 0,
    model_kwargs: dict[str, Any] | None = None,
) -> dict:
    """Walk-forward backtest of one model over `frame`.

    Returns a result dict (metrics/trades/calibration/fold_metrics) usable by
    every downstream consumer; see module docstring for the bet semantics.
    """
    if cost is None:
        cost = CostModel.from_settings()
    embargo = horizon if embargo is None else embargo

    out = ds.add_target(frame, horizon=horizon, close_col=close_col)
    ret = out[f"target_ret_{horizon}d"]
    y = out[f"target_up_{horizon}d"]
    close = out[close_col] if close_col in out.columns else None

    feat_cols = feature_cols if feature_cols is not None else _default_feature_cols(
        out, horizon)
    keep = [c for c in feat_cols if out[c].notna().sum() > 0]
    X = out[keep]
    index = pd.DatetimeIndex(out.index)

    folds = purged_walk_forward_splits(index, test_size, step, min_train, embargo=embargo)

    rows: list[dict] = []
    fold_info: list[dict] = []
    for fi, (train_idx, test_idx) in enumerate(folds):
        prob = _fit_predict_with_median(model, train_idx, test_idx, X, y,
                                        seed=seed + fi, model_kwargs=model_kwargs)
        fold_info.append({
            "fold": fi, "train_size": int(len(train_idx)), "test_size": int(len(test_idx)),
        })
        for dte, p in zip(test_idx, prob):
            if p is None or np.isnan(p):
                continue
            direction = 1.0 if p >= threshold else 0.0
            if allow_short and direction == 0.0:
                direction = -1.0
            rows.append({"date": dte, "prob": p, "direction": direction})
    if not rows:
        return {"model": model, "metrics": None, "trades": pd.DataFrame(),
                "calibration": None, "fold_metrics": fold_info}

    trades = pd.DataFrame(rows).set_index("date")
    trades["ret"] = ret.reindex(trades.index)
    entry = close.reindex(trades.index) if close is not None else None
    costs = []
    for _, t in trades.iterrows():
        if t["ret"] is None or np.isnan(t["ret"]):
            costs.append(np.nan)
            continue
        px = entry.loc[t.name] if entry is not None and t.name in entry.index else None
        rnd = cost.round_trip(float(px)) if px is not None and not np.isnan(px) else 0.0
        costs.append(rnd if t["direction"] != 0 else 0.0)
    trades["cost"] = np.asarray(costs, dtype=float)
    trades["pl"] = (
        trades["direction"] * trades["ret"] - trades["cost"].fillna(0.0)
    )

    valid = trades["pl"].notna() & trades["ret"].notna()
    trades = trades[valid]
    if trades.empty:
        return {"model": model, "metrics": None, "trades": trades,
                "calibration": None, "fold_metrics": fold_info}
    trades = trades.sort_index()

    eval_rows = trades[trades["ret"].notna()].copy()
    signals = pd.DataFrame({
        "direction": eval_rows["direction"].to_numpy(dtype=float),
        "prob": eval_rows["prob"].to_numpy(dtype=float),
        "y": (eval_rows["ret"] > 0).astype(int).to_numpy(dtype=float),
        "ret": eval_rows["ret"].to_numpy(dtype=float),
    }, index=eval_rows.index)

    metrics = None
    if close is not None:
        metrics = strategy_metrics(close, signals, horizon, BACKTEST_CAPITAL, cost)
    if metrics is None:
        metrics = {
            "n_trades": 0, "win_rate": float("nan"), "avg_return": float("nan"),
            "cum_return": float("nan"), "max_dd": float("nan"),
            "sharpe": float("nan"), "sortino": float("nan"),
            "calmar": float("nan"), "profit_factor": float("nan"),
            "accuracy": float("nan"), "precision": float("nan"),
            "recall": float("nan"), "f1": float("nan"),
            "false_buy_pct": float("nan"), "false_sell_pct": float("nan"),
            "roc_auc": float("nan"), "calibration_brier": float("nan"),
            "annualized_return": float("nan"),
            "annualized_volatility": float("nan"),
            "avg_daily_return": float("nan"), "n_days": int(len(eval_rows)),
            "trade_accuracy": float("nan"), "max_concurrent": 0.0,
            "no_leverage_ok": "yes", "confusion": None,
        }

    y_true = (eval_rows["ret"] > 0).astype(int).to_numpy()
    prob = eval_rows["prob"].to_numpy(dtype=float)

    return {
        "model": model,
        "metrics": metrics,
        "trades": trades,
        "calibration": calibration_bins(prob, y_true) if len(prob) else None,
        "fold_metrics": fold_info,
    }


def buy_and_hold_metrics(frame: pd.DataFrame, horizon: int = 20,
                         close_col: str = "Close",
                         capital: float = BACKTEST_CAPITAL,
                         eval_index: pd.DatetimeIndex | None = None) -> dict:
    """Continuous buy-and-hold benchmark over a close window.

    One long position is opened at the first evaluation close and held to the
    last close, so equity = capital * close / close_entry (n_trades = 1) and
    every equity metric comes from that real price path. ``accuracy`` is the
    fraction of ``horizon``-bar forward windows that were up over the window
    (the old "always-long" hit-rate). ``eval_index`` restricts the window to a
    specific evaluation period (e.g. the final hold-out).
    """
    out = ds.add_target(frame, horizon=horizon, close_col=close_col)
    close = out[close_col].dropna().sort_index()
    if eval_index is not None:
        idx = pd.DatetimeIndex(eval_index).intersection(close.index)
        win = close.loc[idx.min():] if len(idx) else close
    else:
        win = close
    names = ["cum_return", "max_dd", "sharpe", "sortino", "calmar",
             "profit_factor", "annualized_return", "annualized_volatility"]
    if len(win) < 2:
        met = {k: float("nan") for k in names}
        met.update({
            "n_trades": 0, "win_rate": float("nan"), "avg_return": float("nan"),
            "accuracy": float("nan"), "avg_daily_return": float("nan"),
            "n_days": int(len(win)), "roc_auc": float("nan"),
            "calibration_brier": float("nan"), "trade_accuracy": float("nan"),
            "precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
            "max_concurrent": 0.0, "no_leverage_ok": "yes",
            "confusion": None,
        })
        return met

    equity = capital * win / float(win.iloc[0])
    met = equity_metrics(equity)
    realized = float(win.iloc[-1] / win.iloc[0] - 1.0)
    met["n_trades"] = 1
    met["win_rate"] = (1.0 if realized > 0 else (0.0 if realized < 0 else float("nan")))
    met["avg_return"] = realized
    met["profit_factor"] = float("inf") if realized > 0 else float("nan")
    met["max_concurrent"] = 1.0
    met["no_leverage_ok"] = "yes"
    met["roc_auc"] = float("nan")
    met["calibration_brier"] = float("nan")
    met["precision"] = float("nan")
    met["recall"] = float("nan")
    met["f1"] = float("nan")
    ret = out.get(f"target_ret_{horizon}d")
    if ret is not None:
        rr = ret.loc[win.index].dropna()
        met["accuracy"] = float((rr > 0).mean()) if len(rr) else float("nan")
    else:
        met["accuracy"] = float("nan")
    met["trade_accuracy"] = met["accuracy"]
    met["confusion"] = None
    return met


def compare_models(
    frame: pd.DataFrame,
    models: tuple[str, ...] = ("voting", "logistic", "rf", "xgboost"),
    buy_hold: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Compact comparison table over the same walk-forward folds."""
    rows = []
    for m in models:
        res = run_backtest(frame, model=m, **kwargs)
        if res["metrics"] is None:
            continue
        met = res["metrics"]
        rows.append({
            "model": m,
            "accuracy": met["accuracy"], "f1": met["f1"],
            "win_rate": met["win_rate"], "avg_return": met["avg_return"],
            "cum_return": met["cum_return"], "max_dd": met["max_dd"],
            "sharpe": met["sharpe"], "calmar": met["calmar"],
            "profit_factor": met["profit_factor"], "n_trades": met["n_trades"],
        })
    if buy_hold:
        bh = buy_and_hold_metrics(frame, horizon=kwargs.get("horizon", 20),
                                  close_col=kwargs.get("close_col", "Close"))
        rows.append({
            "model": "buy_hold",
            "accuracy": bh.get("accuracy"), "f1": float("nan"),
            "win_rate": bh["win_rate"], "avg_return": bh["avg_return"],
            "cum_return": bh["cum_return"], "max_dd": bh["max_dd"],
            "sharpe": bh["sharpe"], "calmar": bh["calmar"],
            "profit_factor": bh["profit_factor"], "n_trades": bh["n_trades"],
        })
    return pd.DataFrame(rows)