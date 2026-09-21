"""Feature-group ablation (Task 8).

Runs the same walk-forward backtest once on the FULL feature set and once
with each feature group removed ("leave-one-group-out"). If a variant keeps
beating the full model it is usually redundant; if removing a group hurts
badly that group carries signal.

Groups are derived from the deterministic, versioned feature lists every
engine publishes (technical / market / alpha / beta / event / fundamental /
sentiment / fundamental-derived macro / regime / options) intersected with
the columns actually present in the frame (macro needs a series list, so it
is matched by its `mac_` prefix).
"""

from __future__ import annotations

import pandas as pd

from app.ml import (
    alpha, beta, events, features, fundamental_features,
    market_features, options_df, regime, sentiment_features,
)
from app.ml.backtest import run_backtest


def default_group_columns(columns: list[str]) -> dict[str, list[str]]:
    """Map present columns to their feature-engine group (no duplicates)."""
    colset = list(columns)
    exact: dict[str, list[str]] = {
        "technical": features.list_technical_features(),
        "market": market_features.list_market_features(),
        "alpha": alpha.list_alpha_features(),
        "beta": beta.list_beta_features(),
        "event": events.list_event_features(),
        "fundamental": fundamental_features.list_fundamental_features(),
        "sentiment": sentiment_features.list_sentiment_features(),
        "regime": regime.list_regime_features(),
        "options": options_df.list_options_features(),
    }
    out: dict[str, list[str]] = {}
    for group, names in exact.items():
        found = [c for c in colset if c in names]
        if found:
            out[group] = found
    macro = [c for c in colset if c.startswith("mac_")]
    if macro:
        out["macro"] = macro
    return out


def run_ablation(
    frame: pd.DataFrame,
    model: str = "rf",
    horizon: int = 20,
    close_col: str = "Close",
    feature_groups: dict[str, list[str]] | None = None,
    feature_cols: list[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Leave-one-group-out ablation table (one row per variant).

    `feature_groups` / `feature_cols` default to the engine-published lists.
    """
    for key in ("feature_groups", "feature_cols"):
        kwargs.pop(key, None)
    groups = feature_groups or default_group_columns([c for c in frame.columns])
    base = feature_cols or [c for c in frame.columns if not c.startswith("target_")]
    metric_cols = ("accuracy", "f1", "win_rate", "avg_return", "cum_return",
                   "max_dd", "sharpe", "profit_factor", "n_trades")

    results: list[dict] = []

    def _run(tag: str, cols: list[str] | None):
        res = run_backtest(frame, model=model, horizon=horizon,
                           close_col=close_col, feature_cols=cols,
                           **kwargs)
        row = {"variant": tag}
        if res["metrics"] is not None:
            m = res["metrics"]
            for c in metric_cols:
                row[c] = m.get(c, float("nan"))
        else:
            for c in metric_cols:
                row[c] = float("nan")
        results.append(row)

    _run("full", base)
    for group, cols in groups.items():
        drop = set(cols)
        remaining = [c for c in base if c not in drop]
        if remaining:
            _run(f"-{group}", remaining)

    return pd.DataFrame(results).set_index("variant")