"""Unit tests for leave-one-group-out feature ablation (Task 8)."""

import numpy as np
import pandas as pd

from app.ml import ablation as ab


def _frame(n: int = 220) -> pd.DataFrame:
    idx = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(4)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0.001, 0.02, n)), index=idx)
    df = pd.DataFrame({"Close": close}, index=idx)
    h = 5
    df["leak"] = (close.shift(-h) / close - 1 > 0).astype(float)
    # representative columns for every group + a stray column
    df["ret_10d"] = close.pct_change(10)
    df["rel_ret_nifty50_20d"] = rng.normal(0, 1, n)
    df["alpha_momentum"] = rng.normal(0, 1, n)
    df["beta"] = 1.0 + rng.normal(0, 0.1, n)
    df["event_count_5d"] = rng.poisson(0.5, n)
    df["fund_pe"] = 20 + rng.normal(0, 5, n)
    df["sent_score_5d"] = rng.normal(0, 0.3, n)
    df["mac_gold_ret_20d"] = rng.normal(0.0005, 0.01, n)
    df["regime_trend"] = rng.choice([-1.0, 0.0, 1.0], n)
    df["opt_pcr"] = 1.2 + rng.normal(0, 0.3, n)
    df["stray_thing"] = rng.normal(0, 1, n)
    return df


def test_default_groups_map_expected_columns():
    cols = [c for c in _frame().columns]
    groups = ab.default_group_columns(cols)
    assert groups["technical"] == ["ret_10d"]
    assert groups["market"] == ["rel_ret_nifty50_20d"]
    assert groups["alpha"] == ["alpha_momentum"]
    assert groups["beta"] == ["beta"]
    assert groups["event"] == ["event_count_5d"]
    assert groups["fundamental"] == ["fund_pe"]
    assert groups["sentiment"] == ["sent_score_5d"]
    assert groups["macro"] == ["mac_gold_ret_20d"]
    assert groups["regime"] == ["regime_trend"]
    assert groups["options"] == ["opt_pcr"]
    all_grouped = {c for v in groups.values() for c in v}
    assert "stray_thing" not in all_grouped
    assert all_grouped == set(cols) - {"Close", "leak", "stray_thing"}


def test_groups_disjoint():
    cols = [c for c in _frame().columns]
    groups = ab.default_group_columns(cols)
    seen = []
    for cols_in_group in groups.values():
        for c in cols_in_group:
            assert c not in seen
            seen.append(c)


def test_run_ablation_table():
    df = _frame()
    out = ab.run_ablation(df, model="logistic", horizon=5, close_col="Close",
                          test_size=30, step=30, min_train=120)
    assert "full" in out.index
    for group in ("-technical", "-market", "-alpha", "-beta", "-event",
                  "-fundamental", "-sentiment", "-macro", "-regime", "-options"):
        assert group in out.index
    assert {"accuracy", "f1", "cum_return", "sharpe", "n_trades"} <= set(out.columns)
    assert (out["n_trades"] > 0).all()


def test_run_ablation_full_row_present_with_trades():
    df = _frame()
    out = ab.run_ablation(df, model="logistic", horizon=5, close_col="Close",
                          test_size=30, step=30, min_train=120)
    full = out.loc["full"]
    assert full["n_trades"] > 10
    assert 0.0 <= full["accuracy"] <= 1.0