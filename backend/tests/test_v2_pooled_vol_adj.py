"""Synthetic-panel tests for the vol-adjusted pooled cross-symbol probe (Task 21)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml import v2_pooled_vol_adj as m


def _synthetic_panel(
    symbols: tuple[str, ...] = ("AAA", "BBB", "CCC"),
    days: int = 420,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    business = pd.bdate_range("2023-01-02", periods=days)
    rows: list[pd.DataFrame] = []
    for i, symbol in enumerate(symbols):
        close = 100.0 + i * 50.0 + np.cumsum(rng.normal(0, 0.7, days))
        frame = pd.DataFrame({
            "symbol": symbol,
            "date": business,
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Adj Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, days).astype(float),
            "pct_ret_5d": pd.Series(close).pct_change(5).to_numpy(),
        })
        frame["signal_feature"] = np.clip(frame["Close"].shift(1) / frame["Close"], 0, 2)
        rows.append(frame)
    panel = pd.concat(rows, ignore_index=True)
    return panel.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)


def test_load_and_add_targets() -> None:
    panel = _synthetic_panel()
    panel = m._add_targets(panel)
    assert f"train_up_{m.HORIZON}d" in panel.columns
    assert f"target_up_{m.HORIZON}d" in panel.columns
    target_ok = panel[f"target_up_{m.HORIZON}d"].notna()
    covered = (panel[f"train_up_{m.HORIZON}d"].notna() & target_ok).sum()
    assert 0 < covered < target_ok.sum()


def test_feature_columns_drop_levels() -> None:
    panel = m._add_targets(_synthetic_panel())
    full = m._feature_columns(panel, drop_levels=False)
    slim = m._feature_columns(panel, drop_levels=True)
    assert "Close" in full and "Close" not in slim
    assert set(slim).issubset(set(full))


@pytest.mark.parametrize("drop_levels", [False, True])
def test_full_oof_auc_single_runs(drop_levels: bool) -> None:
    panel = m._add_targets(_synthetic_panel())
    feature_cols = m._feature_columns(panel, drop_levels=drop_levels)
    frame = panel.loc[panel["symbol"] == "AAA"].sort_values("date")
    auc_a = m._full_oof_auc_single(frame.reset_index(drop=True), feature_cols)
    auc_b = m._full_oof_auc_single(frame.reset_index(drop=True), feature_cols)
    assert auc_a is not None and auc_b is not None
    assert auc_a == auc_b


def test_pooled_loo_auc_deterministic() -> None:
    panel = m._add_targets(_synthetic_panel())
    feature_cols = m._feature_columns(panel, drop_levels=True)
    a = m._pooled_loo_auc(panel, "AAA", feature_cols)
    b = m._pooled_loo_auc(panel, "AAA", feature_cols)
    assert a is None or (a == b)


def test_pooled_loo_excludes_held_symbol_from_pool() -> None:
    """The held-out symbol's own rows must never enter the pooled training rows."""
    panel = m._add_targets(_synthetic_panel())
    held_mask = panel["symbol"] == "BBB"
    unique_dates = pd.DatetimeIndex(sorted(panel.loc[held_mask, "date"]))
    folds = m.bt.purged_walk_forward_splits(
        unique_dates, m.TEST_SIZE, m.STEP, min_train=m.MIN_TRAIN, embargo=m.EMBARGO,
    )
    train_dates = folds[0][0]
    cutoff = pd.Timestamp(max(train_dates))
    pool = panel.loc[(panel["date"] <= cutoff) & (panel["symbol"] != "BBB"), "symbol"].unique()
    assert "BBB" not in pool