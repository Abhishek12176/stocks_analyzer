"""Synthetic-panel tests for the per-symbol + pooled probability blend (Task 22)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml import explain
from app.ml import v2_pooled_blend as m
from app.ml import v2_pooled_vol_adj as v2
from tests.test_v2_pooled_vol_adj import _synthetic_panel


def _labelled() -> pd.DataFrame:
    return v2._add_targets(_synthetic_panel())


def test_per_symbol_oof_probs_indexed_by_date() -> None:
    panel = _labelled()
    frame = panel.loc[panel["symbol"] == "AAA"].sort_values("date").reset_index(drop=True)
    probs = m._per_symbol_oof_probs(frame, v2._feature_columns(panel, drop_levels=False))
    assert isinstance(probs.index, pd.DatetimeIndex)
    assert len(probs) > 0
    assert probs.between(0.0, 1.0).all()


def test_pooled_loo_probs_aligned_to_per_symbol() -> None:
    panel = _labelled()
    frame = panel.loc[panel["symbol"] == "AAA"].sort_values("date").reset_index(drop=True)
    p_per = m._per_symbol_oof_probs(frame, v2._feature_columns(panel, drop_levels=True))
    p_pool = m._pooled_loo_probs(panel, "AAA", v2._feature_columns(panel, drop_levels=True))
    common = p_per.index.intersection(p_pool.index)
    assert len(common) == min(len(p_per), len(p_pool))
    assert p_pool.between(0.0, 1.0).all()


def test_blend_auc_endpoints_reproduce_components() -> None:
    panel = _labelled()
    frame = panel.loc[panel["symbol"] == "BBB"].sort_values("date").reset_index(drop=True)
    held = panel["symbol"] == "BBB"
    label = pd.Series(
        panel.loc[held, "target_up_20d"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(panel.loc[held, "date"].to_numpy()),
    )
    p_per = m._per_symbol_oof_probs(frame, v2._feature_columns(panel, drop_levels=True))
    p_pool = m._pooled_loo_probs(panel, "BBB", v2._feature_columns(panel, drop_levels=True))

    def auc_of(s: pd.Series) -> float:
        idx = s.index.intersection(label.index)
        y = label[idx]
        ok = ~np.isnan(y.to_numpy())
        idx = idx[ok]
        return float(explain._auc(y.iloc[ok].to_numpy(), s.loc[idx].to_numpy()))

    assert m._blend_auc(p_per, p_pool, label, 0.0) == round(auc_of(p_per), 4) or \
        abs(m._blend_auc(p_per, p_pool, label, 0.0) - auc_of(p_per)) < 1e-6
    assert abs(m._blend_auc(p_per, p_pool, label, 1.0) - auc_of(p_pool)) < 1e-6
    assert 0.0 <= m._blend_auc(p_per, p_pool, label, 0.5) <= 1.0


def test_median_of_ignores_missing() -> None:
    assert m._median_of({"A": 0.5, "B": None, "C": 0.6}, ("A", "B", "C")) == 0.55
    assert np.isnan(m._median_of({"A": None}, ("A",)))