"""Unit tests for slow-moving macro/global features (Task 6).

Invariants: series are reindexed to the stock calendar (missing dates stay
NaN, never back-filled) and every feature is causal.
"""

import numpy as np
import pandas as pd
import pytest

from app.config import settings
from app.ml import macro_features as ma


def _stock_frame(n: int = 120) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {"Close": 100 + pd.Series(range(n), index=index) * 0.1}, index=index
    )


def _macro(index: pd.DatetimeIndex, start: float = 1000.0,
           step: float = 1.0) -> pd.Series:
    return pd.Series(start + step * np.arange(len(index)), index=index)


class TestMacroFeatures:
    def test_columns_present_and_version(self):
        df = _stock_frame()
        macro = _macro(df.index)
        out = ma.add_macro_features(df, {"brent": macro})
        for col in ma.list_macro_features(["brent"]):
            assert col in out.columns
        assert out.attrs["feature_version"] == settings.ml_feature_version

    def test_ret_hand_computed(self):
        df = _stock_frame()
        macro = _macro(df.index)  # close = 1000 + i  -> each day +1/1000
        out = ma.add_macro_features(df, {"brent": macro})
        # 20d return at row i = (1000+i - (1000+i-20)) / (1000+i-20) = 20/(980+i)
        i = 30
        expected = 20.0 / (1000 + (i - 20))
        assert out.loc[df.index[i], "mac_brent_ret_20d"] == pytest.approx(expected)

    def test_missing_dates_stay_nan(self):
        df = _stock_frame()
        macro = _macro(df.index[5:])  # starts 5 days late
        out = ma.add_macro_features(df, {"brent": macro})
        assert out["mac_brent_ret_5d"].iloc[:5].isna().all()
        assert out["mac_brent_ret_5d"].iloc[-1] != 0  # eventually non-NaN

    def test_missing_series_absent_columns(self):
        df = _stock_frame()
        out = ma.add_macro_features(df, {"gold": _macro(df.index)})
        assert "mac_gold_ret_20d" in out.columns
        assert "mac_brent_ret_20d" not in out.columns

    def test_vol_and_drawdown_eventually_finite(self):
        df = _stock_frame(n=140)
        macro = _macro(df.index)
        out = ma.add_macro_features(df, {"usd_inr": macro})
        assert out["mac_usd_inr_vol_60d"].iloc[80:].notna().all()
        assert out["mac_usd_inr_dd_60d"].iloc[-1] <= 0

    def test_no_lookahead_via_truncation(self):
        df = _stock_frame(n=140)
        macro = _macro(df.index)
        full = ma.add_macro_features(df, {"brent": macro})
        k = 70
        truncated = ma.add_macro_features(
            df.iloc[:k], {"brent": _macro(df.index[:k])}
        )
        pd.testing.assert_frame_equal(full.iloc[:k], truncated)

    def test_feature_list_deterministic(self):
        cols = ma.list_macro_features(["gold", "brent"])
        assert cols == sorted(set(cols))
        assert len(cols) == 2 * (3 + 2)  # ret(5/20/60) + vol + dd per series

    def test_empty_macro_map(self):
        df = _stock_frame()
        out = ma.add_macro_features(df, {})
        assert len({c for c in out.columns if c.startswith("mac_")}) == 0