"""Unit tests for the technical feature engine (Task 3).

Values are hand-computed from the 6-bar fixture frame below. The
no-lookahead guarantee is verified by a truncation equivalence test.
"""

import numpy as np
import pandas as pd
import pytest

from app.config import settings
from app.ml import features as fe
from app.ml.features import (
    add_technical_features,
    atr,
    bollinger,
    cci,
    fibonacci_retracement,
    heikin_ashi,
    list_technical_features,
    ma_crossover,
    ma_slope,
    mfi,
    momentum,
    multi_horizon_returns,
    obv,
    pivot_levels,
    price_distance,
    roc,
    true_range,
)
from app.services.indicator_service import ema, sma


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [10.0, 10.5, 11.0, 10.8, 11.2, 11.5],
            "High": [11.0, 11.5, 11.6, 11.4, 11.8, 12.0],
            "Low": [9.5, 10.2, 10.6, 10.5, 10.9, 11.2],
            "Close": [10.8, 11.2, 11.1, 11.0, 11.6, 11.9],
            "Volume": [1000, 1200, 1100, 900, 1300, 1500],
        },
        index=pd.date_range("2024-01-01", periods=6, freq="D"),
    )


class TestCoreMomentum:
    def test_ema_hand_computed(self):
        # span=2 => alpha = 2/(2+1) = 2/3
        out = ema(_frame()["Close"], 2)
        assert out.iloc[5] == pytest.approx(11.736626)

    def test_sma_hand_computed(self):
        out = sma(_frame()["Close"], 3)
        assert out.iloc[2] == pytest.approx(11.033333333)
        assert out.iloc[5] == pytest.approx(11.5)

    def test_roc_hand_computed(self):
        out = roc(_frame()["Close"], 2)
        assert out.iloc[2] == pytest.approx(2.7777777, rel=1e-4)
        assert out.iloc[5] == pytest.approx(8.1818181, rel=1e-4)

    def test_momentum_hand_computed(self):
        out = momentum(_frame()["Close"], 2)
        assert out.iloc[2] == pytest.approx(0.3)
        assert out.iloc[5] == pytest.approx(0.9)

    def test_multi_horizon_return(self):
        out = multi_horizon_returns(_frame()["Close"], (3,))
        assert out["ret_3d"].iloc[3] == pytest.approx((11.0 - 10.8) / 10.8)


class TestRangeAndVolatility:
    def test_true_range_hand_computed(self):
        tr = true_range(_frame())
        assert tr.iloc[0] == pytest.approx(1.5)
        assert tr.iloc[1] == pytest.approx(1.3)
        assert tr.iloc[4] == pytest.approx(0.9)
        assert tr.iloc[5] == pytest.approx(0.8)

    def test_atr_wilder(self):
        out = atr(_frame(), 2)
        assert out.iloc[1] == pytest.approx(1.4)
        assert out.iloc[5] == pytest.approx(0.8875)

    def test_adx_di(self):
        adx_df = fe.adx(_frame(), 2)
        assert adx_df["plus_DI"].iloc[1] == pytest.approx(100 * 0.25 / 1.4)
        assert adx_df["minus_DI"].iloc[3] == pytest.approx(100 * 0.05 / 1.05)
        assert adx_df["ADX"].iloc[2] == pytest.approx(100.0)


class TestOscillators:
    def test_bollinger_width_and_pctb(self):
        out = bollinger(_frame()["Close"], length=3, k=2.0)
        assert out["BB_width"].iloc[2] == pytest.approx(0.06162, rel=1e-3)
        assert out["BB_pctB"].iloc[2] == pytest.approx(0.59804, rel=1e-3)
        assert out["BB_mid"].iloc[2] == pytest.approx(11.033333, rel=1e-5)

    def test_cci_hand_computed(self):
        out = cci(_frame(), 3)
        assert out.iloc[2] == pytest.approx(66.66667, abs=0.05)

    def test_obv_hand_computed(self):
        out = obv(_frame())
        assert list(out.round(4)) == pytest.approx([0, 1200, 100, -800, 500, 2000])

    def test_mfi_hand_computed(self):
        out = mfi(_frame(), 3)
        assert out.iloc[4] == pytest.approx(73.2835, abs=0.1)


class TestTrendFeatures:
    def test_price_distance_ma(self):
        close = _frame()["Close"]
        out = price_distance(close, sma(close, 3))
        assert out.iloc[2] == pytest.approx(11.1 / ((10.8 + 11.2 + 11.1) / 3) - 1)

    def test_ma_slope_hand_computed(self):
        # EMA span=2 values: idx0=10.8, idx2=11.088889
        close = _frame()["Close"]
        out = ma_slope(ema(close, 2), lookback=2)
        assert out.iloc[2] == pytest.approx((11.0888889 - 10.8) / 10.8)

    def test_ma_crossover_signal(self):
        fast = pd.Series([1.0, 2.0, 3.0, 4.0])
        slow = pd.Series([4.0, 3.0, 1.0, 6.0])
        out = ma_crossover(fast, slow)
        assert out.iloc[0] == 0.0
        assert out.iloc[1] == 0.0  # still below
        assert out.iloc[2] == 1.0  # crossed above
        assert out.iloc[3] == -1.0  # crossed below


class TestHeikinAshiPivotsFib:
    def test_heikin_ashi_body(self):
        ha = heikin_ashi(_frame())
        assert ha["ha_body_pct"].iloc[5] == pytest.approx(0.5234375 / 11.1265625)
        assert ha["ha_direction"].iloc[2] == 1.0

    def test_pivot_dist_hand_computed(self):
        out = pivot_levels(_frame(), window=3)
        assert out["pivot_dist_high_pct"].iloc[2] == pytest.approx(11.1 / 11.6 - 1)
        assert out["pivot_dist_low_pct"].iloc[2] == pytest.approx(11.1 / 9.5 - 1)  # min(9.5,10.2,10.6)

    def test_fibonacci_level_hand_computed(self):
        out = fibonacci_retracement(_frame(), window=3)
        # window rows 3..5: High=[11.4,11.8,12.0] Low=[10.5,10.9,11.2]
        assert out["fib_level_pct"].iloc[5] == pytest.approx((11.9 - 10.5) / (12.0 - 10.5))
        assert out["fib_dist_618"].iloc[5] == pytest.approx(
            abs(11.9 - (12.0 - 0.618 * 1.5)) / 11.9
        )


class TestOrchestrator:
    def test_add_all_features_columns_exist(self):
        out = add_technical_features(_frame())
        for col in list_technical_features():
            assert col in out.columns

    def test_feature_version_attached(self):
        out = add_technical_features(_frame())
        assert out.attrs["feature_version"] == settings.ml_feature_version

    def test_idempotent(self):
        df1 = add_technical_features(_frame())
        df2 = add_technical_features(df1)  # already has features
        pd.testing.assert_frame_equal(df2, df1)
        assert len(df2.columns) == len(set(df2.columns))

    def test_list_is_deterministic_and_unique(self):
        cols = list_technical_features()
        assert cols == sorted(set(cols))
        assert "ret_20d" in cols
        assert "OBV_slope" in cols

    def test_no_lookahead_via_truncation(self):
        rng = np.random.default_rng(42)
        n = 300
        close = 100 + np.cumsum(rng.normal(0, 1.0, n))
        walk = pd.DataFrame(
            {
                "Open": close + rng.normal(0, 0.2, n),
                "High": close + np.abs(rng.normal(0, 0.5, n)),
                "Low": close - np.abs(rng.normal(0, 0.5, n)),
                "Close": close,
                "Volume": rng.integers(1000, 5000, n),
            },
            index=pd.date_range("2020-01-01", periods=n, freq="B"),
        )
        full = add_technical_features(walk)
        k = 150
        truncated = add_technical_features(walk.iloc[:k])
        pd.testing.assert_frame_equal(full.iloc[:k], truncated)