"""Unit tests for market/risk features (Task 4).

Values are hand-computed from small synthetic frames. No-lookahead is verified
by a truncation-equivalence test (same idea as the technical-engine test).
"""

import numpy as np
import pandas as pd
import pytest

from app.config import settings
from app.ml import market_features as mf


def _stock_frame() -> pd.DataFrame:
    """4 rows; daily returns r1=+2%, r2=-1%, r3=+3%."""
    return pd.DataFrame(
        {
            "Open": [100.0, 102.0, 100.3, 100.6],
            "High": [101.0, 103.0, 101.5, 105.0],
            "Low": [99.5, 101.8, 100.0, 100.0],
            "Close": [100.0, 102.0, 100.98, 104.0094],
            "Volume": [1000, 1000, 1000, 1000],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
    )


def _market_series() -> pd.Series:
    values = [100.0, 101.0, 102.0, 103.0]
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=4, freq="D"))


class TestVolatilityRisk:
    def test_daily_returns(self):
        out = mf.daily_returns(_stock_frame()["Close"])
        assert out.iloc[1] == pytest.approx(0.02)
        assert out.iloc[2] == pytest.approx(-0.01)
        assert out.iloc[3] == pytest.approx(0.03)

    def test_rolling_vol_hand_computed(self):
        # std(ddof=0) of [0.02, -0.01, 0.03] = 0.016997, annualised *sqrt(252)
        vol = mf.rolling_vol(_stock_frame()["Close"], window=3)
        assert vol.iloc[3] == pytest.approx(0.016997 * np.sqrt(252), rel=1e-3)

    def test_downside_vol_hand_computed(self):
        # only -0.01 contributes: sqrt((0.0001)/3) = 0.0057735, annualised
        dv = mf.downside_vol(_stock_frame()["Close"], window=3)
        assert dv.iloc[3] == pytest.approx(0.0057735 * np.sqrt(252), rel=1e-3)

    def test_rolling_percentile(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0])
        out = mf.rolling_percentile(s, 3)
        assert out.iloc[2] == pytest.approx(1.0)  # 3 is max of [1,2,3]
        assert out.iloc[3] == pytest.approx(1.0)

    def test_vol_expansion(self):
        # returns alternate (a, 0, -a) -> any 3-window has zero mean, std=a*sqrt(2/3)
        r1 = np.tile([0.05, 0.0, -0.05], 4)
        r2 = np.tile([0.1, 0.0, -0.1], 3)
        returns = np.concatenate([r1, r2])
        close = 100.0 * pd.Series(returns).fillna(0.0).add(1.0).cumprod()
        out = mf.vol_expansion(close, window=3, lookback=len(r1))
        last = out.iloc[-1]
        assert last == pytest.approx(2.0, rel=1e-3)

    def test_high_low_range_hand_computed(self):
        rng = (101 - 99.5) / 100, (103 - 101.8) / 102, (101.5 - 100.0) / 100.98, (105 - 100) / 104.0094
        out = mf.high_low_range(_stock_frame(), window=3)
        assert out.iloc[2] == pytest.approx(sum(rng[:3]) / 3)
        assert out.iloc[3] == pytest.approx(sum(rng[1:]) / 3)

    def test_gap_frequency_hand_computed(self):
        df = pd.DataFrame(
            {
                "Open": [100, 103, 101, 105],
                "Close": [100, 102, 103, 104],
            },
            index=pd.date_range("2024-01-01", periods=4, freq="D"),
        )
        # gaps at idx1 (3%>1%) and idx3 (1.94%>1%), not idx2 (0.98%<1%)
        out = mf.gap_frequency(df, window=3)
        assert out.iloc[3] == pytest.approx(2 / 3)

    def test_max_drawdown_hand_computed(self):
        s = pd.Series([100.0, 120.0, 90.0, 95.0])
        out = mf.max_drawdown(s, window=3)
        assert out.iloc[2] == pytest.approx(90 / 120 - 1)  # window [100,120,90]
        assert out.iloc[3] == pytest.approx(-0.25)  # window [120,90,95]


class TestRelative:
    def test_excess_return_hand_computed(self):
        stock = _stock_frame()["Close"]
        mkt = _market_series()
        out = mf.excess_return(stock, mkt, 2)
        # pct_change(2) at idx3 compares to idx1: (104.0094/102 - 1) - (103/101 - 1)
        expect = (104.0094 / 102.0 - 1) - (103.0 / 101.0 - 1)
        assert out.iloc[3] == pytest.approx(expect)

    def test_relative_strength_ratio_hand_computed(self):
        stock = _stock_frame()["Close"]
        mkt = _market_series()
        out = mf.relative_strength_ratio(stock, mkt, 2)
        expect = (104.0094 / 102.0) / (103.0 / 101.0) - 1
        assert out.iloc[3] == pytest.approx(expect)

    def test_rolling_corr_perfect_positive(self):
        stock = pd.Series([10.0, 11.0, 12.0, 13.0], index=pd.date_range("2024-01-01", periods=4, freq="D"))
        mkt = pd.Series([100.0, 110.0, 120.0, 130.0], index=stock.index)
        out = mf.rolling_corr(stock, mkt, window=3)
        assert out.iloc[3] == pytest.approx(1.0)

    def test_rolling_corr_perfect_negative(self):
        stock = pd.Series([10.0, 11.0, 12.0, 13.0], index=pd.date_range("2024-01-01", periods=4, freq="D"))
        mkt = pd.Series([100.0, 90.0, 81.818, 75.0], index=stock.index)
        out = mf.rolling_corr(stock, mkt, window=3)
        assert out.iloc[3] == pytest.approx(-1.0)

    def test_vol_to_market_vol_ratio(self):
        # stock returns half of market returns -> vol ratio ~0.5
        index = pd.date_range("2024-01-01", periods=4, freq="D")
        stock = pd.Series([100.0, 102.0, 100.98, 104.0094], index=index)  # returns .02,-.01,.03
        mkt = pd.Series([100.0, 104.0, 101.92, 108.0352], index=index)  # returns .04,-.02,.06
        out = mf.vol_to_market_vol(stock, mkt, window=3)
        assert out.iloc[3] == pytest.approx(0.5, rel=5e-3)

    def test_cross_sectional_rank(self):
        # pandas rolling(2) at row idx3 covers rows idx2,idx3,
        # so 2d cumulative at idx3: A=(1.2)(1.05)-1=.26, B=(1.1)(1.3)-1=.43, C=(1.0)(1.1)-1=.1
        index = pd.date_range("2024-01-01", periods=4, freq="D")
        panel = pd.DataFrame(
            {
                "A": [0.0, 0.10, 0.20, 0.05],
                "B": [0.0, 0.05, 0.10, 0.30],
                "C": [0.0, 0.20, 0.00, 0.10],
            },
            index=index,
        )
        out = mf.cross_sectional_rank(panel, lookback=2)
        assert out.iloc[3]["C"] == pytest.approx(1 / 3)
        assert out.iloc[3]["A"] == pytest.approx(2 / 3)
        assert out.iloc[3]["B"] == pytest.approx(1.0)


class TestOrchestrator:
    def test_add_market_features_columns(self):
        df = _stock_frame()
        out = mf.add_market_features(df, _market_series())
        for col in mf.list_market_features():
            assert col in out.columns
        assert out.attrs["feature_version"] == settings.ml_feature_version

    def test_market_missing_rows_stay_nan(self):
        mkt = _market_series().reindex(pd.date_range("2024-01-05", periods=4, freq="D"))
        out = mf.add_market_features(_stock_frame(), mkt)
        rel_col = "rel_ret_nifty50_10d"
        assert pd.isna(out[rel_col]).all()

    def test_list_deterministic_unique(self):
        cols = mf.list_market_features()
        assert cols == sorted(set(cols))
        assert "vol_20d" in cols and "rel_corr_nifty50_60d" in cols

    def test_idempotent(self):
        df = _stock_frame()
        out1 = mf.add_market_features(df, _market_series())
        out2 = mf.add_market_features(out1, _market_series())
        pd.testing.assert_frame_equal(out1, out2)
        assert len(out2.columns) == len(set(out2.columns))

    def test_no_lookahead_via_truncation(self):
        rng = np.random.default_rng(7)
        n = 300
        close = 100 + np.cumsum(rng.normal(0, 1.0, n))
        mkt = 100 + np.cumsum(rng.normal(0, 0.8, n))
        index = pd.date_range("2020-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "Open": close + rng.normal(0, 0.2, n),
                "High": close + np.abs(rng.normal(0, 0.5, n)),
                "Low": close - np.abs(rng.normal(0, 0.5, n)),
                "Close": close,
                "Volume": rng.integers(1000, 5000, n),
            },
            index=index,
        )
        market = pd.Series(mkt, index=index)
        k = 150
        full = mf.add_market_features(df, market)
        truncated = mf.add_market_features(df.iloc[:k], market.iloc[:k])
        pd.testing.assert_frame_equal(full.iloc[:k], truncated)