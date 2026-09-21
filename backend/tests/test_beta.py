"""Unit tests for the beta / market-exposure model (Task 5)."""

import numpy as np
import pandas as pd
import pytest

from app.config import settings
from app.ml import beta as be


def _closes_from_returns(first: float, returns: list[float]) -> pd.Series:
    closes = [first]
    for r in returns:
        closes.append(closes[-1] * (1 + r))
    index = pd.date_range("2022-01-01", periods=len(closes), freq="B")
    return pd.Series(closes, index=index)


class TestBetaRegression:
    def test_pure_beta_recovered_exactly(self):
        mkt_ret = [0.02, 0.01, -0.01, 0.03]
        stock_ret = [2 * r for r in mkt_ret]  # beta = 2, no noise, no intercept
        market = _closes_from_returns(100.0, mkt_ret)
        stock = _closes_from_returns(100.0, stock_ret)
        out = be.add_beta_features(pd.DataFrame({"Close": stock}), market, window=3)
        # window [0.01,-0.01,0.03] vs [0.02,-0.02,0.06] => beta exactly 2, R2=1
        assert out["beta"].iloc[-1] == pytest.approx(2.0)
        assert out["beta_r2"].iloc[-1] == pytest.approx(1.0)
        assert out["beta_alpha"].iloc[-1] == pytest.approx(0.0, abs=1e-12)

    def test_beta_with_noise_is_approx_and_r2_below_one(self):
        rng = np.random.default_rng(3)
        n = 300
        mkt_ret = rng.normal(0.0003, 0.01, n)
        noise = rng.normal(0, 0.005, n)
        stock_ret = 1.5 * mkt_ret + noise
        market = _closes_from_returns(100.0, list(mkt_ret))
        stock = _closes_from_returns(100.0, list(stock_ret))
        out = be.add_beta_features(pd.DataFrame({"Close": stock}), market, window=60)
        assert out["beta"].iloc[-1] == pytest.approx(1.5, abs=0.15)
        assert 0.5 < out["beta_r2"].iloc[-1] < 1.0
        assert out["residual_vol"].iloc[-1] > 0

    def test_warmup_rows_are_nan(self):
        market = _closes_from_returns(100.0, [0.02, 0.01, -0.01, 0.03])
        stock = _closes_from_returns(100.0, [0.04, 0.02, -0.02, 0.06])
        out = be.add_beta_features(pd.DataFrame({"Close": stock}), market, window=3)
        assert pd.isna(out["beta"].iloc[0:2]).all()  # first window-1 warm-up rows NaN
        assert pd.notna(out["beta"].iloc[-1])

    def test_alignment_missing_market_stays_nan(self):
        index = pd.date_range("2024-01-01", periods=6, freq="D")
        stock = pd.Series([100, 104, 106, 108, 110, 112], index=index)
        market = pd.Series([100, 102, 103, 104, 105], index=index[1:])  # no value on last stock date
        out = be.add_beta_features(pd.DataFrame({"Close": stock}), market, window=3)
        assert pd.notna(out["beta"].iloc[-1])  # common dates computed
        assert pd.isna(out["beta"].iloc[0])     # date missing in market -> NaN, no fabrication

    def test_truncation_no_lookahead(self):
        rng = np.random.default_rng(11)
        n = 350
        mkt_ret = rng.normal(0.0002, 0.01, n)
        stock_ret = 1.3 * mkt_ret + rng.normal(0, 0.004, n)
        market = _closes_from_returns(100.0, list(mkt_ret))
        stock = _closes_from_returns(100.0, list(stock_ret))
        df = pd.DataFrame({"Close": stock})
        k = 180
        full = be.add_beta_features(df, market, window=60)
        truncated = be.add_beta_features(df.iloc[:k], market.iloc[:k], window=60)
        pd.testing.assert_frame_equal(full.iloc[:k], truncated)


class TestBetaApi:
    def test_feature_names(self):
        assert be.list_beta_features() == [
            "beta", "beta_alpha", "beta_r2", "residual_ret", "residual_vol", "residual_vol_pctile",
        ]

    def test_feature_version_attached(self):
        market = _closes_from_returns(100.0, [0.02, 0.01, -0.01, 0.03])
        stock = _closes_from_returns(100.0, [0.04, 0.02, -0.02, 0.06])
        out = be.add_beta_features(pd.DataFrame({"Close": stock}), market, window=3)
        assert out.attrs["feature_version"] == settings.ml_feature_version