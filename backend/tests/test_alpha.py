"""Unit tests for the alpha factor engine (Task 5)."""

import numpy as np
import pandas as pd
import pytest

from app.config import settings
from app.ml import alpha as al


def _close_series(returns: list[float], start: float = 100.0) -> pd.Series:
    closes = [start]
    for r in returns:
        closes.append(closes[-1] * (1 + r))
    return pd.Series(closes, index=pd.date_range("2022-01-01", periods=len(closes), freq="B"))


def _close_from_segments(segments: list[tuple[int, float]]) -> pd.Series:
    """segments = [(n_days, daily_return), ...]"""
    closes = [100.0]
    for n, r in segments:
        for _ in range(n):
            closes.append(closes[-1] * (1 + r))
    return pd.Series(closes, index=pd.date_range("2021-01-01", periods=len(closes), freq="B"))


def _random_drift(mean: float, sd: float, n: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return list(rng.normal(mean, sd, n))


def _frame_from_close(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": pd.Series(1000, index=close.index),
        },
        index=close.index,
    )


class TestRankNormalize:
    def test_hand_computed_percentile(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0])
        out = al.rank_normalize(s, window=3)
        assert out.iloc[2] == pytest.approx(1.0)
        assert out.iloc[3] == pytest.approx(1.0)

    def test_low_value_rank_is_low(self):
        s = pd.Series([10.0, 8.0, 6.0, 4.0])
        out = al.rank_normalize(s, window=3)
        assert out.iloc[3] == pytest.approx(1 / 3)  # 4 is the min of window -> rank 1/3


class TestComposite:
    def test_equal_weighted_mean(self):
        a = pd.Series([0.2, 0.2])
        b = pd.Series([0.8, 0.8])
        out = al.composite_score({"a": a, "b": b}, {"a": 1.0, "b": 1.0})
        assert out.iloc[0] == pytest.approx(0.5)

    def test_missing_rows_use_neutral(self):
        a = pd.Series([0.2, np.nan])
        b = pd.Series([0.4, 0.6])
        out = al.composite_score({"a": a, "b": b}, {"a": 1.0, "b": 1.0})
        assert out.iloc[1] == pytest.approx((0.5 + 0.6) / 2)

    def test_weighted(self):
        a = pd.Series([0.0])
        b = pd.Series([1.0])
        out = al.composite_score({"a": a, "b": b}, {"a": 3.0, "b": 1.0})
        assert out.iloc[0] == pytest.approx(0.25)


class TestFactorGroups:
    def test_momentum_uptrend_high_downtrend_low(self):
        # rank-normalized momentum: accelerating up >> accelerating down
        up_close = _close_from_segments([(250, 0.001), (150, 0.004)])
        down_close = _close_from_segments([(250, -0.001), (150, -0.004)])
        up_factor = al.momentum_factor(up_close)
        down_factor = al.momentum_factor(down_close)
        assert up_factor.iloc[-1] > 0.8
        assert down_factor.iloc[-1] < 0.5
        assert up_factor.iloc[-1] > down_factor.iloc[-1]

    def test_mean_reversion_down_beats_up(self):
        # a sharp crash vs own history => oversold => high mean-reversion score;
        # a sharp rally => overbought => low score (rank-relative semantics)
        crash_close = _close_series(
            list(np.random.default_rng(7).normal(0.0005, 0.002, 250))
            + list(np.random.default_rng(8).normal(-0.03, 0.01, 40)),
            start=100.0,
        )
        rally_close = _close_series(
            list(np.random.default_rng(9).normal(0.0005, 0.002, 250))
            + list(np.random.default_rng(8).normal(0.03, 0.01, 40)),
            start=100.0,
        )
        crash_rev = al.mean_reversion_factor(crash_close, window=126)
        rally_rev = al.mean_reversion_factor(rally_close, window=126)
        assert crash_rev.iloc[-1] > 0.5
        assert crash_rev.iloc[-1] > rally_rev.iloc[-1]

    def test_volatility_factor_calms_when_vol_contracts(self):
        # vol factor = 1 - percentile(own-history vol): a strictly drying-up
        # regime (current vol = own historic minimum) scores high; a ramping-up
        # one (current vol = own historic maximum) scores ~0.
        def _ramp(spec: list[float], seed: int) -> pd.Series:
            rng = np.random.default_rng(seed)
            rets = [rng.normal(0.0005, sd) for sd in spec]
            return _close_series(rets, start=100.0)

        quieting = _ramp([0.05] * 200 + list(np.linspace(0.02, 0.001, 150)), seed=1)
        erupting = _ramp([0.001] * 200 + list(np.linspace(0.0005, 0.05, 150)), seed=2)
        quiet_score = al.volatility_factor(quieting, window=40, rank_window=126)
        erupt_score = al.volatility_factor(erupting, window=40, rank_window=126)
        assert quiet_score.iloc[-1] > 0.8
        assert erupt_score.iloc[-1] < 0.2
        assert quiet_score.iloc[-1] > erupt_score.iloc[-1]

    def test_value_factor_graceful_disable(self):
        assert al.value_factor() is None
        pe = pd.Series([40, 35, 30, 25, 20, 15, 10])
        score = al.value_factor(pe=pe, window=5)
        assert score is not None and not score.isna().all()

    def test_value_factor_prefers_low_pe_high_dy(self):
        pe = pd.Series([50, 40, 30, 20, 10])          # descending
        dy = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])  # ascending
        score = al.value_factor(pe=pe, dividend_yield=dy, window=3)
        # last row: lowest PE (rank 1) + highest dividend yield (rank 1) -> highest composite
        assert score.iloc[-1] >= score.iloc[:-1].max() - 1e-9


class TestValidationMetadata:
    def test_defaults_unvalidated(self):
        meta = al.factor_metadata()
        assert set(meta) == {"momentum", "mean_reversion", "volatility", "value"}
        assert all(m["oos_validated"] is False for m in meta.values())

    def test_mark_validated(self):
        al.mark_factor_validated("momentum")
        assert al.factor_metadata()["momentum"]["oos_validated"] is True
        al.FACTOR_GROUPS["momentum"]["oos_validated"] = False  # reset

    def test_unknown_group_raises(self):
        with pytest.raises(KeyError):
            al.mark_factor_validated("nope")


class TestOrchestrator:
    def test_add_alpha_features_columns_and_version(self):
        close = _close_series(_random_drift(0.002, 0.002, 400, seed=1))
        out = al.add_alpha_features(_frame_from_close(close))
        for col in ["alpha_momentum", "alpha_mean_reversion", "alpha_volatility"]:
            assert col in out.columns
        assert "alpha_value" not in out.columns  # no fundamental input provided
        assert out.attrs["feature_version"] == settings.ml_feature_version

    def test_add_alpha_with_value_input(self):
        close = _close_series(_random_drift(0.002, 0.002, 400, seed=1))
        pe = pd.Series(20.0, index=close.index)
        out = al.add_alpha_features(_frame_from_close(close), pe=pe)
        assert "alpha_value" in out.columns

    def test_list_alpha_features_deterministic(self):
        assert al.list_alpha_features() == sorted(set(al.list_alpha_features()))

    def test_truncation_no_lookahead(self):
        close = _close_series(_random_drift(0.001, 0.002, 400, seed=2))
        df = _frame_from_close(close)
        full = al.add_alpha_features(df)
        k = 200
        truncated = al.add_alpha_features(df.iloc[:k])
        pd.testing.assert_frame_equal(full.iloc[:k], truncated)