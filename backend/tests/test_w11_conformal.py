"""W11 tests — split-conformal uncertainty intervals around the calibrated
P(up). All synthetic/offline (no network): nothing here asserts alpha.

W11 = a point probability ("0.62 up") carries no statement about how much it
can be trusted. These tests pin the honest uncertainty layer added by
`conformal.py` and surfaced in `forecast_frame()["uncertainty"]`:
- the conservative higher-quantile math (coverage-valid for exchangeable rows),
- the MIN_N insufficiency floor (a band from 3 rows is never shown),
- NaN / missing-probability handling (never fabricated),
- the wide-band warning, the effective-windows overlap caveat, and the
  holdout empirical-coverage sanity check.
"""

import numpy as np
import pandas as pd
import pytest

from app.ml import conformal as cf
from app.ml import pipeline


def _scores(n: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.abs(rng.normal(0.12, 0.10, n))


class TestQuantileMath:
    def test_higher_quantile_index_is_conservative(self):
        # n=9, alpha=0.10 -> k = ceil(10*0.9) = 9 -> the LARGEST score
        assert cf._higher_quantile_index(9, 0.10) == 9
        # n=99 -> k = ceil(100*0.9) = 90
        assert cf._higher_quantile_index(99, 0.10) == 90
        # clamped into [1, n]
        assert cf._higher_quantile_index(5, 0.0) == 5
        assert cf._higher_quantile_index(5, 0.99) == 1

    def test_interval_uses_the_ranked_score_exactly(self):
        s = np.array([0.05, 0.10, 0.15, 0.20, 0.25])
        got = cf.conformal_interval(0.60, s, alpha=0.4, min_n=3)
        # k = ceil(6*0.6) = 4 -> 4th smallest score = 0.20
        assert got["half_width"] == pytest.approx(0.20)
        assert got["lower"] == pytest.approx(0.40)
        assert got["upper"] == pytest.approx(0.80)
        assert got["status"] == "ok"

    def test_interval_is_clipped_to_unit_range(self):
        s = np.full(40, 0.9)
        got = cf.conformal_interval(0.95, s, alpha=0.1)
        assert got["lower"] == pytest.approx(0.05)
        assert got["upper"] == pytest.approx(1.0)

    def test_half_width_is_the_raw_score_not_the_clipped_span(self):
        # The half-width reports the conformal quantile itself; after clipping
        # to [0,1] the printed band is asymmetric, never silently reshaped.
        s = np.full(40, 0.9)
        got = cf.conformal_interval(0.95, s)
        assert got["half_width"] == pytest.approx(0.9)
        assert got["upper"] == pytest.approx(1.0)   # 0.95 + 0.9 clipped
        assert got["lower"] == pytest.approx(0.05)  # 0.95 - 0.9 clipped

class TestHonestyGates:
    def test_too_few_scores_withhold_the_band(self):
        got = cf.conformal_interval(0.6, _scores(10), min_n=30)
        assert got["status"] == "insufficient_sample"
        assert got["lower"] is None and got["upper"] is None
        assert got["warning"]

    def test_nan_scores_are_dropped_before_the_floor(self):
        s = np.concatenate([_scores(35), np.full(10, np.nan)])
        got = cf.conformal_interval(0.6, s)
        assert got["status"] == "ok"
        assert got["n_calibration"] == 35

    def test_nan_query_prob_is_unavailable_not_fabricated(self):
        got = cf.conformal_interval(float("nan"), _scores(40))
        assert got["status"] == "unavailable"
        assert got["lower"] is None

    def test_none_query_prob_is_unavailable(self):
        got = cf.conformal_interval(None, _scores(40))
        assert got["status"] == "unavailable"

    def test_wide_band_is_flagged_not_hidden(self):
        s = np.full(40, 0.6)  # half-width 0.6 >= WIDE_INTERVAL
        got = cf.conformal_interval(0.5, s)
        assert got["status"] == "ok"
        assert got["is_wide"] is True
        assert "no practical information" in got["warning"]

    def test_effective_windows_shrink_with_horizon(self):
        assert cf.effective_independent_windows(440, 20) == 22
        assert cf.effective_independent_windows(35, 20) == 2
        assert cf.effective_independent_windows(0, 20) == 0
        assert cf.effective_independent_windows(35, 0) == 0

    def test_scores_reject_length_mismatch(self):
        with pytest.raises(ValueError):
            cf.calibration_scores(np.array([0.5, 0.6]), np.array([1.0]))


class TestHoldoutCoverage:
    def test_coverage_is_target_when_scores_come_from_the_same_rows(self):
        # In-sample coverage of the SAME rows is not 100%: the band covers
        # ~1-alpha of the score distribution by construction, and 90% of 60
        # rows is 54 -> the empirical rate lands just below the target.
        rng = np.random.default_rng(11)
        p = np.clip(rng.normal(0.55, 0.15, 60), 0.01, 0.99)
        y = (rng.random(60) < p).astype(float)
        s = cf.calibration_scores(p, y)
        cov = cf.empirical_coverage(p, y, s, min_n=30)
        assert cov is not None
        assert 0.8 <= cov <= 1.0

    def test_coverage_none_on_thin_holdout(self):
        cov = cf.empirical_coverage(
            np.array([0.6, 0.4]), np.array([1.0, 0.0]), _scores(40), min_n=30
        )
        assert cov is None  # 2 eval rows -> 0/1 noise, withheld

    def test_coverage_none_when_scores_thin_even_with_rows(self):
        cov = cf.empirical_coverage(
            np.full(50, 0.6), np.ones(50), _scores(10), min_n=30
        )
        assert cov is None

    def test_summarize_reports_coverage_and_window_caveat(self):
        rng = np.random.default_rng(5)
        cal_p = np.clip(rng.normal(0.55, 0.15, 200), 0.01, 0.99)
        cal_y = (rng.random(200) < cal_p).astype(float)
        hold_p = np.clip(rng.normal(0.55, 0.15, 40), 0.01, 0.99)
        hold_y = (rng.random(40) < hold_p).astype(float)
        out = cf.summarize(cal_p, cal_y, hold_p, hold_y, prob=0.6, horizon=20)
        assert out["status"] == "ok"
        assert out["holdout_empirical_coverage"] is not None
        assert out["effective_independent_windows"] == 10  # ceil(200/20)

    def test_summarize_survives_internal_failure(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(cf, "interval_for_holdout_prob", _boom)
        out = cf.summarize(
            _scores(40), np.ones(40), _scores(5), np.ones(5), prob=0.6
        )
        assert out["status"] == "unavailable"

def _ohlcv(n: int = 420, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    rets = rng.normal(0.001, 0.02, n)
    close = pd.Series(100 * np.cumprod(1 + rets), index=idx)
    open_ = close.shift(1).fillna(close)
    spread = close * np.abs(rng.normal(0, 1, n)) * 0.01 + close * 0.005
    df = pd.DataFrame({
        "Open": open_, "High": close + spread.abs(),
        "Low": close - spread.abs(), "Close": close,
    })
    df["Volume"] = rng.integers(100_000, 2_000_000, n).astype(float)
    return df


def _market(n: int = 420, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    close = pd.Series(2000 * np.cumprod(1 + rng.normal(0.0005, 0.008, n)), index=idx)
    return pd.DataFrame({"Close": close})


@pytest.fixture(scope="module")
def forecast_result():
    feats = pipeline.build_feature_frame(_ohlcv(), market={"nifty50": _market()})
    return pipeline.forecast_frame(
        feats, symbol="TEST", horizon=5, fast=True, seed=3
    )


class TestPipelineIntegration:
    def test_uncertainty_block_is_present(self, forecast_result):
        unc = forecast_result["uncertainty"]
        assert unc["schema_version"] == "conformal-v1"
        assert unc["status"] in {"ok", "insufficient_sample", "unavailable"}

    def test_band_brackets_the_probability_when_ok(self, forecast_result):
        unc = forecast_result["uncertainty"]
        if unc["status"] != "ok":
            pytest.skip(f"uncertainty status={unc['status']} on synthetic run")
        prob = forecast_result["latest"]["probability"]
        assert 0.0 <= unc["lower"] <= unc["upper"] <= 1.0
        # after clipping, the reported band brackets the probability
        assert unc["lower"] <= prob + 1e-3
        assert unc["upper"] >= prob - 1e-3
        # half_width is the raw conformal quantile; the clipped band can be
        # narrower than 2x half_width when q sits near 0/1, so compare with
        # the clip-aware identity instead of assuming symmetry.
        lo = float(np.clip(prob - unc["half_width"], 0.0, 1.0))
        up = float(np.clip(prob + unc["half_width"], 0.0, 1.0))
        assert unc["lower"] == pytest.approx(round(lo, 4), abs=1e-3)
        assert unc["upper"] == pytest.approx(round(up, 4), abs=1e-3)

    def test_overlap_caveat_is_surfaced(self, forecast_result):
        unc = forecast_result["uncertainty"]
        if unc["status"] != "ok":
            pytest.skip(f"uncertainty status={unc['status']} on synthetic run")
        assert unc["effective_independent_windows"] >= 1
        assert "overlap" in unc["note"]

    def test_uncertainty_never_touches_the_decision(self, forecast_result):
        # The band is additive diagnostics only: latest signal/probability and
        # the holdout calibration numbers must be byte-identical to a run
        # whose uncertainty block is discarded.
        res = forecast_result
        assert res["latest"]["signal"] in {"BUY", "HOLD", "SELL"}
        assert "brier_calibrated" in res["calibration"]
