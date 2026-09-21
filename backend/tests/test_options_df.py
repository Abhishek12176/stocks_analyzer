"""Unit tests for the F&O / options-derivatives feature layer (Task 7).

Contract: any data failure must gracefully disable the layer — no fabricated
OI/PCR/IVGreeks. Black-Scholes greeks on a synthetic ATM reference are
validated against standard sign/scale conventions.
"""

import numpy as np
import pandas as pd
import pytest

from app.config import settings
from app.ml import options_df as od


@pytest.fixture
def stock_frame(n: int = 60) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=n)
    return pd.DataFrame({"Close": 100.0}, index=idx)


class TestGreeks:
    def test_atm_call_greeks_zero_riskfree(self):
        g = od.bs_greeks(spot=100.0, strike=100.0, ttm_years=30 / 365,
                         sigma=0.01, risk_free=0.0)
        assert g["delta"] == pytest.approx(0.5, abs=1e-3)
        assert g["gamma"] > 0
        assert g["theta"] < 0
        assert g["vega"] > 0
        assert g["rho"] > 0                          # rho = K*T*N(d2) still positive

    def test_deep_itm_delta_approaches_one(self):
        g = od.bs_greeks(spot=130.0, strike=100.0, ttm_years=30 / 365,
                         sigma=0.2, risk_free=0.0)
        assert g["delta"] >= 0.95

    def test_deep_otm_delta_approaches_zero(self):
        g = od.bs_greeks(spot=70.0, strike=100.0, ttm_years=30 / 365,
                         sigma=0.2, risk_free=0.0)
        assert g["delta"] <= 0.05

    def test_value_error_on_bad_inputs(self):
        with pytest.raises(ValueError):
            od.bs_greeks(spot=100.0, strike=100.0, ttm_years=0.0, sigma=0.2)
        with pytest.raises(ValueError):
            od.bs_greeks(spot=100.0, strike=100.0, ttm_years=0.05, sigma=0.0)


def _calls() -> pd.DataFrame:
    strikes = np.arange(90, 111, 2, dtype=float)
    return pd.DataFrame(
        {"strike": strikes, "openInterest": 100.0, "impliedVolatility": 0.25}
    )


def _puts() -> pd.DataFrame:
    strikes = np.arange(90, 111, 2, dtype=float)
    return pd.DataFrame(
        {"strike": strikes, "openInterest": 200.0, "impliedVolatility": 0.26}
    )


class TestChainToMetrics:
    def test_pcr_and_total_oi(self):
        m = od.chain_to_metrics(_calls(), _puts(), spot=100.0, days_to_expiry=40)
        assert m["pcr"] == pytest.approx(2.0)          # put OI 200 vs call 100
        assert m["total_oi"] == pytest.approx((100 + 200) * 11)
        assert m["atm_iv"] == pytest.approx(0.25)      # strike 100 nearest to spot

    def test_greeks_derived_for_atm(self):
        m = od.chain_to_metrics(_calls(), _puts(), spot=100.0, days_to_expiry=40)
        for key in ("delta", "gamma", "theta", "vega", "rho"):
            assert not np.isnan(m[key])
        assert 0.4 < m["delta"] < 0.6                  # ATM call delta

    def test_empty_chain_yields_nan_metrics(self):
        m = od.chain_to_metrics(pd.DataFrame(), pd.DataFrame(), spot=100.0,
                                days_to_expiry=40)
        assert all(np.isnan(m[k]) for k in
                   ("pcr", "total_oi", "atm_iv", "delta", "gamma", "theta", "vega", "rho"))

    def test_zero_call_oi_guards_pcr(self):
        calls = pd.DataFrame({"strike": [100.0], "openInterest": [0.0],
                              "impliedVolatility": [0.25]})
        puts = pd.DataFrame({"strike": [100.0], "openInterest": [100.0],
                             "impliedVolatility": [0.25]})
        m = od.chain_to_metrics(calls, puts, spot=100.0, days_to_expiry=40)
        assert np.isnan(m["pcr"])                      # division guard, not inf


class TestGracefulDisable:
    def test_no_snapshots_no_columns(self, stock_frame):
        out = od.add_options_features(stock_frame)
        assert not any(c.startswith("opt_") for c in out.columns)

    def test_provider_failure_graceful(self, monkeypatch):
        def boom(*_a, **_k):
            raise ConnectionError("free tape refused")

        monkeypatch.setattr(od, "_load_option_chain", boom)
        res = od.get_options_inputs("RELIANCE.NS", spot=2500.0)
        assert res["is_available"] is False
        assert res["metrics"] is None
        assert "ConnectionError" in res["error"]

    def test_master_toggle_disables_layer(self, monkeypatch, stock_frame):
        monkeypatch.setattr(settings, "ml_options_enabled", False)
        res = od.get_options_inputs("RELIANCE.NS", spot=2500.0)
        assert res["is_available"] is False
        assert "disabled" in res["error"]

    def test_all_nan_chain_counted_unavailable(self, monkeypatch):
        monkeypatch.setattr(
            od, "_load_option_chain",
            lambda *_a, **_k: {"symbol": "X", "expiry": "2024-06-27",
                               "calls": pd.DataFrame(), "puts": pd.DataFrame()},
        )
        res = od.get_options_inputs("X", spot=100.0, days_to_expiry=20)
        assert res["is_available"] is False


class TestPitMerge:
    def test_snapshot_visible_from_publication_day(self, stock_frame):
        df = stock_frame
        snapshots = [
            {"available_at": df.index[10], "pcr": 1.25, "total_oi": 5000.0,
             "atm_iv": 0.20},
        ]
        out = od.add_options_features(df, snapshots=snapshots)
        assert out["opt_pcr"].iloc[:10].isna().all()
        assert np.allclose(out["opt_pcr"].iloc[10:].to_numpy(), 1.25)
        assert np.allclose(out["opt_total_oi"].iloc[10:].to_numpy(), 5000.0)

    def test_later_snapshot_supersedes(self, stock_frame):
        df = stock_frame
        snapshots = [
            {"available_at": df.index[10], "pcr": 1.0},
            {"available_at": df.index[30], "pcr": 2.5},
        ]
        out = od.add_options_features(df, snapshots=snapshots)
        assert out["opt_pcr"].iloc[15] == pytest.approx(1.0)
        assert out["opt_pcr"].iloc[35] == pytest.approx(2.5)

    def test_missing_keys_skipped(self, stock_frame):
        df = stock_frame
        out = od.add_options_features(df, snapshot={"pcr": 1.1, "available_at": df.index[5]})
        assert "opt_pcr" in out.columns
        assert "opt_delta" not in out.columns

    def test_feature_version_attr(self, stock_frame):
        df = stock_frame
        out = od.add_options_features(df, snapshots=[
            {"available_at": df.index[5], "pcr": 1.1}])
        assert out.attrs["feature_version"] == settings.ml_feature_version

    def test_no_lookahead_via_truncation(self, stock_frame):
        df = stock_frame
        snaps = [
            {"available_at": df.index[10], "pcr": 1.0},
            {"available_at": df.index[25], "pcr": 2.0},
        ]
        k = 30
        full = od.add_options_features(df, snapshots=snaps)
        trunc = od.add_options_features(df.iloc[:k], snapshots=[
            s for s in snaps if s["available_at"] <= df.index[k - 1]])
        assert list(full.columns) == list(trunc.columns)
        assert (full["opt_pcr"].iloc[:k].isna() == trunc["opt_pcr"].isna()).all()
        for i in range(k):
            if not np.isnan(full["opt_pcr"].iloc[i]):
                assert full["opt_pcr"].iloc[i] == trunc["opt_pcr"].iloc[i]


def test_options_feature_list():
    assert od.list_options_features() == [
        "opt_pcr", "opt_total_oi", "opt_atm_iv",
        "opt_delta", "opt_gamma", "opt_theta", "opt_vega", "opt_rho",
    ]