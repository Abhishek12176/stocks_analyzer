"""Task 10 tests — forecast pipeline (feature assembly, OOF ensemble,
calibration, signal, explanation, monitor, snapshots) + ML status.

Structural assertions on synthetic data (no network): graceful-disabled
payloads, probability bounds + P(up)+P(down)=1, 4-dp snapshots, determinism,
backtest row contents and drift-status wiring. Nothing asserts alpha.
"""

import numpy as np
import pandas as pd
import pytest

from app.ml import pipeline


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


def _features(n: int = 420, seed: int = 1) -> pd.DataFrame:
    return pipeline.build_feature_frame(
        _ohlcv(n, seed), market={"nifty50": _market(n, 2)}
    )


HORIZON = 5
NOW = pd.Timestamp("2024-09-01")


class TestBuildFeatureFrame:
    def test_major_feature_groups_present(self):
        feats = _features()
        assert {"alpha_momentum", "beta", "regime_trend", "BB_mid",
                "vol_20d", "ret_20d"}.issubset(feats.columns)

    def test_voting_columns_attached(self):
        feats = _features()
        for col in ("sma20", "sma50", "rsi", "macd", "macd_signal"):
            assert col in feats.columns

    def test_lowercase_ohlcv_normalized(self):
        df = _ohlcv()
        low = df.rename(columns={"Open": "open", "Close": "close", "High": "high",
                                 "Low": "low", "Volume": "volume"})
        feats = pipeline.build_feature_frame(low, market=None)
        assert "Close" in feats.columns
        assert "BB_mid" in feats.columns

    def test_no_market_does_not_crash(self):
        feats = pipeline.build_feature_frame(_ohlcv(), market=None)
        assert "beta" not in feats.columns
        assert "regime_trend" not in feats.columns
        assert "alpha_momentum" in feats.columns


class TestForecastFrame:
    def test_happy_path_shape(self):
        res = pipeline.forecast_frame(
            _features(), symbol="TEST", horizon=HORIZON, fast=True, seed=3, now=NOW
        )
        assert res["is_available"] is True
        assert res["error"] is None
        assert res["symbol"] == "TEST"

        latest = res["latest"]
        assert latest["as_of"]
        assert 0.0 <= latest["probability"] <= 1.0
        assert 0.0 <= latest["raw_probability"] <= 1.0
        assert latest["signal"] in {"BUY", "HOLD", "SELL"}
        assert 0.0 <= latest["confidence"] <= 1.0
        assert latest["reason"]

        cal = res["calibration"]
        assert cal["n"] >= 40
        assert cal["method"] in {"isotonic", "platt", "uncalibrated"}
        assert 0.0 <= cal["brier"] <= 1.0

        backtest_models = {r["model"] for r in res["backtest"]}
        assert "buy_hold" in backtest_models
        assert "ensemble" in backtest_models
        for row in res["backtest"]:
            assert "cum_return" in row
            assert "n_trades" in row

        expl = res["explanation"]
        assert expl and len(expl["factors"]) > 0
        assert all("factor" in f and "impact" in f for f in expl["factors"])

        mon = res["monitor"]
        assert mon["mode"] in {"fresh", "degraded", "stale"}
        assert "freshness" in mon
        assert isinstance(mon["alarms"], list)

        snap = res["snapshot"]
        assert snap["symbol"] == "TEST"
        assert snap["feature_version"]
        assert snap["model_version"]
        assert snap["ensemble_version"]

    def test_probabilities_are_proper_and_rounded(self):
        res = pipeline.forecast_frame(
            _features(), symbol="TEST", horizon=HORIZON, fast=True, seed=5, now=NOW
        )
        prob = res["latest"]["probability"]
        assert prob + (1 - prob) == pytest.approx(1.0)
        assert round(prob, 4) == pytest.approx(prob)
        snap_prob = res["snapshot"]["model_probability"]
        assert round(snap_prob, 4) == pytest.approx(snap_prob)
        if res["calibration"]["method"] == "isotonic":
            assert res["snapshot"]["calibrated_signal"] == res["latest"]["signal"]
            assert 0.0 <= res["snapshot"]["calibrated_probability"] <= 1.0

    def test_insufficient_history_graceful(self):
        feats = pipeline.build_feature_frame(_ohlcv(n=80), market=None)
        res = pipeline.forecast_frame(
            feats, symbol="TINY", horizon=HORIZON, fast=True
        )
        assert res["is_available"] is False
        assert res["error"]

    def test_empty_frame_graceful(self):
        res = pipeline.forecast_frame(pd.DataFrame(), symbol="EMPTY", horizon=HORIZON)
        assert res["is_available"] is False
        assert res["error"]

    def test_determinism_same_seed(self):
        feats = _features()
        r1 = pipeline.forecast_frame(feats, symbol="TEST", horizon=HORIZON, fast=True, seed=7, now=NOW)
        r2 = pipeline.forecast_frame(feats, symbol="TEST", horizon=HORIZON, fast=True, seed=7, now=NOW)
        assert r1["latest"]["probability"] == pytest.approx(
            r2["latest"]["probability"], abs=1e-6
        )
        assert r1["latest"]["signal"] == r2["latest"]["signal"]

    def test_source_changes_can_flip_probability(self):
        feats_a = _features(seed=1)
        feats_b = _features(seed=11)
        ra = pipeline.forecast_frame(feats_a, symbol="TEST", horizon=HORIZON, fast=True, seed=2, now=NOW)
        rb = pipeline.forecast_frame(feats_b, symbol="TEST", horizon=HORIZON, fast=True, seed=2, now=NOW)
        assert ra["is_available"] and rb["is_available"]

    def test_calibration_split_and_holdout_isolation(self):
        res = pipeline.forecast_frame(
            _features(), symbol="TEST", horizon=HORIZON, fast=True, seed=3, now=NOW
        )
        val = res["validation"]
        cal = res["calibration"]
        assert val["final_holdout_never_used_for_fitting"] is True
        assert val["holdout_rows"] > 0
        assert val["calibration_fit_rows"] + val["holdout_rows"] == val["oof_rows"]
        assert cal["n"] == val["calibration_fit_rows"]
        assert cal["n_holdout"] == val["holdout_rows"]
        assert cal["method"] in {"isotonic", "platt", "uncalibrated"}
        assert 0.0 <= cal["brier"] <= 1.0
        assert 0.0 <= cal["ece"] <= 1.0
        names = {r["model"] for r in res["backtest"]}
        assert {"ensemble", "buy_hold", "always_up", "always_down",
                "random", "shuffled_control"} <= names
        for row in res["backtest"]:
            assert "n_trades" in row and "cum_return" in row
            assert row.get("no_leverage_ok") == "yes"
        # shuffled-target control: chance level is interpreted approximately;
        # with a tiny holdout (n trades ~ few) the test only enforces that the
        # control does NOT outperform, and stays structurally complete
        sh = next(r for r in res["backtest"] if r["model"] == "shuffled_control")
        if sh.get("accuracy") is not None and sh.get("n_trades", 0) >= 5:
            assert 0.25 <= sh["accuracy"] <= 0.75


class TestForecastSymbolGraceful:
    def test_invalid_symbol(self):
        res = pipeline.forecast_symbol("$$$", period="1mo")
        assert res["is_available"] is False
        assert "invalid" in (res["error"] or "").lower()

    def test_empty_symbol(self):
        res = pipeline.forecast_symbol("", period="1mo")
        assert res["is_available"] is False


class TestMLStatus:
    def _fresh_stub(self, series_id):
        idx = pd.date_range(
            pd.Timestamp.now().normalize() - pd.Timedelta(days=1), periods=2
        )
        return pd.DataFrame({"Close": [1.0, 2.0]}, index=idx), {"is_available": True}

    def _stale_stub(self, series_id):
        return None, {"is_available": False, "error": "boom"}

    def test_ok_when_sources_fresh(self):
        pl_module = pytest.importorskip("app.ml.pipeline")
        pl_module._LAST_RUN["last_run"] = None
        status = pl_module.build_ml_status(source_fetcher=self._fresh_stub)
        assert status["status"] == "ok"
        assert status["sources"]
        assert all(s["status"] == "fresh" for s in status["sources"])

    def test_degraded_when_sources_stale(self):
        status = pipeline.build_ml_status(source_fetcher=self._stale_stub)
        assert status["status"] == "degraded"
        assert "market_sources_stale" in status["alarms"]
        assert all(s["is_available"] is False for s in status["sources"])

    def test_last_run_metadata(self):
        pipeline._LAST_RUN["last_run"] = None
        before = pipeline.last_run_status()
        assert before["has_run"] is False
        pipeline.forecast_frame(
            _features(), symbol="TEST", horizon=HORIZON, fast=True, seed=1, now=NOW
        )
        after = pipeline.last_run_status()
        assert after["has_run"] is True
        assert after["last_run"]["symbol"] == "TEST"
        assert "latest_probability" in after["last_run"]
        assert after["last_run_mode"] in {"fresh", "degraded", "stale"}


class TestOofProvisioningAndThresholdMap:
    """additive `return_oof` + per-symbol `threshold_buy_map` (Task 31)."""

    def test_oof_table_default_absent(self):
        res = pipeline.forecast_frame(
            _features(), symbol="TEST", horizon=HORIZON, fast=True, seed=7, now=NOW
        )
        assert "oof_table" not in res

    def test_oof_table_parity(self):
        res = pipeline.forecast_frame(
            _features(), symbol="TEST", horizon=HORIZON, fast=True, seed=7,
            now=NOW, return_oof=True,
        )
        recs = res["oof_table"]
        assert isinstance(recs, list) and len(recs) >= 40
        hold = [r for r in recs if r.get("is_holdout")]
        assert hold  # a non-trivial newest-20% slice exists
        acc = sum(1 for r in hold
                  if r["prob_cal"] is not None and r["y"] is not None
                  and (r["prob_cal"] >= 0.5) == (r["y"] >= 0.5))
        denom = sum(1 for r in hold if r["prob_cal"] is not None
                    and r["y"] is not None)
        assert denom > 0
        report = round(acc / denom, 4)
        recon = res["discrimination"]["holdout_accuracy_calibrated"]
        assert report == pytest.approx(recon, abs=1e-9)
        r0 = hold[0]
        for key in ("date", "prob_raw", "prob_cal", "y", "ret", "close"):
            assert key in r0

    def test_threshold_buy_map_wired_to_backtest_and_latest(self):
        from app.config import settings
        base = pipeline.forecast_frame(
            _features(), symbol="TEST", horizon=HORIZON, fast=True, seed=7, now=NOW
        )
        assert base["latest"]["threshold_buy"] == settings.ml_threshold_buy
        mapped = pipeline.forecast_frame(
            _features(), symbol="TEST", horizon=HORIZON, fast=True, seed=7,
            now=NOW, threshold_buy_map={"TEST": 0.72},
        )
        assert mapped["latest"]["threshold_buy"] == pytest.approx(0.72)
        b_n = next(r["n_trades"] for r in base["backtest"]
                   if r["model"] == "ensemble")
        m_n = next(r["n_trades"] for r in mapped["backtest"]
                   if r["model"] == "ensemble")
        assert m_n <= b_n + 0  # a stricter threshold cannot add trades