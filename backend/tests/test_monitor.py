"""Unit tests for freshness / PSI drift / degraded-mode monitoring (Task 9)."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.ml import monitor as mo


def _now() -> datetime:
    return datetime(2026, 9, 5, tzinfo=timezone.utc)


class TestFreshness:
    def test_fresh(self):
        out = mo.freshness_status(_now() - timedelta(days=3), now=_now(), max_days=5)
        assert out["status"] == "fresh"

    def test_stale(self):
        out = mo.freshness_status(_now() - timedelta(days=9), now=_now(), max_days=5)
        assert out["status"] == "stale"
        assert out["age_days"] == 9

    def test_missing_date_is_stale(self):
        assert mo.freshness_status(None, now=_now())["status"] == "stale"

    def test_naive_date_zero(self):
        out = mo.freshness_status(_now() - timedelta(days=4), now=_now(), max_days=4)
        assert out["status"] == "fresh"


class TestPsi:
    def test_identical_distribution_is_zero(self):
        x = np.random.default_rng(0).normal(0, 1, 500)
        assert mo.psi(x, x) == pytest.approx(0.0, abs=1e-9)

    def test_shifted_distribution_positive(self):
        a = np.random.default_rng(1).normal(0, 1, 500)
        b = np.random.default_rng(2).normal(3, 1, 500)
        assert mo.psi(a, b) > 0.2

    def test_nan_inputs_dropped(self):
        x = np.random.default_rng(3).normal(0, 1, 300)
        with_nan = x.copy()
        with_nan[::7] = np.nan
        assert mo.psi(x, with_nan) < 0.01

    def test_all_nan_nan(self):
        assert np.isnan(mo.psi(np.array([np.nan]), np.array([np.nan])))


class TestDriftChecks:
    def test_feature_drift_no_alarm_identical(self):
        x = np.random.default_rng(4).normal(0, 1, 300)
        out = mo.feature_drift({"a": x}, {"a": x}, threshold=0.25)
        assert out["alarm"] is False and out["max_psi"] < 0.01

    def test_feature_drift_alarm_on_shift(self):
        a = np.random.default_rng(5).normal(0, 1, 300)
        b = np.random.default_rng(6).normal(4, 1, 300)
        out = mo.feature_drift({"a": a}, {"a": b}, threshold=0.25)
        assert out["alarm"] is True and out["flagged"] == ["a"]

    def test_prediction_drift(self):
        a = np.random.default_rng(7).uniform(0, 1, 400)
        b = np.clip(a + 0.6, 0, 1)
        out = mo.prediction_drift(a, b, threshold=0.25)
        assert out["alarm"] is True

    def test_calibration_drift(self):
        out = mo.calibration_drift(0.20, 0.30, threshold=0.05)
        assert out["alarm"] is True
        assert mo.calibration_drift(0.25, 0.20, threshold=0.05)["alarm"] is False


class TestAssess:
    FEAT_REF = {"a": np.random.default_rng(0).normal(0, 1, 300)}
    FEAT_CUR = {"a": np.random.default_rng(0).normal(0, 1, 300)}
    PRED_REF = np.random.default_rng(1).uniform(0, 1, 300)
    PRED_CUR = np.random.default_rng(1).uniform(0, 1, 300)

    def test_all_good_fresh(self):
        out = mo.assess(_now() - timedelta(days=1), self.FEAT_REF, self.FEAT_CUR,
                        self.PRED_REF, self.PRED_CUR, 0.20, 0.21, now=_now())
        assert out["mode"] == "fresh" and out["alarms"] == []

    def test_stale_data_mode(self):
        out = mo.assess(_now() - timedelta(days=99), self.FEAT_REF, self.FEAT_CUR,
                        self.PRED_REF, self.PRED_CUR, 0.20, 0.21, now=_now())
        assert out["mode"] == "stale" and "data_stale" in out["alarms"]

    def test_drift_alarm_degraded(self):
        feats = {"a": np.random.default_rng(8).normal(5, 1, 300)}
        out = mo.assess(_now() - timedelta(days=1), self.FEAT_REF, feats,
                        self.PRED_REF, self.PRED_CUR, 0.20, 0.40, now=_now())
        assert out["mode"] == "degraded"
        assert "feature_drift" in out["alarms"] or "calibration_drift" in out["alarms"]

    def test_report_shape(self):
        out = mo.assess(_now() - timedelta(days=1), self.FEAT_REF, self.FEAT_CUR,
                        self.PRED_REF, self.PRED_CUR, 0.20, 0.20, now=_now())
        for key in ("mode", "freshness", "feature_drift", "feature_psi",
                    "prediction_drift", "calibration_drift", "alarms"):
            assert key in out