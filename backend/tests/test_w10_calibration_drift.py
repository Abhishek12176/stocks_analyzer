"""W10 tests — live calibration-drift monitor (reference persistence +
reference-vs-forward-sample comparison + honest gating). All synthetic/offline
(no network): the reference file and the scored W9 log are injected, so every
alarm decision is deterministic. Nothing here asserts alpha.

W10 = `monitor.calibration_drift` existed but nothing in production ever
persisted a reference, so the alarm could never fire on real data. These
tests pin the closed loop: best-effort reference recording from the forecast
result, a report that compares the reference against the W9 scored
predictions, the insufficiency floor, and the no-reference honesty guard.
"""

import numpy as np
import pandas as pd
import pytest

from app.ml import calibration_monitor as cm


def _scored_rec(probability: float, label_up: bool, symbol: str = "RELIANCE"):
    return {
        "prediction_id": f"{symbol}_v1_TESTID",
        "symbol": symbol,
        "as_of": "2024-01-31",
        "horizon": 20,
        "probability": probability,
        "scored": True,
        "actual_future_outcome": {
            "label_up": bool(label_up),
            "return_20d": 0.01 if label_up else -0.01,
        },
    }


def _scored_sample(seed: int = 3, prob: float = 0.7, n: int = 8):
    """Deterministic scored sample with a known Brier."""
    rng = np.random.default_rng(seed)
    return [_scored_rec(prob, bool(rng.random() < 0.5)) for _ in range(n)]


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


class TestReferenceStore:
    def test_record_and_reload_roundtrip(self, tmp_path):
        state = tmp_path / "state.json"
        res = cm.record_calibration_reference(
            {"method": "platt", "n_holdout": 88, "brier": 0.28,
             "brier_calibrated": 0.27, "ece": 0.20, "ece_calibrated": 0.19},
            symbol="RELIANCE", state_path=state,
        )
        assert res["status"] == "recorded"
        ref = cm.load_reference(state)
        assert ref["brier"] == pytest.approx(0.27)
        assert ref["ece"] == pytest.approx(0.19)
        assert ref["method"] == "platt"
        assert ref["n_holdout"] == 88
        assert ref["recorded_at"]

    def test_update_overwrites_and_flags(self, tmp_path):
        state = tmp_path / "state.json"
        kw = dict(symbol="RELIANCE", state_path=state)
        cm.record_calibration_reference({"brier_calibrated": 0.30}, **kw)
        res = cm.record_calibration_reference({"brier_calibrated": 0.26}, **kw)
        assert res["status"] == "updated"
        assert cm.load_reference(state)["brier"] == pytest.approx(0.26)

    def test_missing_brier_is_skipped_not_fabricated(self, tmp_path):
        state = tmp_path / "state.json"
        res = cm.record_calibration_reference({}, state_path=state)
        assert res["status"] == "skipped"
        assert cm.load_reference(state) is None
        res2 = cm.record_calibration_reference(
            {"brier_calibrated": float("nan")}, state_path=state
        )
        assert res2["status"] == "skipped"

    def test_corrupt_state_file_reads_as_none(self, tmp_path):
        state = tmp_path / "state.json"
        state.write_text("{not json", encoding="utf-8")
        assert cm.load_reference(state) is None

    def test_clear_reference_removes_file(self, tmp_path):
        state = tmp_path / "state.json"
        cm.record_calibration_reference({"brier_calibrated": 0.3}, state_path=state)
        assert cm.clear_reference(state) is True
        assert cm.load_reference(state) is None
        assert cm.clear_reference(state) is False

class TestDriftReport:
    def test_no_sample_reports_without_alarm(self):
        rep = cm.calibration_drift_report([])
        assert rep["status"] == "no_scored_predictions"
        assert rep["is_alarm"] is False

    def test_no_reference_is_unknown_not_healthy(self, tmp_path):
        rep = cm.calibration_drift_report(
            _scored_sample(), reference=None, state_path=tmp_path / "missing.json"
        )
        assert rep["status"] == "no_reference"
        assert rep["is_alarm"] is False
        assert "UNKNOWN" in rep["warning"]

    def test_reference_unusable_brier_is_not_alarm(self, tmp_path):
        rep = cm.calibration_drift_report(
            _scored_sample(), reference={"brier": None},
            state_path=tmp_path / "unused.json",
        )
        assert rep["status"] == "reference_unusable"
        assert rep["is_alarm"] is False

    def test_insufficient_sample_never_alarms(self):
        ref = {"brier": 0.10, "ece": 0.10}
        rep = cm.calibration_drift_report(
            _scored_sample(n=4), reference=ref
        )
        assert rep["status"] == "insufficient_sample"
        assert rep["is_alarm"] is False
        assert rep["n_recent"] == 4

    def test_math_matches_manual_brier_delta(self):
        ref = {"brier": 0.20, "ece": 0.15}
        probs = np.full(8, 0.8)
        ups = np.array([1, 0, 1, 1, 0, 1, 0, 1], dtype=bool)
        sample = [_scored_rec(float(p), bool(u)) for p, u in zip(probs, ups)]
        rep = cm.calibration_drift_report(sample, reference=ref)
        expected = float(np.mean((probs - ups.astype(float)) ** 2)) - 0.20
        assert rep["delta_brier"] == pytest.approx(round(expected, 4))
        assert rep["delta_ece"] is not None

    def test_alarm_fires_only_above_threshold(self):
        ref = {"brier": 0.10, "ece": 0.10}
        probs = np.full(8, 0.9)
        ups = np.zeros(8, dtype=bool)  # every call wrong: Brier = 0.81
        sample = [_scored_rec(float(p), bool(u)) for p, u in zip(probs, ups)]
        rep = cm.calibration_drift_report(sample, reference=ref)
        assert rep["is_alarm"] is True
        assert rep["status"] == "drift_detected"
        ok = cm.calibration_drift_report(sample, reference=ref, threshold=0.9)
        assert ok["is_alarm"] is False
        assert ok["status"] == "ok"

    def test_default_threshold_comes_from_settings(self):
        from app.config import settings

        ref = {"brier": 0.10, "ece": 0.10}
        probs = np.full(8, 0.9)
        ups = np.zeros(8, dtype=bool)
        sample = [_scored_rec(float(p), bool(u)) for p, u in zip(probs, ups)]
        rep = cm.calibration_drift_report(sample, reference=ref)
        assert rep["threshold"] == pytest.approx(settings.ml_brier_drift_threshold)

class TestLiveBuilderAndHooks:
    def test_live_report_off_real_scorecard_log(self, tmp_path):
        from app.ml import scorecard as sc

        closes = pd.Series(
            100.0 * np.cumprod(1 + np.full(60, 0.002)),
            index=pd.bdate_range("2024-01-02", periods=60),
        )
        for i in range(2):
            sc.record_prediction(
                {"is_available": True, "symbol": "RELIANCE", "horizon": 20,
                 "versions": {},
                 "latest": {"as_of": "2024-01-31", "close": 100.0,
                            "probability": 0.7, "signal": "BUY", "direction": 1},
                 "snapshot": {"prediction_id": "RELIANCE_v1_T"},
                 "data_fingerprint": f"fp-{i}"},
                as_of=str(closes.index[i * 20].date()), horizon=20,
                symbol="RELIANCE", log_dir=tmp_path,
            )
        sc.score_pending_predictions(
            "RELIANCE", close_lookup=lambda s: closes, log_dir=tmp_path
        )
        state = tmp_path / "state.json"
        cm.record_calibration_reference(
            {"brier_calibrated": 0.20}, state_path=state
        )
        rep = cm.build_live_calibration_drift(
            "RELIANCE", log_dir=tmp_path, state_path=state
        )
        assert rep["status"] == "insufficient_sample"  # only 2 scored
        assert rep["is_alarm"] is False
        assert rep["live"]["brier"] == pytest.approx(0.3 ** 2, abs=1e-6)

    def test_ml_status_includes_calibration_drift_block(
        self, monkeypatch
    ):
        from app.ml import pipeline as pl

        idx = pd.date_range(pd.Timestamp.now().normalize(), periods=2)
        stub = lambda sid: (
            pd.DataFrame({"Close": [1.0, 2.0]}, index=idx),
            {"is_available": True},
        )
        monkeypatch.setattr(pl, "_default_source_fetcher", stub)
        monkeypatch.setattr(
            cm, "build_live_calibration_drift",
            lambda *a, **k: {"status": "drift_detected", "is_alarm": True,
                             "delta_brier": 0.2},
        )
        monkeypatch.setattr(
            "app.ml.pipeline.last_run_status",
            lambda: {"has_run": False, "last_run": None, "last_run_mode": None},
        )
        status = pl.build_ml_status()
        assert status["calibration_drift"]["status"] == "drift_detected"
        assert "calibration_drift" in status["alarms"]
        assert status["status"] == "degraded"

    def test_ml_status_survives_drift_module_failure(
        self, monkeypatch
    ):
        import sys

        from app.ml import pipeline as pl

        idx = pd.date_range(pd.Timestamp.now().normalize(), periods=2)
        stub = lambda sid: (
            pd.DataFrame({"Close": [1.0, 2.0]}, index=idx),
            {"is_available": True},
        )
        monkeypatch.setattr(pl, "_default_source_fetcher", stub)
        monkeypatch.setattr(
            "app.ml.pipeline.last_run_status",
            lambda: {"has_run": False, "last_run": None, "last_run_mode": None},
        )
        monkeypatch.setitem(
            sys.modules, "app.ml.calibration_monitor", None
        )
        status = pl.build_ml_status()
        assert "calibration_drift" in status
        assert status["calibration_drift"]["is_alarm"] is False

    def test_pipeline_run_persists_reference(self, tmp_path, monkeypatch):
        """The forecast_frame W10 hook must persist the reference best-effort."""
        from app.ml import pipeline as pl

        monkeypatch.setattr(pl, "BACKEND_ROOT", tmp_path, raising=False)
        monkeypatch.setattr(
            cm, "default_state_path", lambda: tmp_path / "state.json"
        )
        rng = np.random.default_rng(1)
        idx = pd.bdate_range("2023-01-02", periods=420)
        close = pd.Series(
            100 * np.cumprod(1 + rng.normal(0.001, 0.02, 420)), index=idx
        )
        df = pd.DataFrame({
            "Open": close.shift(1).fillna(close), "High": close * 1.01,
            "Low": close * 0.99, "Close": close,
            "Volume": rng.integers(100_000, 2_000_000, 420).astype(float),
        })
        feats = pl.build_feature_frame(df, market={"nifty50": df.copy()})
        pl.forecast_frame(feats, symbol="TEST", horizon=5, fast=True, seed=3)
        ref = cm.load_reference(tmp_path / "state.json")
        assert ref is not None and np.isfinite(ref["brier"])
        assert 0.0 <= ref["brier"] <= 1.0
        cal = pl._LAST_RUN["last_run"]
        assert cal["oof_brier_calibrated"] == pytest.approx(ref["brier"])


class TestCalibrationDriftRoute:
    def test_route_reports_alarm_contract(self, client, monkeypatch):
        from app.ml import calibration_monitor as cm_mod

        monkeypatch.setattr(
            cm_mod, "build_live_calibration_drift",
            lambda *a, **k: {
                "schema_version": "calibration-drift-v1",
                "generated_at": "2024-01-01T00:00:00+00:00",
                "symbol": None, "min_recent_for_alarm": 5, "threshold": 0.05,
                "n_recent": 8, "reference": {"brier": 0.2},
                "live": {"brier": 0.3, "ece": 0.1},
                "delta_brier": 0.1, "delta_ece": 0.05, "is_alarm": True,
                "status": "drift_detected", "note": "n", "warning": "w",
            },
        )
        resp = client.get("/api/v1/ml/calibration-drift")
        assert resp.status_code == 200
        body = resp.json()
        assert body["schemaVersion"] == "calibration-drift-v1"
        assert body["isAlarm"] is True
        assert body["status"] == "drift_detected"

    def test_route_degrades_gracefully(self, client, monkeypatch):
        from app.ml import calibration_monitor as cm_mod

        def _boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(cm_mod, "build_live_calibration_drift", _boom)
        resp = client.get("/api/v1/ml/calibration-drift")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "sample_unavailable"
        assert "unavailable" in body["warning"]
