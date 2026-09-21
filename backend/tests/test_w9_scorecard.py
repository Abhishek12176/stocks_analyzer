"""W9 tests — forward (in-time) prediction scorecard (log + realized-outcome
scoring + report). All synthetic/offline (no network): the price history is
injected, so every scoring decision is deterministic. Nothing here asserts
alpha.

W9 = nothing ever scored what actually happened 20 trading days after a
prediction. These tests pin the missing loop: append-only idempotent logging,
outcome evaluation against the T -> T+horizon close, baseline-relative
forward-test metrics, and the insufficient-sample guard.
"""

import json

import numpy as np
import pandas as pd
import pytest

from app.ml import scorecard as sc


def _snapshot(probability=0.62, signal="BUY", direction=1, close=100.0,
              symbol="RELIANCE"):
    return {
        "is_available": True,
        "symbol": symbol,
        "horizon": 20,
        "generated_at": "2024-01-01T00:00:00+00:00",
        "versions": {
            "feature": "v1", "model": "baseline-v1", "ensemble": "ensemble-v1",
            "calibration": "isotonic",
        },
        "latest": {
            "as_of": "2024-01-31T00:00:00",
            "close": close,
            "raw_probability": 0.6,
            "probability": probability,
            "signal": signal,
            "direction": direction,
        },
        "snapshot": {"prediction_id": f"{symbol}_v1_TESTID"},
        "data_fingerprint": "fp-test-1",
    }


def _closes(start="2024-01-02", n=120, drift=0.001, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    px = 100.0 * np.cumprod(1 + rng.normal(drift, 0.02, n))
    return pd.Series(px, index=idx)


class TestPredictionLog:
    def test_record_and_reload_roundtrip(self, tmp_path):
        res = sc.record_prediction(
            _snapshot(), as_of="2024-01-31", horizon=20, log_dir=tmp_path
        )
        assert res["status"] == "recorded"
        rows = sc.load_predictions("RELIANCE", tmp_path)
        assert len(rows) == 1
        row = rows[0]
        for key in sc.required_prediction_keys():
            assert key in row, key
        assert row["scored"] is False
        assert row["actual_future_outcome"] is None

    def test_duplicate_is_idempotent(self, tmp_path):
        kw = dict(as_of="2024-01-31", horizon=20, log_dir=tmp_path)
        assert sc.record_prediction(_snapshot(), **kw)["status"] == "recorded"
        assert sc.record_prediction(_snapshot(), **kw)["status"] == "duplicate"
        assert len(sc.load_predictions("RELIANCE", tmp_path)) == 1

    def test_same_day_new_snapshot_is_separate(self, tmp_path):
        snap = _snapshot()
        snap["data_fingerprint"] = "fp-test-2"
        assert sc.record_prediction(snap, as_of="2024-01-31", horizon=20,
                                    log_dir=tmp_path)["status"] == "recorded"
        assert sc.record_prediction(_snapshot(), as_of="2024-01-31", horizon=20,
                                    log_dir=tmp_path)["status"] == "recorded"
        assert len(sc.load_predictions("RELIANCE", tmp_path)) == 2

    def test_graceful_snapshot_is_skipped(self, tmp_path):
        res = sc.record_prediction({"is_available": False, "symbol": "X"},
                                   as_of="2024-01-31", horizon=20, log_dir=tmp_path)
        assert res["status"] == "skipped"
        assert sc.load_predictions("X", tmp_path) == []

    def test_all_symbols_load(self, tmp_path):
        sc.record_prediction(_snapshot(symbol="TCS"), as_of="2024-01-31",
                             horizon=20, log_dir=tmp_path)
        sc.record_prediction(_snapshot(symbol="INFY"), as_of="2024-01-31",
                             horizon=20, log_dir=tmp_path)
        assert len(sc.load_predictions(None, tmp_path)) == 2


class TestOutcomeEvaluation:
    def test_scoreable_prediction_gets_exact_label(self):
        closes = _closes()  # index[9] = 2024-01-15
        as_of = closes.index[9]
        entry = float(closes.iloc[9])
        exit_px = float(closes.iloc[9 + 20])
        expected_ret = exit_px / entry - 1.0

        rec = sc._resolve_prediction_fields(
            _snapshot(close=entry), "RELIANCE", str(as_of), 20,
        )
        ev = sc.evaluate_prediction_against_closes(rec, closes)
        assert ev is not None and ev["scored"] is True
        out = ev["actual_future_outcome"]
        assert out["label_up"] == (expected_ret > 0)
        assert out["return_20d"] == pytest.approx(expected_ret, rel=1e-4)
        assert out["entry_close"] == pytest.approx(entry, rel=1e-3)
        assert out["exit_close"] == pytest.approx(exit_px, rel=1e-3)

    def test_unelapsed_horizon_stays_unscored(self):
        closes = _closes(n=120)
        as_of = closes.index[-5]  # only 4 bars left: not enough for horizon 20
        rec = sc._resolve_prediction_fields(
            _snapshot(), "RELIANCE", str(as_of), 20,
        )
        assert sc.evaluate_prediction_against_closes(rec, closes) is None

    def test_record_outcome_called_not_fabricated(self):
        # `versions.record_outcome` is the ONLY writer of the outcome field.
        closes = _closes()
        rec = sc._resolve_prediction_fields(
            _snapshot(), "RELIANCE", str(closes.index[9]), 20,
        )
        ev = sc.evaluate_prediction_against_closes(rec, closes)
        assert ev["actual_future_outcome"]["evaluated_at"]


class TestScoringAndReport:
    def _seed(self, tmp_path, symbol="RELIANCE", n=8):
        closes = _closes(n=300, drift=0.002)
        # Space predictions 20 bars apart so outcomes are independent windows.
        for i in range(n):
            as_of = closes.index[i * 20]
            sc.record_prediction(
                _snapshot(symbol=symbol), as_of=str(as_of), horizon=20,
                symbol=symbol, log_dir=tmp_path,
            )
        return closes

    def test_pending_scoring_marks_log(self, tmp_path):
        closes = self._seed(tmp_path)
        out = sc.score_pending_predictions(
            "RELIANCE", close_lookup=lambda s: closes, log_dir=tmp_path
        )
        assert out["symbols"]["RELIANCE"]["n_newly_scored"] == 8
        assert out["symbols"]["RELIANCE"]["n_waiting"] == 0
        rows = sc.load_predictions("RELIANCE", tmp_path)
        assert all(r["scored"] for r in rows)

    def test_report_matches_manual_math(self, tmp_path):
        closes = self._seed(tmp_path)
        sc.score_pending_predictions(
            "RELIANCE", close_lookup=lambda s: closes, log_dir=tmp_path
        )
        rep = sc.scorecard_report("RELIANCE", log_dir=tmp_path)
        assert rep["n_logged"] == 8 and rep["n_scored"] == 8
        assert rep["is_sufficient_sample"] is True
        m = rep["metrics"]
        rows = sc.load_predictions("RELIANCE", tmp_path)
        ups = np.array([r["actual_future_outcome"]["label_up"] for r in rows])
        probs = np.array([r["probability"] for r in rows])
        assert m["accuracy"] == pytest.approx(float((((probs >= 0.5) == ups)).mean()))
        assert m["always_up_accuracy"] == pytest.approx(float(ups.mean()))
        assert m["edge_vs_best_baseline_accuracy"] == pytest.approx(
            m["accuracy"] - m["best_baseline_accuracy"]
        )
        assert m["brier"] == pytest.approx(
            float(np.mean((probs - ups.astype(float)) ** 2))
        )
        assert rep["per_symbol"]["RELIANCE"]["n"] == 8

    def test_insufficient_sample_warns_not_concludes(self, tmp_path):
        closes = self._seed(tmp_path, n=2)
        sc.score_pending_predictions(
            "RELIANCE", close_lookup=lambda s: closes, log_dir=tmp_path
        )
        rep = sc.scorecard_report("RELIANCE", log_dir=tmp_path)
        assert rep["n_scored"] == 2
        assert rep["is_sufficient_sample"] is False
        assert "warning" in rep

    def test_empty_log_reports_cleanly(self, tmp_path):
        rep = sc.scorecard_report("NODATA", log_dir=tmp_path)
        assert rep["n_scored"] == 0
        assert rep["metrics"] == {"status": "no_scored_predictions"}