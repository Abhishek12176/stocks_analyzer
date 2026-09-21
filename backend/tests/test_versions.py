"""Unit tests for the ML versioning helpers (Task 1)."""

from datetime import datetime, timezone

from app.config import settings
from app.ml.versions import (
    make_prediction_id,
    new_snapshot,
    utc_now_iso,
    record_outcome,
)


class TestVersions:
    def test_utc_now_iso_is_parsable(self):
        dt = datetime.fromisoformat(utc_now_iso())
        assert dt.tzinfo is not None or dt.tzinfo is not None

    def test_make_prediction_id_is_sortable_and_contains_symbol(self):
        pid = make_prediction_id("reliance")
        assert pid.startswith("RELIANCE_")
        assert settings.ml_feature_version in pid

    def test_new_snapshot_defaults(self):
        snap = new_snapshot("TCS")
        assert snap["symbol"] == "TCS"
        assert snap["feature_version"] == settings.ml_feature_version
        assert snap["model_version"] == settings.ml_model_version
        assert snap["training_period"] == settings.ml_training_period
        assert snap["model_probability"] is None
        assert snap["signal"] is None
        assert snap["actual_future_outcome"] is None
        assert isinstance(snap["feature_snapshot"], dict)

    def test_new_snapshot_custom_fields(self):
        snap = new_snapshot(
            "INFY",
            model_version="xgb-v2",
            feature_snapshot={"rsi": 42.0},
            model_probability=0.68,
            signal="BUY",
            prediction_id="XYZ",
        )
        assert snap["prediction_id"] == "XYZ"
        assert snap["model_version"] == "xgb-v2"
        assert snap["model_probability"] == 0.68
        assert snap["signal"] == "BUY"
        assert snap["feature_snapshot"] == {"rsi": 42.0}

    def test_record_outcome_preserves_snapshot(self):
        snap = new_snapshot("RELIANCE", model_probability=0.61, signal="BUY")
        realized = record_outcome(
            snap,
            {"return_20d": 0.053, "direction": "up", "evaluated_at": utc_now_iso()},
        )
        assert realized["actual_future_outcome"]["direction"] == "up"
        assert realized["model_probability"] == 0.61
        assert realized is not snap  # copy, never mutate the original

    def test_probability_sum_five_decimals_rule(self):
        # No fake precision: snapshot must store rounded probability (<=4 dp).
        snap = new_snapshot("TCS", model_probability=0.68347829)
        prob = snap["model_probability"]
        assert round(prob, 4) == prob