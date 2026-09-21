"""Task-9 snapshot metadata additions (ensemble/calibration extras)."""

import numpy as np
import pytest

from app.ml import versions as v


def test_snapshot_extra_metadata_allowlisted():
    snap = v.new_snapshot(
        "RELIANCE",
        model_probability=0.63,
        signal="BUY",
        ensemble_version="ensemble-v1",
        calibration_method="isotonic",
        calibrated_probability=0.71,
        calibrated_signal="BUY",
        random_junk="dropped",
    )
    assert snap["ensemble_version"] == "ensemble-v1"
    assert snap["calibration_method"] == "isotonic"
    assert snap["calibrated_probability"] == 0.71
    assert snap["calibrated_signal"] == "BUY"
    assert "random_junk" not in snap

def test_snapshot_extra_not_stored_when_none():
    snap = v.new_snapshot("TCS", model_probability=0.5)
    assert "ensemble_version" not in snap

def test_calibrated_probability_validated():
    with pytest.raises(ValueError):
        v.new_snapshot("TCS", calibrated_probability=1.7)

def test_calibrated_probability_rounded_4dp():
    snap = v.new_snapshot("TCS", calibrated_probability=0.671239)
    assert snap["calibrated_probability"] == 0.6712