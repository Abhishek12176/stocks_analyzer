"""Unit tests for Platt / isotonic calibration + ECE (Task 9)."""

import numpy as np
import pytest

from app.ml import calibration as cal


def _miscalibrated(seed: int = 3):
    """Synthetic (prob, label) pairs that are systematically over-confident."""
    rng = np.random.default_rng(seed)
    prob = rng.uniform(0.05, 0.95, 800)
    y = (rng.uniform(0.0, 1.0, 800) < prob).astype(int)
    over = 0.5 + (prob - 0.5) * 1.6  # stretches away from 0.5
    return np.clip(over, 0.02, 0.98), y


class TestLogitHelpers:
    def test_logit_inverse(self):
        p = np.array([0.1, 0.5, 0.9])
        assert np.allclose(cal._sigmoid(cal._logit(p)), p)

    def test_sigmoid_stable_extremes(self):
        assert cal._sigmoid(np.array([-800.0, 800.0]))[0] == pytest.approx(0.0, abs=1e-9)
        assert cal._sigmoid(np.array([800.0]))[0] == pytest.approx(1.0, abs=1e-9)


class TestPlatt:
    def test_platt_reduces_ece_on_overconfident(self):
        p, y = _miscalibrated()
        cat = cal.fit_calibrator(p, y, method="platt")
        q = cat.predict(p)
        before = cal.ece(p, y)
        after = cal.ece(q, y)
        assert after < before
        assert (q >= 0).all() and (q <= 1).all()

    def test_platt_monotone_and_params(self):
        p = np.linspace(0.1, 0.9, 9)
        y = (p > 0.5).astype(int) * 1.0
        cat = cal.fit_calibrator(p, y, method="platt")
        q = cat.predict(p)
        assert _is_monotone(q)
        d = cat.to_dict()
        assert d["method"] == "platt" and "a" in d and "b" in d

    def test_platt_edge_values(self):
        cat = cal.fit_calibrator(np.array([0.2, 0.2, 0.8, 0.8]),
                                 np.array([0, 0, 1, 1]), method="platt")
        assert cat.predict(np.array([0.0, 1.0]))[1] > 0.5


class TestIsotonic:
    def test_isotonic_monotone_nondecreasing(self):
        rng = np.random.default_rng(2)
        p = rng.uniform(0, 1, 500)
        y = (rng.uniform(0, 1, 500) < p).astype(int)
        cat = cal.fit_calibrator(p, y, method="isotonic")
        q = cat.predict(p)
        assert _sorted_monotone(p, q)
        assert (q >= 0).all() and (q <= 1).all()

    def test_isotonic_exact_solution_fits_perfectly(self):
        # Structured data where the true calibration is a step function.
        p = np.concatenate([np.full(100, 0.2), np.full(100, 0.8)])
        y = np.concatenate([np.zeros(100), np.ones(100)])
        cat = cal.fit_calibrator(p, y, method="isotonic")
        q = cat.predict(p)
        assert np.allclose(q[:100], 0.0, atol=1e-3)
        assert np.allclose(q[100:], 1.0, atol=1e-3)
        assert cal.ece(q, y) < 1e-3

    def test_isotonic_handles_ties(self):
        p = np.array([0.2, 0.2, 0.2, 0.8, 0.8])
        y = np.array([0, 0, 1, 0, 1])
        cat = cal.fit_calibrator(p, y, method="isotonic")
        q = cat.predict(p)
        assert _sorted_monotone(p, q)
        assert q[0] == q[1] == q[2]


class TestFitCalibrator:
    def test_deterministic(self):
        p, y = _miscalibrated(seed=9)
        a = cal.fit_calibrator(p, y, method="isotonic").predict(p)
        b = cal.fit_calibrator(p, y, method="isotonic").predict(p)
        assert np.array_equal(a, b)

    def test_nan_input_is_passthrough(self):
        cat = cal.fit_calibrator(np.array([0.2, 0.8]), np.array([0, 1]),
                                 method="platt")
        q = cat.predict(np.array([np.nan, 0.5]))
        assert np.isnan(q[0]) and ~np.isnan(q[1])

    def test_unknown_method(self):
        with pytest.raises(ValueError):
            cal.fit_calibrator(np.array([0.5, 0.6]), np.array([0, 1]), method="nope")

    def test_sum_up_down_is_one(self):
        p, y = _miscalibrated()
        cat = cal.fit_calibrator(p, y, method="isotonic")
        q = cat.predict(p)
        assert np.allclose(q + (1 - q), 1.0)


def _is_monotone(a: np.ndarray) -> bool:
    a = np.asarray(a)
    return bool((np.diff(a) >= -1e-12).all())


def _sorted_monotone(p: np.ndarray, q: np.ndarray) -> bool:
    order = np.argsort(np.asarray(p), kind="mergesort")
    return _is_monotone(np.asarray(q)[order])