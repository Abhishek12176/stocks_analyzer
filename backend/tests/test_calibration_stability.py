"""Calibration-stability tests (audit follow-up, changes A-F, 07-Sep-2026).

Covers the measurement-hardening changes approved after the TCS 88-vs-67-69
calibration instability audit:

A. empirical Brier standard error + the 1SE multiplier as a swept parameter
   (`cal.brier_standard_error`, `pipeline._select_calibrator(one_se_mult=...)`);
B. raw-probability support/collapse diagnostic (`cal.raw_prob_support`) — the
   collapse is reported, never smoothed over;
C. the old 60% block-range cut is removed — selection uses distinct-levels +
   support + 1SE, and the selection trace is exposed;
D. isotonic degeneracy cutoff (distinct output levels) as a swept parameter
   (`pipeline._select_calibrator(n_level_cutoff=...)`, sweep {2,3,4});
E. selection distinguishes *mapping* stabilization (Platt fallback on
   degeneracy) from *fixing* the underlying low-support/collapse problem — no
   smoother is used to hide collapse, so the support diagnostic stays visible;
F. audit semantics: selection is a pure function of calibration-fit rows (the
   holdout can never be part of the input signature), and diagnostics document
   the metric-level replay scope.

All tests are structural/deterministic on synthetic data; nothing asserts
trading alpha.
"""

import numpy as np
import pytest

from app.ml import calibration as cal
from app.ml import pipeline


def _step_two_level(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Perfect 2-level map: learned isotonic has exactly 2 distinct levels."""
    p = np.concatenate([np.full(n, 0.2), np.full(n, 0.8)])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    return p, y


def _collapsed_probs(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Raw probs clustered within +/-0.06 of 0.50 (near-chance raw signal)."""
    rng = np.random.default_rng(seed)
    p = np.clip(rng.normal(0.5, 0.02, n), 0.0, 1.0)
    y = rng.integers(0, 2, n).astype(float)
    return p, y


def _wide_probs(n: int = 400, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Raw probs spread over [0.05, 0.95] (healthy support)."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, n)
    y = rng.integers(0, 2, n).astype(float)
    return p, y


class TestBrierStandardError:
    """Change A: SE is empirical (per-row loss variance), not a fixed constant."""

    def test_matches_large_sample_formula(self):
        # losses: (0.25-1)^2 = 0.5625 and (0.75-1)^2 = 0.0625
        # sample std (ddof=1) / sqrt(2) = 0.3535534 / 1.41421 = 0.25
        se = cal.brier_standard_error(
            np.array([0.25, 0.75]), np.array([1.0, 1.0])
        )
        assert se == pytest.approx(0.25, abs=1e-6)

    def test_zero_when_per_row_loss_constant(self):
        # both rows have loss 0.36 -> sample std 0 -> SE 0
        se = cal.brier_standard_error(
            np.array([0.4, 0.6]), np.array([1.0, 0.0])
        )
        assert se == pytest.approx(0.0, abs=1e-12)

    def test_nan_passthrough(self):
        assert np.isnan(cal.brier_standard_error(np.array([]), np.array([])))


class TestRawProbSupport:
    """Change B: collapse diagnostic on raw (fit-row) probabilities."""

    def test_collapsed_probs_flagged(self):
        support = cal.raw_prob_support(_collapsed_probs()[0])
        assert support["pct_near_0p05"] > 0.9
        assert support["std"] < 0.1

    def test_wide_probs_not_collapsed(self):
        support = cal.raw_prob_support(_wide_probs()[0])
        assert support["pct_near_0p05"] < 0.3
        assert support["std"] > 0.1

    def test_empty_input_reports_nan(self):
        support = cal.raw_prob_support(np.array([]))
        assert support["n"] == 0
        assert np.isnan(support["pct_near_0p05"])

    def test_reports_several_bands(self):
        support = cal.raw_prob_support(_wide_probs()[0])
        assert set(support) == {
            "n", "pct_near_0p02", "pct_near_0p05", "pct_near_0p10",
            "min", "max", "std",
        }


class TestIsotonicDegeneracyDiagnostics:
    """Change D: distinct-level count + per-block support are exposed."""

    def test_two_level_map_reports_two_levels(self):
        p, y = _step_two_level()
        params = cal.isotonic_fit(p, y)
        assert cal.isotonic_n_distinct_levels(params) == 2
        assert sum(cal.isotonic_block_sizes(params)) == len(p)

    def test_to_dict_keeps_new_keys(self):
        params = cal.isotonic_fit(*_step_two_level())
        d = cal.fit_calibrator(*_step_two_level(), method="isotonic").to_dict()
        assert d["method"] == "isotonic"
        assert "block_sizes" in d and "n_distinct_levels" in d


class TestSelectCalibrator:
    """Selection is a pure function of fit rows + parameters (changes A, C, D)."""

    def test_insufficient_rows_uncalibrated(self):
        p, y = _step_two_level(n=40)  # 80 rows < n_min=100
        calibrator, method, diag = pipeline._select_calibrator(p, y)
        assert method == "uncalibrated"
        assert calibrator is None
        assert "insufficient fit rows" in diag["reason"]

    def test_fit_rows_only_signature(self):
        # The signature has no holdout parameter by construction (change F):
        # the newest-20% holdout can never be used for selection.
        import inspect
        sig = inspect.signature(pipeline._select_calibrator)
        assert "hold" not in sig.parameters

    def test_degenerate_isotonic_falls_back_to_platt(self):
        # A 2-level map is a near-constant walk at cutoff=3 -> degeneracy.
        p, y = _step_two_level()
        calib, method, diag = pipeline._select_calibrator(
            p, y, n_level_cutoff=3
        )
        assert method == "platt"
        assert calib is not None
        assert diag["degenerated_to_platt"] is True
        assert diag["reason"].startswith("degenerate isotonic")

    def test_level_cutoff_sweep_includes_cutoff_2(self):
        # At cutoff=2 the 2-level map is not degenerate. Nested temporal
        # selection, rather than in-sample fit loss, now determines the winner.
        p, y = _step_two_level()
        calib, method, diag = pipeline._select_calibrator(
            p, y, n_level_cutoff=2, one_se_mult=0.0
        )
        assert method in {"isotonic", "platt"}
        assert diag["n_isotonic_distinct_levels"] == 2
        assert diag["selection_train_rows"] > 0
        assert diag["selection_rows"] > 0

    def test_level_cutoff_sweep_234_monotone(self):
        p, y = _step_two_level()
        methods = [
            pipeline._select_calibrator(p, y, n_level_cutoff=c, one_se_mult=0.0)[1]
            for c in (2, 3, 4)
        ]
        assert methods[0] in {"isotonic", "platt"}
        assert methods[1] == "platt"
        assert methods[2] == "platt"

    def test_one_se_multiplier_sweep_parameterized(self):
        # Tight tie: isotonic and platt are close on noise-only data. A huge
        # multiplier can only ever push selection away from isotonic -> platt.
        p, y = _wide_probs(n=600, seed=7)
        for mult in (0.5, 1.0, 1.5):
            _, method, diag = pipeline._select_calibrator(
                p, y, one_se_mult=mult
            )
            assert method in {"platt", "isotonic"}
            assert "reason" in diag
        # Decisive directional behavior across the parameter range.
        assert pipeline._select_calibrator(p, y, one_se_mult=1e6)[1] == "platt"

    def test_deterministic_same_inputs(self):
        p, y = _wide_probs()
        a = pipeline._select_calibrator(p, y)
        b = pipeline._select_calibrator(p, y)
        assert a[1] == b[1]
        assert a[2] == b[2]

    def test_collapse_reported_not_smoothed(self):
        # Change B+E: near-chance raw probs are *reported* as collapsed; the
        # selection never claims smoothing fixed the underlying problem.
        p, y = _collapsed_probs()
        _, method, diag = pipeline._select_calibrator(p, y)
        support = diag["support"]
        assert support["pct_near_0p05"] > 0.9
        assert "reason" in diag
        assert method in {"platt", "uncalibrated", "isotonic"}
        # No smoothing anywhere in the selection vocabulary.
        blob = " ".join(str(v) for v in (method, diag.get("reason", "")))
        assert "smooth" not in blob.lower()


class TestSensitivityReport:
    """Change E/F: the holdout-crossing + collapse numbers are surfaced."""

    def test_crossing_count_correct(self):
        hold_p = np.array([0.48, 0.62, 0.55, 0.40])
        hold_q = np.array([0.52, 0.58, 0.55, 0.45])
        report = pipeline._calibration_sensitivity_report(
            {"reason": "tie -> platt", "support": cal.raw_prob_support(hold_p)},
            hold_p, hold_q,
        )
        assert report["n_holdout_with_probs"] == 4
        assert report["n_crossing_0p50"] == 1
        assert report["selection_reason"] == "tie -> platt"

    def test_empty_holdout_safe(self):
        report = pipeline._calibration_sensitivity_report(
            {"reason": "none", "support": None}, np.array([]), np.array([])
        )
        assert report["n_holdout_with_probs"] == 0
        assert report["n_crossing_0p50"] == 0
        assert report["support"] is None

    def test_report_has_support_and_method_trace(self):
        report = pipeline._calibration_sensitivity_report(
            {"reason": "x", "n_isotonic_distinct_levels": 2,
             "support": cal.raw_prob_support(_wide_probs()[0])},
            np.array([0.6]), np.array([0.61]),
        )
        assert "support" in report
        assert report["n_isotonic_distinct_levels"] == 2
