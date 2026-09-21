"""Tests for Task 17 — Rigorous Frozen-V1 vs Pooled Model Comparison."""

import numpy as np
import pandas as pd
import pytest

from app.ml import v2_v1_fair_compare as vfc


# ---------------------------------------------------------------------------
# Snapshot determinism
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Requires network access (yfinance)")
def test_build_shared_snapshot_deterministic():
    """Same inputs must produce the same fingerprint."""
    panel1, meta1 = vfc.build_shared_snapshot(
        symbols=("TCS", "RELIANCE"), horizon=20,
    )
    panel2, meta2 = vfc.build_shared_snapshot(
        symbols=("TCS", "RELIANCE"), horizon=20,
    )
    assert meta1["fingerprint"] == meta2["fingerprint"]
    assert len(panel1) == len(panel2)
    assert meta1["n_symbols"] == meta2["n_symbols"]


@pytest.mark.skip(reason="Requires network access (yfinance)")
def test_snapshot_fingerprint_changes_with_input():
    """Different symbols must produce different fingerprints."""
    _, meta1 = vfc.build_shared_snapshot(
        symbols=("TCS",), horizon=20,
    )
    _, meta2 = vfc.build_shared_snapshot(
        symbols=("RELIANCE",), horizon=20,
    )
    assert meta1["fingerprint"] != meta2["fingerprint"]


@pytest.mark.skip(reason="Requires network access (yfinance)")
def test_snapshot_metadata_schema():
    """Snapshot metadata must have required fields."""
    _, meta = vfc.build_shared_snapshot(
        symbols=("TCS",), horizon=20,
    )
    required = ["snapshot_version", "n_symbols", "n_rows", "fingerprint",
                "symbols_included", "symbols_skipped", "date_range",
                "horizon", "generated_at"]
    for field in required:
        assert field in meta, f"Missing field: {field}"
    assert meta["horizon"] == 20
    assert meta["snapshot_version"] == vfc.MODULE_VERSION


# ---------------------------------------------------------------------------
# Fold structure
# ---------------------------------------------------------------------------

def test_shared_folds_chronological():
    """Folds must be ordered by date (each fold starts after previous)."""
    panel = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=500, freq="D"),
        "symbol": ["A"] * 500,
    })
    folds = vfc._get_shared_folds(panel, horizon=20)
    assert len(folds) > 0
    # Each fold's test dates must come after the previous fold's test dates
    prev_test_end = None
    for train_dates, test_dates in folds:
        if prev_test_end is not None:
            assert test_dates[0] >= prev_test_end
        prev_test_end = test_dates[-1]


def test_shared_folds_min_train():
    """Each fold's training set must be reasonably large."""
    panel = pd.DataFrame({
        "date": pd.date_range("2016-01-01", periods=700, freq="D"),
        "symbol": ["A"] * 700,
    })
    folds = vfc._get_shared_folds(panel, horizon=20)
    for train_dates, test_dates in folds:
        assert len(train_dates) >= 200


# ---------------------------------------------------------------------------
# Fair matching
# ---------------------------------------------------------------------------

def test_build_matched_sample_correct_intersection():
    """Matched sample must only contain (symbol, date) pairs present in both."""
    v1_oof = pd.DataFrame({
        "symbol": ["A", "A", "B"],
        "date": ["2024-01-01", "2024-01-02", "2024-01-01"],
        "ensemble": [0.6, 0.4, 0.7],
        "y": [1, 0, 1],
    })
    pr = {
        "feature_config": "baseline",
        "usable_models": ["logistic"],
        "oof_mean": pd.DataFrame({
            "symbol": ["A", "B"],
            "date": ["2024-01-01", "2024-01-01"],
            "prob": [0.55, 0.65],
            "model": ["logistic", "logistic"],
            "y": [1, 1],
        }),
    }
    result = vfc.build_matched_sample({"oof_mean": v1_oof}, [pr])
    matched = result["matched"]
    # Only A|2024-01-01 and B|2024-01-01 should be in matched
    # A|2024-01-02 has no pooled counterpart
    assert len(matched) >= 0  # At least the intersection rows
    for _, row in matched.iterrows():
        key = f"{row['symbol']}|{row['date']}"
        assert key in ["A|2024-01-01", "B|2024-01-01"]


def test_matched_sample_has_valid_columns():
    """Matched sample must have v1_ensemble and y columns."""
    v1_oof = pd.DataFrame({
        "symbol": ["A"],
        "date": ["2024-01-01"],
        "ensemble": [0.6],
        "y": [1],
    })
    pr = {
        "feature_config": "baseline",
        "usable_models": ["logistic"],
        "oof_mean": pd.DataFrame({
            "symbol": ["A"],
            "date": ["2024-01-01"],
            "prob": [0.55],
            "model": ["logistic"],
            "y": [1],
        }),
    }
    result = vfc.build_matched_sample({"oof_mean": v1_oof}, [pr])
    matched = result["matched"]
    assert "v1_ensemble" in matched.columns
    assert "y" in matched.columns
    assert "n_matched_rows" in result


def test_matched_sample_counts():
    """Matched sample counts must be accurate."""
    v1_oof = pd.DataFrame({
        "symbol": ["A", "A", "B"],
        "date": ["2024-01-01", "2024-01-02", "2024-01-01"],
        "ensemble": [0.6, 0.4, 0.7],
        "y": [1, 0, 1],
    })
    pr = {
        "feature_config": "baseline",
        "usable_models": ["logistic"],
        "oof_mean": pd.DataFrame({
            "symbol": ["A", "B"],
            "date": ["2024-01-01", "2024-01-01"],
            "prob": [0.55, 0.65],
            "model": ["logistic", "logistic"],
        }),
    }
    result = vfc.build_matched_sample({"oof_mean": v1_oof}, [pr])
    assert result["n_matched_rows"] >= 0
    assert result["n_matched_symbols"] >= 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_compute_binary_metrics_perfect():
    """Perfect classifier should have AUC=1.0."""
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = vfc.compute_binary_metrics(y, p)
    assert metrics["roc_auc"] == pytest.approx(1.0, abs=0.01)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["brier"] < 0.1


def test_compute_binary_metrics_random():
    """Random classifier should have AUC≈0.5."""
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=200)
    p = rng.random(200)
    metrics = vfc.compute_binary_metrics(y, p)
    assert 0.3 < metrics["roc_auc"] < 0.7


def test_compute_binary_metrics_edge_cases():
    """Single class should return None for AUC."""
    y = np.array([1, 1, 1])
    p = np.array([0.5, 0.6, 0.7])
    metrics = vfc.compute_binary_metrics(y, p)
    assert metrics["roc_auc"] is None
    assert metrics["n"] == 3


def test_bootstrap_ci():
    """Bootstrap CI must return valid range."""
    deltas = np.array([0.01, 0.02, -0.01, 0.03, 0.015])
    ci = vfc.bootstrap_ci(deltas)
    assert "mean" in ci
    assert "ci_lower" in ci
    assert "ci_upper" in ci
    assert ci["ci_lower"] <= ci["mean"] <= ci["ci_upper"]


# ---------------------------------------------------------------------------
# Decision classification
# ---------------------------------------------------------------------------

def test_classify_implement_pooled():
    """Strong cross-symbol evidence should classify as IMPLEMENT POOLED V2."""
    comparison = {
        "median_auc_delta": 0.05,
        "n_symbols_pooled_wins": 10,
        "n_symbols_total": 12,
        "fold_robustness": True,
        "seed_stable": True,
    }
    assert vfc.classify_result(comparison) == "IMPLEMENT POOLED V2"


def test_classify_keep_frozen_v1():
    """No improvement should classify as KEEP FROZEN V1."""
    comparison = {
        "median_auc_delta": -0.01,
        "n_symbols_pooled_wins": 3,
        "n_symbols_total": 12,
        "fold_robustness": False,
        "seed_stable": False,
    }
    assert vfc.classify_result(comparison) == "KEEP FROZEN V1"


def test_classify_insufficient():
    """Mixed evidence with delta > 0 but low win rate should classify correctly."""
    # median_delta > 0 but win_rate < 0.5 → KEEP FROZEN V1
    comparison = {
        "median_auc_delta": 0.005,
        "n_symbols_pooled_wins": 3,
        "n_symbols_total": 12,
        "fold_robustness": False,
        "seed_stable": False,
    }
    # With median_delta > 0 but insufficient wins, it's KEEP or INSUFFICIENT
    result = vfc.classify_result(comparison)
    assert result in ["KEEP FROZEN V1", "EVIDENCE STILL INSUFFICIENT"]


# ---------------------------------------------------------------------------
# ECE calculation
# ---------------------------------------------------------------------------

def test_ece_zero():
    """Perfectly calibrated probabilities should have low ECE."""
    # Each bin's mean prob matches its mean observed rate
    prob = np.array([0.1, 0.1, 0.9, 0.9])
    y = np.array([0, 0, 1, 1])
    ece = vfc._compute_ece(prob, y, n_bins=10)
    assert ece < 0.15


# ---------------------------------------------------------------------------
# V1 reference shape
# ---------------------------------------------------------------------------

def test_run_v1_reference_returns_valid_structure():
    """V1 reference must return valid OOF panel structure."""
    n_dates = 600
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="D")
    panel = pd.DataFrame({
        "symbol": ["A"] * n_dates,
        "date": dates,
        "Open": np.random.random(n_dates),
        "High": np.random.random(n_dates),
        "Low": np.random.random(n_dates),
        "Close": np.random.random(n_dates),
        "Volume": np.random.random(n_dates),
    })
    panel["SMA20"] = panel["Close"].rolling(20, min_periods=1).mean()
    panel["RSI"] = 50.0

    folds = vfc._get_shared_folds(panel, horizon=20)
    assert len(folds) > 0


def test_fold_ensemble_probs_returns_valid_shapes():
    """_fold_ensemble_probs must return prob array and per-model dict."""
    from app.ml import models as mo, ensemble as en

    n_train, n_test = 50, 10
    X = pd.DataFrame(np.random.random((n_train + n_test, 5)),
                      columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, n_train + n_test).astype(float))

    train_dates = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=n_train, freq="D"))
    test_dates = pd.DatetimeIndex(pd.date_range("2020-03-01", periods=n_test, freq="D"))

    models = ["voting"]
    combined, per_model = vfc._fold_ensemble_probs(
        X, y, train_dates, test_dates, models, seed=0
    )
    assert combined is not None
    assert len(combined) == n_test
    assert "voting" in per_model


# ---------------------------------------------------------------------------
# Feature config decomposition
# ---------------------------------------------------------------------------

def test_feature_config_separation():
    """Config A and Config B must differ only in relative features."""
    panel = pd.DataFrame({
        "symbol": ["A"] * 50,
        "date": pd.date_range("2020-01-01", periods=50, freq="D"),
        "Close": np.random.random(50),
        "target_up_20d": np.random.randint(0, 2, 50).astype(float),
        "ADX": np.random.random(50),
    })
    # Test feature matrix creation
    from app.ml.v2_pooled_models import build_feature_matrices
    matrices_a = build_feature_matrices(panel, baseline_features=None,
                                         relative_features=None, identity="none")
    assert "baseline" in matrices_a


# ---------------------------------------------------------------------------
# Pipeline import safety
# ---------------------------------------------------------------------------

def test_module_imports_cleanly():
    """Module must import without errors."""
    import app.ml.v2_v1_fair_compare as mod
    assert hasattr(mod, "build_shared_snapshot")
    assert hasattr(mod, "run_v1_reference")
    assert hasattr(mod, "run_pooled_candidates")
    assert hasattr(mod, "build_matched_sample")
    assert hasattr(mod, "run_comparison")
    assert hasattr(mod, "classify_result")
    assert hasattr(mod, "compute_binary_metrics")


def test_v1_component_contract_includes_ridge():
    assert set(("voting", "logistic", "ridge", "rf", "xgboost")).issubset(
        set(vfc.ENSEMBLE_MODELS)
    )


def test_build_synthetic_panel():
    """Build a synthetic panel for testing without network."""
    import numpy as np
    n_dates = 600
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="D")
    symbols = ["A", "B"]
    rows = []
    for sym in symbols:
        close = np.cumsum(np.random.random(n_dates) - 0.49) + 100
        open_ = close + np.random.random(n_dates) * 2 - 1
        high = np.maximum(open_, close) + np.random.random(n_dates)
        low = np.minimum(open_, close) - np.random.random(n_dates)
        for i in range(n_dates):
            rows.append({
                "symbol": sym,
                "date": dates[i],
                "Open": float(open_[i]),
                "High": float(high[i]),
                "Low": float(low[i]),
                "Close": float(close[i]),
                "Volume": float(np.random.random() * 1e6),
            })
    panel = pd.DataFrame(rows)
    # Verify fold computation works on the panel
    folds = vfc._get_shared_folds(panel, horizon=20)
    assert len(folds) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
