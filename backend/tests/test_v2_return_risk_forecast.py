import json

import numpy as np
import pandas as pd
import pytest

from app.ml import v2_return_risk_forecast as rrf


def _make_synthetic_panel(
    n_dates: int = 500,
    n_symbols: int = 3,
    seed: int = 17,
) -> pd.DataFrame:
    """Build a small synthetic panel with required columns."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_dates, freq="B")
    records = []
    for sym_idx in range(n_symbols):
        symbol = f"SYM{sym_idx}"
        close = 100.0
        for d in dates:
            ret = rng.normal(0.0005, 0.02)
            close *= 1 + ret
            row = {
                "symbol": symbol,
                "date": d,
                "Close": close,
                "feature_a": rng.normal(),
                "feature_b": rng.normal(),
                "target_ret_20d": rng.normal(0.0, 0.05),
                "target_up_20d": int(rng.random() > 0.5),
            }
            records.append(row)
    return pd.DataFrame(records)


@pytest.fixture
def synthetic_panel():
    return _make_synthetic_panel()


class TestModuleConstants:
    def test_version_strings(self):
        assert isinstance(rrf.MODULE_VERSION, str)
        assert isinstance(rrf.SCHEMA_VERSION, str)

    def test_classifications(self):
        assert "PROMISING" in rrf.CLASSIFICATIONS
        assert "REJECTED" in rrf.CLASSIFICATIONS
        assert "EVIDENCE STILL INSUFFICIENT" in rrf.CLASSIFICATIONS

    def test_default_seeds(self):
        assert rrf.DEFAULT_SEEDS == (17, 31, 53)


class TestRegressor:
    def test_ridge_builds(self):
        reg = rrf._build_regressor("ridge")
        assert hasattr(reg, "fit")

    def test_random_forest_builds(self):
        reg = rrf._build_regressor("random_forest")
        assert hasattr(reg, "fit")

    def test_xgboost_builds(self):
        reg = rrf._build_regressor("xgboost")
        assert hasattr(reg, "fit")

    def test_unknown_regressor_raises(self):
        with pytest.raises(ValueError, match="unknown regressor"):
            rrf._build_regressor("nope")

    def test_fit_regressor_ridge(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 5))
        y = rng.normal(size=100)
        preds = rrf._fit_regressor("ridge", X[:80], y[:80], X[80:], seed=17)
        assert len(preds) == 20
        assert all(np.isfinite(preds))

    def test_fit_regressor_insufficient_data(self):
        X = np.zeros((5, 3))
        y = np.full(5, np.nan)
        preds = rrf._fit_regressor("ridge", X, y, X, seed=17)
        assert all(np.isnan(preds))


class TestDirection:
    def test_fit_direction_logistic(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 5))
        y = (rng.random(100) > 0.5).astype(float)
        p_up = rrf._fit_direction("logistic", X[:80], y[:80], X[80:], seed=17)
        assert len(p_up) == 20
        assert all(0.0 <= v <= 1.0 for v in p_up)


class TestRiskEstimation:
    def test_residual_uncertainty(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        result = rrf._residual_uncertainty(y_true, y_pred)
        assert result["cond_vol"] is not None
        assert result["residual_mad"] is not None
        assert result["rmse"] is not None
        assert result["cond_vol"] > 0

    def test_residual_uncertainty_insufficient(self):
        result = rrf._residual_uncertainty(np.array([1.0]), np.array([1.1]))
        assert result["cond_vol"] is None

    def test_quantile_intervals(self):
        preds = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        result = rrf._quantile_intervals(preds, 0.02)
        assert "q_lower" in result
        assert "q_upper" in result
        assert result["q_lower"] < result["q_upper"]

    def test_conditional_volatility(self):
        residuals = np.array([0.01, -0.02, 0.015, -0.01, 0.005])
        vol = rrf._conditional_volatility_from_residuals(residuals)
        assert vol > 0
        assert np.isfinite(vol)

    def test_conditional_volatility_insufficient(self):
        vol = rrf._conditional_volatility_from_residuals(np.array([np.nan]))
        assert np.isnan(vol)


class TestEntropyDiagnostic:
    def test_insufficient_data(self):
        result = rrf._entropy_interaction_diagnostic([])
        assert result["interpretation"] == "insufficient_data"
        assert result["n"] == 0

    def test_with_data(self):
        rng = np.random.default_rng(42)
        p_vals = rng.uniform(0.05, 0.95, 100)
        records = [
            {"p_up": float(p), "ret_error": float(rng.normal())}
            for p in p_vals
        ]
        result = rrf._entropy_interaction_diagnostic(records)
        assert result["n"] == 100
        assert result["corr"] is not None
        assert result["interpretation"] in (
            "entropy_reduces_regret", "moderate_signal", "no_signal", "zero_variance",
        )

    def test_zero_entropy_variance(self):
        records = [{"p_up": 0.5, "ret_error": float(i)} for i in range(50)]
        result = rrf._entropy_interaction_diagnostic(records)
        assert result["interpretation"] == "zero_variance"

    def test_p_up_at_extremes_skipped(self):
        records = [
            {"p_up": 0.0, "ret_error": 1.0},
            {"p_up": 1.0, "ret_error": 1.0},
            {"p_up": 0.5, "ret_error": 1.0},
        ]
        result = rrf._entropy_interaction_diagnostic(records)
        assert result["n"] == 1


class TestClassification:
    def test_empty_holdout_gives_insufficient(self):
        result = rrf._classify_result(
            holdout_metrics={}, null_rmse=None, symbol_metrics={},
            fold_metrics=[], ent_diag={"interpretation": "no_signal"},
        )
        assert result == "EVIDENCE STILL INSUFFICIENT"

    def test_null_rmse_comparable_gives_rejected(self):
        holdout = {"config_a_ridge": {"rmse": 0.05}}
        result = rrf._classify_result(
            holdout_metrics=holdout, null_rmse=0.048, symbol_metrics={},
            fold_metrics=[], ent_diag={"interpretation": "no_signal"},
        )
        assert result == "REJECTED"

    def test_no_rmse_gives_rejected(self):
        holdout = {"config_a_ridge": {"rmse": None}}
        result = rrf._classify_result(
            holdout_metrics=holdout, null_rmse=None, symbol_metrics={},
            fold_metrics=[], ent_diag={"interpretation": "no_signal"},
        )
        assert result == "REJECTED"

    def test_good_symbols_promising(self):
        holdout = {"config_a_ridge": {"rmse": 0.02}}
        symbols = {f"SYM{i}": {"rmse": 0.03} for i in range(10)}
        folds = [{"fold": i, "rmse": 0.02} for i in range(4)]
        ent = {"interpretation": "moderate_signal"}
        result = rrf._classify_result(
            holdout_metrics=holdout, null_rmse=0.1,
            symbol_metrics=symbols, fold_metrics=folds, ent_diag=ent,
        )
        assert result == "PROMISING"

    def test_few_good_symbols_insufficient(self):
        holdout = {"config_a_ridge": {"rmse": 0.03}}
        symbols = {f"SYM{i}": {"rmse": 0.1} for i in range(10)}
        symbols["SYM0"] = {"rmse": 0.03}
        folds = [{"fold": i, "rmse": 0.02} for i in range(4)]
        ent = {"interpretation": "no_signal"}
        result = rrf._classify_result(
            holdout_metrics=holdout, null_rmse=0.1,
            symbol_metrics=symbols, fold_metrics=folds, ent_diag=ent,
        )
        assert result in ("EVIDENCE STILL INSUFFICIENT", "REJECTED")


class TestFiniteHelper:
    def test_numpy_integer(self):
        assert rrf._finite(np.int64(5)) == 5

    def test_numpy_float(self):
        assert rrf._finite(np.float64(3.14)) == pytest.approx(3.14)

    def test_nan_float(self):
        assert rrf._finite(float("nan")) is None

    def test_inf_float(self):
        assert rrf._finite(float("inf")) is None

    def test_dict(self):
        result = rrf._finite({"a": np.int64(1), "b": 2.0})
        assert result == {"a": 1, "b": 2.0}

    def test_list(self):
        result = rrf._finite([np.int64(1), float("nan")])
        assert result == [1, None]

    def test_string_passthrough(self):
        assert rrf._finite("hello") == "hello"


class TestSyntheticPanel:
    def test_panel_shape(self, synthetic_panel):
        assert len(synthetic_panel) > 0
        assert "symbol" in synthetic_panel.columns
        assert "date" in synthetic_panel.columns
        assert "target_ret_20d" in synthetic_panel.columns
        assert "target_up_20d" in synthetic_panel.columns

    def test_panel_symbols(self, synthetic_panel):
        symbols = synthetic_panel["symbol"].unique()
        assert len(symbols) == 3

    def test_panel_dates_ordered(self, synthetic_panel):
        dates = synthetic_panel.sort_values("date")["date"].values
        assert all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1))


class TestJsonDump:
    def test_round_trip(self, tmp_path):
        data = {"a": 1.0, "b": [np.float64(3.14), None], "c": {"d": np.int64(5)}}
        path = tmp_path / "test.json"
        rrf._json_dump(data, path)
        loaded = json.loads(path.read_text())
        assert loaded["a"] == 1.0
        assert loaded["b"][0] == pytest.approx(3.14)
        assert loaded["b"][1] is None
        assert loaded["c"]["d"] == 5
