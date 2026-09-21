"""Unit tests for global/local explanation helpers (Task 9)."""

import numpy as np
import pandas as pd
import pytest

from app.ml import explain as ex


def _frame(n=200):
    idx = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    y = pd.Series((x + 0.5 * noise > 0).astype(int), index=idx)
    df = pd.DataFrame({"x_signal": x, "x_noise": noise}, index=idx)
    return df, y


class TestAuc:
    def test_perfect(self):
        assert ex._auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0

    def test_reversed(self):
        assert ex._auc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])) == 0.0

    def test_ties_safe(self):
        auc = ex._auc(np.array([0, 0, 1, 1]), np.array([0.5, 0.5, 0.5, 0.5]))
        assert auc == pytest.approx(0.5)

    def test_single_class_nan(self):
        assert np.isnan(ex._auc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])))

    def test_nan_scores_dropped(self):
        auc = ex._auc(np.array([0, 0, 1, 1]), np.array([np.nan, 0.1, 0.8, 0.6]))
        assert auc == pytest.approx(1.0)


class TestPermutation:
    def test_signal_more_important_than_noise(self):
        X, y = _frame()
        pred = ex.fit_predictor("logistic", X, y, seed=1)
        res = ex.permutation_importance(X.tail(120), y.tail(120), pred,
                                        n_permutes=3, seed=2)
        assert res["importance"]["x_signal"] > 0.01
        assert res["importance"]["x_signal"] > res["importance"]["x_noise"]
        assert res["baseline"] > 0.5
        assert res["metric"] == "auc"

    def test_deterministic(self):
        X, y = _frame()
        pred = ex.fit_predictor("logistic", X, y, seed=1)
        a = ex.permutation_importance(X.tail(120), y.tail(120), pred,
                                      n_permutes=3, seed=5)
        b = ex.permutation_importance(X.tail(120), y.tail(120), pred,
                                      n_permutes=3, seed=5)
        assert a["importance"] == b["importance"]

    def test_logloss_metric(self):
        X, y = _frame()
        pred = ex.fit_predictor("logistic", X, y, seed=1)
        res = ex.permutation_importance(X.tail(120), y.tail(120), pred,
                                        n_permutes=2, seed=2, metric="logloss")
        assert res["baseline"] > 0
        assert "x_signal" in res["importance"]

    def test_unknown_metric(self):
        X, y = _frame()
        pred = ex.fit_predictor("logistic", X, y, seed=1)
        with pytest.raises(ValueError):
            ex.permutation_importance(X, y, pred, metric="bogus")


class TestExplainRow:
    def _prob_fn(self):
        def fn(row_df: pd.DataFrame) -> np.ndarray:
            a = row_df["up"].to_numpy()
            b = row_df["down"].to_numpy()
            p = 0.2 + 0.6 * a - 0.4 * b
            return np.clip(p, 0.0, 1.0)
        return fn

    def test_positive_and_negative_factors(self):
        row = pd.Series({"up": 0.5, "down": 0.5})
        res = ex.explain_row(self._prob_fn(), row, delta=0.2)
        assert res["positive"][0][0] == "up"
        assert res["negative"][0][0] == "down"
        assert res["baseline_prob"] == pytest.approx(0.3)

    def test_top_factors(self):
        row = pd.Series({"up": 0.5, "down": 0.5, "flat": 0.5})
        res = ex.explain_row(self._prob_fn(), row, delta=0.2)
        tops = ex.top_factors(res, "positive", k=2)
        assert tops and tops[0][0] == "up" and tops[0][1] > 0

    def test_impacts_sign(self):
        row = pd.Series({"up": 0.5, "down": 0.5})
        res = ex.explain_row(self._prob_fn(), row, delta=0.2)
        assert res["impacts"]["up"] > 0
        assert res["impacts"]["down"] < 0