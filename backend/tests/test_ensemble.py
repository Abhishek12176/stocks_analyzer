"""Unit tests for the ensemble merging of per-model P(up) (Task 9)."""

import numpy as np
import pandas as pd
import pytest

from app.ml import ensemble as en
from app.ml import models as mo


class TestCombineProbs:
    def test_equal_average(self):
        out = en.combine_probs([np.full(4, 0.2), np.full(4, 0.6)])
        assert np.allclose(out, 0.4)

    def test_weighted_average(self):
        out = en.combine_probs([np.full(4, 0.0), np.full(4, 1.0)], weights=[0.75, 0.25])
        assert np.allclose(out, 0.25)

    def test_dict_weights(self):
        out = en.combine_probs([np.array([0.0, 1.0])], weights={"a": 2.0})
        assert np.allclose(out, [0.0, 1.0])

    def test_prob_complement_sums_to_one(self):
        p = en.combine_probs([np.array([0.3, 0.7, 0.5]), np.array([0.5, 0.5, 0.5])])
        assert np.allclose(p + (1 - p), 1.0)

    def test_single_nan_row_renormalizes(self):
        a = np.array([0.2, np.nan])
        b = np.array([0.6, 0.8])
        out = en.combine_probs([a, b])
        assert np.isclose(out[0], 0.4)
        assert np.isclose(out[1], 0.8)  # only model b produced a value

    def test_all_nan_stays_nan(self):
        out = en.combine_probs([np.array([np.nan, np.nan]), np.array([np.nan, np.nan])])
        assert np.isnan(out).all()

    def test_output_clipped(self):
        out = en.combine_probs([np.array([1.5, -0.5]), np.array([1.5, -0.5])])
        assert (out >= 0).all() and (out <= 1).all()

    def test_zero_or_missing_weights_rejected(self):
        with pytest.raises(ValueError):
            en.combine_probs([np.array([0.5]), np.array([0.5])], weights=[1.0, 0.0])
        with pytest.raises(ValueError):
            en.combine_probs([np.array([0.5])], weights=[1.0, 2.0])


class TestFitPredictEnsemble:
    def _data(self, n=200):
        idx = pd.bdate_range("2022-01-03", periods=n)
        rng = np.random.default_rng(1)
        X = pd.DataFrame({"x": rng.normal(0, 1, n)}, index=idx)
        y = pd.Series((X["x"] > 0).astype(int), index=idx)
        Xte = pd.DataFrame({"x": rng.normal(0, 1, 40)})
        return X, y, Xte

    def test_ensemble_returns_probs_and_complements(self):
        X, y, Xte = self._data()
        res = en.fit_predict_ensemble(X, y, Xte,
                                      models=("logistic", "rf", "xgboost"))
        assert set(res["model_probs"]) <= set(map(str, res["models"]))
        assert (res["prob"] >= 0).all() and (res["prob"] <= 1).all()
        p = res["prob"]
        assert np.allclose(p + (1 - p), 1.0)
        assert res["version"]

    def test_weights_dict_honored(self):
        X, y, Xte = self._data()
        res = en.fit_predict_ensemble(X, y, Xte,
                                      models=("logistic", "rf"),
                                      weights={"logistic": 0.9, "rf": 0.1})
        assert res["weights"][0] == 0.9
        lo_pred = mo.fit_and_predict("logistic", X, y, Xte)
        manual = en.combine_probs([res["model_probs"]["logistic"],
                                   res["model_probs"]["rf"]],
                                  weights=[0.9, 0.1])
        assert np.allclose(res["prob"], manual)

    def test_unavailable_models_are_skipped(self):
        X, y, Xte = self._data()
        res = en.fit_predict_ensemble(X, y, Xte, models=("madeup", "logistic"))
        assert res["models"] == ["logistic"]

    def test_unknown_model_dict_rejected(self):
        X, y, Xte = self._data()
        with pytest.raises(ValueError):
            en.fit_predict_ensemble(X, y, Xte, models=("logistic",),
                                    weights={"madeup": 1.0})

    def test_ensemble_voting_included(self):
        X, y, Xte = self._data()
        res = en.fit_predict_ensemble(X, y, Xte, models=("voting", "logistic"))
        assert "voting" in res["models"]
        assert (res["prob"] >= 0).all() and (res["prob"] <= 1).all()