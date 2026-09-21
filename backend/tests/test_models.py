"""Unit tests for the task-8 model zoo + voting baseline.

The voting baseline is tested against hand-computed buckets (Trend / RSI /
MACD / Sentiment). sklearn / xgboost-backed tests skip cleanly when those
libraries are missing.
"""

import numpy as np
import pandas as pd
import pytest

from app.ml import models as mo


def test_resolve_column_alias_order():
    cols = ["x", "rsi_14", "rsi"]
    assert mo.resolve_column(cols, mo.VOTING_ALIASES["rsi"]) == "rsi"
    assert mo.resolve_column(["rsi_14"], mo.VOTING_ALIASES["rsi"]) == "rsi_14"
    assert mo.resolve_column(["other"], mo.VOTING_ALIASES["rsi"]) is None


def test_voting_bull_row():
    df = pd.DataFrame([{
        "sma20": 110.0, "sma50": 100.0, "rsi": 55.0,
        "macd": 1.0, "signal": 0.5, "sentiment": 0.5,
    }])
    assert mo.voting_direction_from_frame(df).iloc[0] == 1.0


def test_voting_bear_row():
    df = pd.DataFrame([{
        "sma20": 100.0, "sma50": 110.0, "rsi": 75.0,
        "macd": 0.5, "signal": 1.0, "sentiment": -0.5,
    }])
    assert mo.voting_direction_from_frame(df).iloc[0] == -1.0


def test_voting_oversold_rsi_vote_solo():
    df = pd.DataFrame([{"rsi": 20.0}])          # only RSI present -> +1
    assert mo.voting_direction_from_frame(df).iloc[0] == 1.0


def test_voting_overbought_rsi_vote_solo():
    df = pd.DataFrame([{"rsi": 85.0}])
    assert mo.voting_direction_from_frame(df).iloc[0] == -1.0


def test_voting_missing_columns_abstain():
    df = pd.DataFrame([{"unrelated": 3.0}])
    assert mo.voting_direction_from_frame(df).iloc[0] == 0.0


def test_voting_neutral_scores_zero():
    df = pd.DataFrame([{"sma20": 110.0, "sma50": 100.0, "rsi": 45.0,
                        "macd": 0.5, "signal": 1.0, "sentiment": 0.0}])
    # trend +1 cancels MACD -1 (and RSI/sentiment abstain) -> HOLD
    assert mo.voting_direction_from_frame(df).iloc[0] == 0.0


def test_model_names_shape():
    names = mo.model_names()
    assert names[0] == "voting"
    assert len(names) == len(set(names))


def _separable(n: int = 240, seed: int = 0) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    X = pd.DataFrame({"x0": rng.normal(0, 1, n), "x1": rng.normal(0, 1, n)}, index=idx)
    y = pd.Series((X["x0"] > 0).astype(int), index=idx)
    Xte = pd.DataFrame({"x0": rng.normal(0, 1, 40), "x1": rng.normal(0, 1, 40)})
    return X, y, Xte


@pytest.mark.skipif("logistic" not in mo.model_names(), reason="sklearn missing")
def test_logistic_separable():
    X, y, Xte = _separable()
    prob = mo.fit_and_predict("logistic", X, y, Xte)
    assert (prob >= 0).all() and (prob <= 1).all()
    pred = (prob >= 0.5).astype(int)
    true = (Xte["x0"] > 0).astype(int).to_numpy()
    assert (pred == true).all()


@pytest.mark.skipif("ridge" not in mo.model_names(), reason="sklearn missing")
def test_ridge_separable():
    X, y, Xte = _separable(seed=3)
    prob = mo.fit_and_predict("ridge", X, y, Xte)
    pred = (prob >= 0.5).astype(int)
    true = (Xte["x0"] > 0).astype(int).to_numpy()
    assert (pred == true).all()


@pytest.mark.skipif("rf" not in mo.model_names(), reason="sklearn missing")
def test_rf_separable_and_seeded_deterministic():
    X, y, Xte = _separable(seed=5)
    p1 = mo.fit_and_predict("rf", X, y, Xte, seed=42)
    p2 = mo.fit_and_predict("rf", X, y, Xte, seed=42)
    assert np.allclose(p1, p2, atol=1e-12)
    pred = (p1 >= 0.5).astype(int)
    true = (Xte["x0"] > 0).astype(int).to_numpy()
    assert (pred == true).mean() >= 0.9


@pytest.mark.skipif("xgboost" not in mo.model_names(), reason="xgboost missing")
def test_xgboost_separable():
    X, y, Xte = _separable(seed=7)
    prob = mo.fit_and_predict("xgboost", X, y, Xte)
    pred = (prob >= 0.5).astype(int)
    true = (Xte["x0"] > 0).astype(int).to_numpy()
    assert (pred == true).mean() >= 0.9


@pytest.mark.skipif("logistic" not in mo.model_names(), reason="sklearn missing")
def test_nan_test_rows_stay_nan():
    X, y, Xte = _separable()
    Xte = Xte.copy()
    Xte.iloc[1, 0] = np.nan
    prob = mo.fit_and_predict("logistic", X, y, Xte)
    assert np.isnan(prob[1])
    assert ~np.isnan(prob[[0, 2]]).any()


@pytest.mark.skipif("logistic" not in mo.model_names(), reason="sklearn missing")
def test_nan_train_rows_dropped():
    X, y, Xte = _separable()
    X = X.copy()
    X.iloc[5, 1] = np.nan
    prob = mo.fit_and_predict("logistic", X, y, Xte)
    pred = (prob >= 0.5).astype(int)
    true = (Xte["x0"] > 0).astype(int).to_numpy()
    assert (pred == true).all()