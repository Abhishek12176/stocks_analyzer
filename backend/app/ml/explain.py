"""Deterministic feature importance + local explanation (Task 9).

Full multi-tree SHAP libraries are heavy and optional; this module delivers
the same "why" answers the task asks for, with pure numpy:

- `_auc()`            : ROC-AUC with average-rank tie handling (sklearn-free).
- `_logloss()`        : binary cross-entropy on clipped probabilities.
- `permutation_importance()` : global scores = drop in AUC/logloss when a
  feature's column is shuffled (POSITIVE = feature carries signal; noisy
  features land near 0). Deterministic per (seed, feature).
- `explain_row()`     : local impact of perturbing each feature +/- delta on
  the probability (any `predict_prob` callable), returning the strongest
  positive (pushes P(up) up) and negative (pushes P(up) down) factors.
- `fit_predictor()`   : wraps a model name + training set into `predict_prob`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from app.ml import models as mo

PredictFn = Callable[[pd.DataFrame], np.ndarray]


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC (average-rank method; safe with ties)."""
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(y_score, dtype=float)
    valid = ~np.isnan(s)
    y, s = y[valid], s[valid]
    npos = int((y == 1).sum())
    nneg = int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.zeros(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    sum_pos = float(ranks[y == 1].sum())
    return float((sum_pos - npos * (npos + 1.0) / 2.0) / (npos * nneg))


def _logloss(y_true: np.ndarray, prob: np.ndarray) -> float:
    p = np.clip(np.asarray(prob, dtype=float), 1e-9, 1.0 - 1e-9)
    y = np.asarray(y_true, dtype=float)
    valid = ~np.isnan(p)
    y, p = y[valid], p[valid]
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit_predictor(model: str, X_train: pd.DataFrame, y_train: pd.Series,
                  seed: int = 0) -> PredictFn:
    """Return `predict_prob(X_test)` for a model name + training set."""
    def predict_prob(X_test: pd.DataFrame) -> np.ndarray:
        return mo.fit_and_predict(model, X_train, y_train, X_test, seed=seed)
    return predict_prob


def permutation_importance(
    X_val: pd.DataFrame,
    y_val: pd.Series,
    predict_prob: PredictFn,
    n_permutes: int = 3,
    seed: int = 0,
    metric: str = "auc",
) -> dict:
    """Global permutation importances (positive = valuable)."""
    y = np.asarray(y_val, dtype=float)
    if metric == "auc":
        score_fn: Callable[[np.ndarray], float] = lambda p: _auc(y, p)
    elif metric == "logloss":
        score_fn = lambda p: _logloss(y, p)
    else:
        raise ValueError(f"unknown metric: {metric}")

    base = score_fn(predict_prob(X_val))
    if np.isnan(base):
        raise ValueError("baseline metric is NaN (no valid labels/rows)")

    rngs = [np.random.default_rng(seed + i) for i in range(max(n_permutes, 1))]
    importance: dict[str, float] = {}
    for col in X_val.columns:
        scores = []
        Xp = X_val.copy()
        orig = Xp[col].to_numpy()
        for rng in rngs:
            shuffled = rng.permutation(orig)
            Xp[col] = shuffled
            scores.append(base - score_fn(predict_prob(Xp)))
        importance[col] = float(np.mean(scores))
    return {"baseline": float(base), "metric": metric,
            "n_permutes": int(n_permutes), "importance": importance}


def explain_row(
    predict_prob: PredictFn,
    row: pd.Series | dict,
    delta: float = 0.15,
) -> dict:
    """Local +/- impact of each feature around the given row."""
    base = float(predict_prob(pd.DataFrame([row]))[0])
    impacts: dict[str, float] = {}
    for feat, value in pd.Series(row).items():
        hi = pd.Series(row).copy()
        lo = pd.Series(row).copy()
        hi[feat] = value + delta
        lo[feat] = value - delta
        p_hi = float(predict_prob(pd.DataFrame([hi]))[0])
        p_lo = float(predict_prob(pd.DataFrame([lo]))[0])
        impacts[feat] = float((p_hi - p_lo) / 2.0)
    positive = {k: v for k, v in impacts.items() if v > 0}
    negative = {k: v for k, v in impacts.items() if v < 0}
    return {
        "baseline_prob": base,
        "delta": float(delta),
        "impacts": impacts,
        "positive": sorted(positive.items(), key=lambda kv: -kv[1]),
        "negative": sorted(negative.items(), key=lambda kv: kv[1]),
    }


def top_factors(result: dict, kind: str = "positive", k: int = 5) -> list[tuple[str, float]]:
    """Top-k (feature, impact) pairs from `explain_row`."""
    items = result.get(kind, [])
    return list(items)[:k]