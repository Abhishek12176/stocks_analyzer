"""Ensemble model (Task 9).

Averaging the per-model P(up) estimates is a simple, robust way to cancel
model-specific noise. This module:
- `combine_probs()`  : weighted/equal average of P(up) vectors. Missing (NaN)
  output of one model for a row is handled by renormalising the weights over
  the models that DID produce a value for that row (never fabricate a number);
  a row where every model is NaN stays NaN.
- `fit_predict_ensemble()` : fit each model once, combine their test P(up).

Every model in `model_names()` can participate (voting included: its coarse
0.6/0.4/0.5 buckets average naturally with the probabilistic models).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import settings
from app.ml import models as mo


def combine_probs(
    model_probs: list[np.ndarray | pd.Series],
    weights: list[float] | dict[str, float] | None = None,
) -> np.ndarray:
    """Probability averaging with per-row NaN awareness.

    `weights` may be a list aligned to `model_probs`, a dict keyed by model
    name (only used by `fit_predict_ensemble`), or None (equal weights).
    Output is clipped to [0, 1].
    """
    arr = np.vstack([np.asarray(p, dtype=float) for p in model_probs])
    k, n = arr.shape
    if weights is None:
        w = np.ones(k, dtype=float)
    else:
        w = np.asarray(list(weights.values()) if isinstance(weights, dict)
                       else weights, dtype=float)
        if len(w) != k or not np.isfinite(w).all():
            raise ValueError("weights must be finite and aligned to model_probs")
        if w.min() <= 0.0:
            raise ValueError("weights must be strictly positive")
    w = w / w.sum()
    wmat = w[:, None]
    present = ~np.isnan(arr)
    denom = (present * wmat).sum(axis=0)
    numer = np.where(present, arr * wmat, 0.0).sum(axis=0)
    out = np.where(denom == 0.0, np.nan, numer / np.where(denom == 0.0, 1.0, denom))
    return np.clip(out, 0.0, 1.0)


def fit_predict_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    models: tuple[str, ...] = ("logistic", "rf", "xgboost"),
    weights: dict[str, float] | None = None,
    seed: int = 0,
) -> dict:
    """Fit every requested (installed) model and combine its test P(up).

    Returns a dict with the combined `prob`, per-model `model_probs`,
    `models` actually used, `weights`, and the configured `version`.
    """
    used = [m for m in models if m in mo.model_names()]
    if not used:
        raise ValueError("no usable model in the requested set")
    if weights is not None:
        missing = set(weights) - set(used)
        if missing:
            raise ValueError(f"weights reference unavailable models: {sorted(missing)}")

    per: dict[str, np.ndarray] = {}
    for i, m in enumerate(used):
        per[m] = mo.fit_and_predict(m, X_train, y_train, X_test, seed=seed + i)

    if weights is None:
        weights_list: list[float] | dict[str, float] | None = None
    else:
        weights_list = {m: weights[m] for m in used}
    combined = combine_probs([per[m] for m in used], weights=weights_list)
    return {
        "prob": combined,
        "model_probs": per,
        "models": used,
        "weights": [1.0 / len(used)] * len(used)
        if weights is None else [weights[m] for m in used],
        "version": settings.ml_ensemble_version,
    }