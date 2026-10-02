"""ML model zoo + the existing rule-based voting baseline (Task 8).

Models (all causal, sklearn-free import guarantees):
- "voting"   : faithful, row-by-row replica of the existing rule-based engine in
               `services/signal_service.generate_trade_signal` (SMA20/50 trend,
               RSI<30/>70, MACD>signal, news sentiment sign). No sklearn needed.
- "logistic" : LogisticRegression (balanced class weights).
- "ridge"    : RidgeClassifier (decision-function -> Platt-style sigmoid prob).
- "rf"       : RandomForestClassifier (depth-capped, balanced).
- "xgboost"  : XGBClassifier (hist, depth-capped, no gpu).

sklearn / xgboost imports are guarded so the whole package still imports
cleanly (and the suite can skip) on machines without them.
"""

from __future__ import annotations

from math import exp
from typing import Any

import numpy as np
import pandas as pd

# Guarded heavy imports — unavailable backends just drop out of `available_models`.
try:  # pragma: no cover - exercised implicitly
    from scipy.special import expit  # type: ignore
except Exception:  # noqa: BLE001  (scipy ships with sklearn, but keep it safe)
    def expit(x):  # type: ignore[no-redef]
        x = float(x)
        if x > 0:
            return 1.0 / (1.0 + exp(-x))
        ex = exp(x)
        return ex / (1.0 + ex)


try:  # pragma: no cover - exercised implicitly
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_OK = True
except Exception:  # noqa: BLE001
    LogisticRegression = RidgeClassifier = RandomForestClassifier = None  # type: ignore[assignment]
    _SKLEARN_OK = False

try:  # pragma: no cover - exercised implicitly
    from xgboost import XGBClassifier  # type: ignore
    _XGB_OK = True
except Exception:  # noqa: BLE001
    XGBClassifier = None  # type: ignore[assignment]
    _XGB_OK = False


# ---------------------------------------------------------------------------
# Voting baseline (replica of signal_service.generate_trade_signal)
# ---------------------------------------------------------------------------

VOTING_ALIASES: dict[str, tuple[str, ...]] = {
    "sma20": ("sma20", "sma_20", "ma20"),
    "sma50": ("sma50", "sma_50", "ma50"),
    "rsi": ("rsi", "rsi14", "rsi_14"),
    "macd": ("macd", "macd_line"),
    "signal": ("macd_signal", "signal_line", "macd_signal_line", "signal", "macd_signal_9"),
    "sentiment": ("sentiment", "sentiment_score", "sent_score_5d", "sent_score_14d"),
    "prediction": ("prediction",),
}

VOTING_BULL_PB = 0.6
VOTING_BEAR_PB = 0.4
VOTING_HOLD_PB = 0.5


def resolve_column(columns, aliases) -> str | None:
    """First available column for any alias name."""
    for alias in aliases:
        if alias in columns:
            return alias
    return None


def voting_direction_from_frame(df: pd.DataFrame) -> pd.Series:
    """Per-row voting direction (+1 buy / -1 sell / 0 hold) matching the 5
    buckets used by `generate_trade_signal` (Trend, RSI, MACD, Forecast,
    Sentiment). Any missing indicator simply abstains."""
    cols = list(df.columns)

    def votes(row: pd.Series) -> float:
        score = 0.0
        c = resolve_column(cols, VOTING_ALIASES["sma20"])
        c50 = resolve_column(cols, VOTING_ALIASES["sma50"])
        if c is not None and c50 is not None and pd.notna(row[c]) and pd.notna(row[c50]):
            score += 1.0 if row[c] > row[c50] else -1.0
        c = resolve_column(cols, VOTING_ALIASES["rsi"])
        if c is not None and pd.notna(row[c]):
            if row[c] < 30:
                score += 1.0
            elif row[c] > 70:
                score -= 1.0
        cm = resolve_column(cols, VOTING_ALIASES["macd"])
        cs = resolve_column(cols, VOTING_ALIASES["signal"])
        if cm is not None and cs is not None and pd.notna(row[cm]) and pd.notna(row[cs]):
            score += 1.0 if row[cm] > row[cs] else -1.0
        cp = resolve_column(cols, VOTING_ALIASES["prediction"])
        if cp is not None and isinstance(row[cp], str):
            if "bullish" in row[cp].lower() or "Bullish" in row[cp]:
                score += 1.0
            elif "bearish" in row[cp].lower() or "Bearish" in row[cp]:
                score -= 1.0
        csn = resolve_column(cols, VOTING_ALIASES["sentiment"])
        if csn is not None and pd.notna(row[csn]):
            if row[csn] > 0.3:
                score += 1.0
            elif row[csn] < -0.3:
                score -= 1.0
        if score > 0:
            return 1.0
        if score < 0:
            return -1.0
        return 0.0

    return df.apply(votes, axis=1)


def model_names() -> list[str]:
    out = ["voting"]
    if _SKLEARN_OK:
        out += ["logistic", "ridge", "rf"]
    if _XGB_OK:
        out += ["xgboost"]
    return out


# ---------------------------------------------------------------------------
# Training / prediction helpers
# ---------------------------------------------------------------------------

DEFAULT_RF = dict(n_estimators=50, max_depth=8, min_samples_leaf=30,
                  class_weight="balanced", n_jobs=1)
DEFAULT_XGB = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                   n_jobs=1, eval_metric="logloss")


def _scale_pos_weight(y: pd.Series) -> float:
    """XGBoost imbalance weight = neg/pos, matching class_weight='balanced'."""
    yv = y.to_numpy(dtype=float)
    yv = yv[~np.isnan(yv)].astype(int)
    pos = int((yv == 1).sum())
    neg = int((yv == 0).sum())
    if pos == 0 or neg == 0:
        return 1.0
    return neg / pos


def _build_estimator(model: str, seed: int = 0, scale_pos_weight: float = 1.0):
    if model == "logistic":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        ])
    if model == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RidgeClassifier(alpha=1.0, class_weight="balanced")),
        ])
    if model == "rf":
        return RandomForestClassifier(**(dict(DEFAULT_RF, random_state=seed)))
    if model == "xgboost":
        return XGBClassifier(**(dict(DEFAULT_XGB, random_state=seed, scale_pos_weight=scale_pos_weight)))
    raise ValueError(f"unknown model: {model}")


def _prob_up(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    # RidgeClassifier: Platt-style sigmoid of the decision function.
    return np.asarray([expit(float(dec)) for dec in model.decision_function(X)], dtype=float)


def predict_up(est, X_test: pd.DataFrame) -> np.ndarray:
    """P(up) on an already-fitted estimator (NaN test rows -> NaN)."""
    pred_in = X_test.to_numpy(dtype=float)
    if np.isnan(pred_in).any():
        out = np.full(len(X_test), np.nan)
        ok = ~np.isnan(pred_in).any(axis=1)
        out[ok] = _prob_up(est, pred_in[ok])
        return out
    return _prob_up(est, pred_in)


def fit_model(
    model: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int = 0,
) -> Any:
    """Fit a model ONCE and return the estimator (call `predict_up` on it).

    Rows with NaN features/labels are dropped from training. For "voting" the
    estimator is not needed (probabilities come from the frame directly), so a
    fitted proxy is returned to keep the signature uniform.
    """
    if model == "voting":
        return _build_estimator("rf", seed=seed).fit(
            np.zeros((2, X_train.shape[1])), np.array([0, 1])
        )
    est = _build_estimator(
        model, seed=seed,
        scale_pos_weight=_scale_pos_weight(y_train) if model == "xgboost" else 1.0,
    )
    Xv_raw = X_train.to_numpy(dtype=float)
    yv_raw = y_train.to_numpy(dtype=float)
    mask = (y_train.notna().to_numpy()) & ~np.isnan(Xv_raw).any(axis=1)
    if int(mask.sum()) == 0:
        raise ValueError(f"no valid training rows for {model}")
    est.fit(Xv_raw[mask], yv_raw[mask].astype(int))
    return est


def fit_and_predict(
    model: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    seed: int = 0,
    **kwargs,
) -> np.ndarray:
    """Train a probabilistic up/down model and return P(up) for X_test.

    Rows with NaN features/labels are dropped from training; test rows simply
    keep NaN so upstream can impute/measure honestly (never fabricated).
    """
    if model == "voting":
        return np.clip(X_test.apply(
            lambda r: voting_direction_from_frame(pd.DataFrame([r])).iloc[0], axis=1
        ).map({
            1.0: VOTING_BULL_PB, -1.0: VOTING_BEAR_PB, 0.0: VOTING_HOLD_PB,
        }).to_numpy(dtype=float), 0.0, 1.0)

    est = _build_estimator(
        model, seed=seed,
        scale_pos_weight=_scale_pos_weight(y_train) if model == "xgboost" else 1.0,
    )
    Xv_raw = X_train.to_numpy(dtype=float)
    yv_raw = y_train.to_numpy(dtype=float)
    mask = (y_train.notna().to_numpy()) & ~np.isnan(Xv_raw).any(axis=1)
    if int(mask.sum()) == 0:
        return np.full(len(X_test), np.nan)
    yv = yv_raw[mask].astype(int)
    Xv = Xv_raw[mask]
    est.fit(Xv, yv)
    pred_in = X_test.to_numpy(dtype=float)
    if np.isnan(pred_in).any():
        out = np.full(len(X_test), np.nan)
        ok = ~np.isnan(pred_in).any(axis=1)
        out[ok] = _prob_up(est, pred_in[ok])
        return out
    return _prob_up(est, pred_in)