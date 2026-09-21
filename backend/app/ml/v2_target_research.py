"""Offline, research-only target alternatives for the V2 study.

This module is intentionally not imported by :mod:`app.ml.pipeline`.  It keeps
the production target frozen while making a small, pre-registered comparison
possible:

* ``target_up_*`` is the exact V1 ``return > 0`` binary target;
* ``target_meaningful_up_*`` uses one fixed 1% return hurdle;
* ``target_vol_adj_*`` uses a fixed one trailing-volatility-unit hurdle, where
  volatility is computed from returns strictly before the row;
* ``target_three_class_*`` is DOWN/NEUTRAL/UP around the same fixed 1% hurdle;
* ``target_regression_*`` is the unchanged continuous forward return.

All functions are deterministic and operate on data supplied by the caller.
No network access, model persistence, production integration, threshold
search, or backtest accounting is performed here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from app.ml import dataset as v1_dataset
from app.ml import fingerprint as fp
from app.ml.backtest import purged_walk_forward_splits

TARGET_VERSION = "v2-target-research-1"
SCHEMA_VERSION = "v2-target-panel-1"
MEANINGFUL_RETURN_THRESHOLD = 0.01
VOL_ADJUSTED_THRESHOLD = 1.0
DEFAULT_VOLATILITY_WINDOW = 20
DEFAULT_HORIZON = v1_dataset.DEFAULT_HORIZON
DOWN = -1
NEUTRAL = 0
UP = 1

_NEAR_ZERO_BUCKETS = (0.001, 0.005, 0.01, 0.02)


def _check_positive(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def forward_return(close: pd.Series, horizon: int = DEFAULT_HORIZON) -> pd.Series:
    """Return ``Close[T+h] / Close[T] - 1`` without changing the index."""
    horizon = _check_positive(horizon, "horizon")
    values = pd.to_numeric(close, errors="coerce")
    denominator = values.where(values.ne(0))
    return values.shift(-horizon).div(denominator).sub(1.0)


def _date_values(frame: pd.DataFrame, date_col: str) -> pd.Series:
    if date_col in frame.columns:
        dates = pd.to_datetime(frame[date_col], errors="coerce")
    elif isinstance(frame.index, pd.DatetimeIndex):
        dates = pd.Series(frame.index, index=frame.index)
    else:
        dates = pd.Series(pd.NaT, index=frame.index)
    if dates.isna().any():
        raise ValueError("a panel target frame requires valid dates")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    return dates.dt.normalize()


def _panel_forward_return(
    frame: pd.DataFrame,
    *,
    close_col: str,
    symbol_col: str,
    date_col: str,
    horizon: int,
) -> pd.Series:
    """Compute a forward return independently for every symbol."""
    if close_col not in frame.columns:
        raise ValueError(f"frame missing {close_col!r}")
    dates = _date_values(frame, date_col)
    symbols = (
        frame[symbol_col].astype(str)
        if symbol_col in frame.columns
        else pd.Series("__single_symbol__", index=frame.index)
    )
    keys = pd.DataFrame({"symbol": symbols.to_numpy(), "date": dates.to_numpy()})
    if keys.duplicated().any():
        raise ValueError("frame must contain one row per (symbol, date)")
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    for symbol in sorted(symbols.unique()):
        positions = np.flatnonzero(symbols.to_numpy() == symbol)
        ordered = positions[np.argsort(dates.iloc[positions].to_numpy(), kind="mergesort")]
        values = pd.to_numeric(frame.iloc[ordered][close_col], errors="coerce")
        values = values.where(values.ne(0))
        returns = values.shift(-horizon).div(values).sub(1.0)
        output.iloc[ordered] = returns.to_numpy()
    return output


def causal_past_volatility(
    close: pd.Series,
    window: int = DEFAULT_VOLATILITY_WINDOW,
    *,
    ddof: int = 1,
) -> pd.Series:
    """Trailing volatility at T from returns ending at T-1 only.

    The explicit ``shift(1)`` is deliberate: changing a close at T or any
    future close cannot change the volatility used by a target at T.
    """
    window = _check_positive(window, "window")
    if int(ddof) < 0:
        raise ValueError("ddof must be non-negative")
    returns = pd.to_numeric(close, errors="coerce").pct_change(fill_method=None)
    return returns.shift(1).rolling(window=window, min_periods=window).std(ddof=int(ddof))


def _panel_past_volatility(
    frame: pd.DataFrame,
    *,
    close_col: str,
    symbol_col: str,
    date_col: str,
    window: int,
) -> pd.Series:
    dates = _date_values(frame, date_col)
    symbols = (
        frame[symbol_col].astype(str)
        if symbol_col in frame.columns
        else pd.Series("__single_symbol__", index=frame.index)
    )
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    for symbol in sorted(symbols.unique()):
        positions = np.flatnonzero(symbols.to_numpy() == symbol)
        ordered = positions[np.argsort(dates.iloc[positions].to_numpy(), kind="mergesort")]
        vol = causal_past_volatility(frame.iloc[ordered][close_col], window)
        output.iloc[ordered] = vol.to_numpy()
    return output


def v1_binary_label(forward: pd.Series) -> pd.Series:
    """Frozen V1 label: positive return is UP; zero is DOWN."""
    result = (forward > 0).astype(float)
    return result.where(forward.notna())


def frozen_v1_binary(forward: pd.Series) -> pd.Series:
    """Explicit name for the frozen production comparison target."""
    return v1_binary_label(forward)


def meaningful_return_label(
    forward: pd.Series,
    threshold: float = MEANINGFUL_RETURN_THRESHOLD,
) -> pd.Series:
    """Binary meaningful-gain label using a fixed, non-tuned return hurdle."""
    threshold = float(threshold)
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    result = (forward > threshold).astype(float)
    return result.where(forward.notna())


def fixed_meaningful_return(forward: pd.Series) -> pd.Series:
    """Apply the pre-registered 1% meaningful-return hurdle."""
    return meaningful_return_label(forward, MEANINGFUL_RETURN_THRESHOLD)


def three_class_label(
    forward: pd.Series,
    threshold: float = MEANINGFUL_RETURN_THRESHOLD,
) -> pd.Series:
    """DOWN=-1, NEUTRAL=0, UP=1 with strict threshold boundaries."""
    threshold = float(threshold)
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    result = pd.Series(NEUTRAL, index=forward.index, dtype=float)
    result.loc[forward < -threshold] = DOWN
    result.loc[forward > threshold] = UP
    return result.where(forward.notna())


def fixed_three_class(
    forward: pd.Series,
    threshold: float = MEANINGFUL_RETURN_THRESHOLD,
) -> pd.Series:
    """Explicit alias for the fixed DOWN/NEUTRAL/UP candidate."""
    return three_class_label(forward, threshold)


def volatility_adjusted_label(
    forward: pd.Series,
    past_volatility: pd.Series,
    threshold: float = VOL_ADJUSTED_THRESHOLD,
) -> pd.Series:
    """Binary label for a return exceeding a fixed past-volatility unit."""
    threshold = float(threshold)
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    score = forward.div(past_volatility.where(past_volatility > 0))
    result = (score > threshold).astype(float)
    return result.where(score.notna())


def volatility_adjusted_class(
    forward: pd.Series,
    past_volatility: pd.Series,
    threshold: float = VOL_ADJUSTED_THRESHOLD,
) -> pd.Series:
    """Three-class signed version of :func:`volatility_adjusted_label`."""
    threshold = float(threshold)
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    score = forward.div(past_volatility.where(past_volatility > 0))
    result = pd.Series(NEUTRAL, index=forward.index, dtype=float)
    result.loc[score < -threshold] = DOWN
    result.loc[score > threshold] = UP
    return result.where(score.notna())


def causal_volatility_adjusted_direction(
    forward: pd.Series,
    past_volatility: pd.Series,
    threshold: float = VOL_ADJUSTED_THRESHOLD,
) -> pd.Series:
    """Explicit alias for the causal volatility-adjusted binary candidate."""
    return volatility_adjusted_label(forward, past_volatility, threshold)


def add_target_candidates(
    frame: pd.DataFrame,
    *,
    horizon: int = DEFAULT_HORIZON,
    close_col: str = "Close",
    symbol_col: str = "symbol",
    date_col: str = "date",
    volatility_window: int = DEFAULT_VOLATILITY_WINDOW,
    meaningful_threshold: float = MEANINGFUL_RETURN_THRESHOLD,
    volatility_threshold: float = VOL_ADJUSTED_THRESHOLD,
    as_of: Any = None,
) -> pd.DataFrame:
    """Attach all candidate targets to a copy of a causal feature frame.

    ``as_of`` is applied before forward labels are constructed.  Consequently
    rows after the cutoff cannot provide a label for a row before the cutoff.
    """
    horizon = _check_positive(horizon, "horizon")
    volatility_window = _check_positive(volatility_window, "volatility_window")
    out = frame.copy()
    if not isinstance(out, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if as_of is not None:
        dates = _date_values(out, date_col)
        cutoff = pd.Timestamp(as_of)
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_localize(None)
        out = out.loc[dates <= cutoff.normalize()].copy()
    forward = _panel_forward_return(
        out, close_col=close_col, symbol_col=symbol_col, date_col=date_col, horizon=horizon
    )
    past_vol = _panel_past_volatility(
        out,
        close_col=close_col,
        symbol_col=symbol_col,
        date_col=date_col,
        window=volatility_window,
    )
    out[f"target_ret_{horizon}d"] = forward
    out[f"target_up_{horizon}d"] = v1_binary_label(forward)
    out[f"target_meaningful_up_{horizon}d"] = meaningful_return_label(
        forward, meaningful_threshold
    )
    out[f"past_volatility_{volatility_window}d"] = past_vol
    out[f"target_vol_adj_up_{horizon}d"] = volatility_adjusted_label(
        forward, past_vol, volatility_threshold
    )
    out[f"target_vol_adj_class_{horizon}d"] = volatility_adjusted_class(
        forward, past_vol, volatility_threshold
    )
    out[f"target_three_class_{horizon}d"] = three_class_label(
        forward, meaningful_threshold
    )
    out[f"target_regression_{horizon}d"] = forward
    out.attrs["target_version"] = TARGET_VERSION
    out.attrs["schema_version"] = SCHEMA_VERSION
    out.attrs["target_config"] = {
        "horizon": horizon,
        "meaningful_threshold": float(meaningful_threshold),
        "volatility_threshold": float(volatility_threshold),
        "volatility_window": volatility_window,
        "volatility_ddof": 1,
        "as_of": str(pd.Timestamp(as_of).normalize().date()) if as_of is not None else None,
    }
    return out


def build_target_candidates(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Readable alias for :func:`add_target_candidates`."""
    return add_target_candidates(*args, **kwargs)


def continuous_return_target(forward: pd.Series) -> pd.Series:
    """Return the continuous target unchanged, preserving NaN tails."""
    return pd.to_numeric(forward, errors="coerce").copy()


def build_targets(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Short alias retained for experiment scripts."""
    return add_target_candidates(*args, **kwargs)


def target_manifest(
    *,
    horizon: int = DEFAULT_HORIZON,
    meaningful_threshold: float = MEANINGFUL_RETURN_THRESHOLD,
    volatility_window: int = DEFAULT_VOLATILITY_WINDOW,
    volatility_threshold: float = VOL_ADJUSTED_THRESHOLD,
) -> dict[str, Any]:
    """Return the immutable target definitions used in reports."""
    return {
        "target_version": TARGET_VERSION,
        "schema_version": SCHEMA_VERSION,
        "horizon": int(horizon),
        "v1_binary": {
            "name": f"target_up_{horizon}d",
            "formula": "1 if Close[T+h]/Close[T]-1 > 0, else 0",
            "status": "frozen comparison",
        },
        "meaningful_binary": {
            "name": f"target_meaningful_up_{horizon}d",
            "formula": f"1 if forward return > {float(meaningful_threshold):g}, else 0",
            "threshold": float(meaningful_threshold),
            "rationale": (
                "A fixed 1% 20-day hurdle is a deliberately conservative "
                "noise/cost screen; it is pre-registered and not tuned on data."
            ),
        },
        "volatility_adjusted": {
            "name": f"target_vol_adj_up_{horizon}d",
            "formula": "1 if forward return / past_volatility > 1, else 0",
            "threshold": float(volatility_threshold),
            "past_volatility": (
                f"sample std of the prior {int(volatility_window)} close returns; "
                "the return at T is excluded"
            ),
        },
        "three_class": {
            "name": f"target_three_class_{horizon}d",
            "formula": "DOWN=-1 if r < -threshold; NEUTRAL if |r| <= threshold; UP=1 if r > threshold",
            "threshold": float(meaningful_threshold),
            "labels": {"DOWN": DOWN, "NEUTRAL": NEUTRAL, "UP": UP},
        },
        "regression": {
            "name": f"target_regression_{horizon}d",
            "formula": "Close[T+h]/Close[T]-1",
            "meaning": "continuous forward return",
        },
        "leakage_policy": (
            "features and past volatility end at T; forward values are labels only; "
            "as_of truncation precedes target construction"
        ),
    }


def label_stats(frame: pd.DataFrame, target_columns: Sequence[str]) -> dict[str, Any]:
    """Counts and proportions, retaining NaN/undefined rows explicitly."""
    result: dict[str, Any] = {}
    for column in target_columns:
        if column not in frame:
            raise ValueError(f"missing target column {column!r}")
        series = frame[column]
        defined = series.dropna()
        counts = {str(key): int(value) for key, value in defined.value_counts().sort_index().items()}
        n = int(len(defined))
        entry: dict[str, Any] = {
            "rows": int(len(series)),
            "defined": n,
            "undefined": int(len(series) - n),
            "counts": counts,
            "proportions": {key: (value / n if n else None) for key, value in counts.items()},
        }
        if set(defined.unique()).issubset({-1.0, 0.0, 1.0}):
            entry["class_counts"] = {
                "DOWN": int((defined == DOWN).sum()),
                "NEUTRAL": int((defined == NEUTRAL).sum()),
                "UP": int((defined == UP).sum()),
            }
        elif set(defined.unique()).issubset({0.0, 1.0}):
            entry["class_counts"] = {
                "DOWN": int((defined == 0).sum()),
                "UP": int((defined == 1).sum()),
            }
        result[column] = entry
    return result


def near_zero_buckets(
    forward: pd.Series,
    *,
    boundaries: Sequence[float] = _NEAR_ZERO_BUCKETS,
) -> dict[str, int]:
    """Deterministic absolute-return buckets used to audit threshold effects."""
    bounds = tuple(float(value) for value in boundaries)
    if any(value <= 0 for value in bounds) or tuple(sorted(bounds)) != bounds:
        raise ValueError("boundaries must be positive and sorted")
    values = pd.to_numeric(forward, errors="coerce").dropna().abs()
    edges = (0.0,) + bounds + (float("inf"),)
    result: dict[str, int] = {}
    for left, right in zip(edges[:-1], edges[1:]):
        label = f"[{left:g},{right:g})" if np.isfinite(right) else f"[{left:g},inf)"
        result[label] = int(((values >= left) & (values < right)).sum())
    return result


def agreement_correlation_diagnostics(
    frame: pd.DataFrame,
    target_columns: Sequence[str],
) -> dict[str, Any]:
    """Pairwise agreement for labels and Pearson correlation for numeric targets."""
    agreements: dict[str, Any] = {}
    correlations: dict[str, Any] = {}
    for left_i, left in enumerate(target_columns):
        for right in target_columns[left_i + 1 :]:
            if left not in frame or right not in frame:
                raise ValueError("all target columns must be present")
            pair = frame[[left, right]].dropna()
            key = f"{left}__{right}"
            n = int(len(pair))
            agreements[key] = {
                "n": n,
                "equal": int((pair[left] == pair[right]).sum()) if n else 0,
                "rate": float((pair[left] == pair[right]).mean()) if n else None,
            }
            if n >= 2 and pair[left].nunique() > 1 and pair[right].nunique() > 1:
                corr = float(pair[left].corr(pair[right]))
            else:
                corr = None
            correlations[key] = {"n": n, "pearson": corr}
    return {"agreement": agreements, "correlation": correlations}


def _numeric_features(
    frame: pd.DataFrame,
    feature_cols: Sequence[str] | None,
) -> list[str]:
    if feature_cols is not None:
        missing = [column for column in feature_cols if column not in frame]
        if missing:
            raise ValueError(f"missing feature columns: {missing}")
        return list(feature_cols)
    excluded = {"symbol", "date"}
    return [
        column
        for column in frame.columns
        if column not in excluded
        and not str(column).startswith("target_")
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _impute_fit_transform(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    medians = train.median(numeric_only=True).reindex(train.columns).fillna(0.0)
    return (
        train.fillna(medians).fillna(0.0).to_numpy(dtype=float),
        test.fillna(medians).fillna(0.0).to_numpy(dtype=float),
    )


def _classification_predictions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    from sklearn.linear_model import LogisticRegression

    classes = np.unique(y_train)
    if len(classes) < 2:
        return np.full(len(x_test), classes[0] if len(classes) else np.nan), None
    model = LogisticRegression(max_iter=1000, random_state=int(seed), solver="lbfgs")
    model.fit(x_train, y_train)
    return model.predict(x_test), model.predict_proba(x_test)


def _regression_predictions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    from sklearn.linear_model import Ridge

    model = Ridge(alpha=1.0)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def _classification_metrics(y: np.ndarray, pred: np.ndarray, proba: np.ndarray | None) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

    if len(y) == 0:
        return {"n": 0, "accuracy": None, "balanced_accuracy": None, "log_loss": None}
    balanced = (
        float(accuracy_score(y, pred))
        if len(np.unique(y)) < 2
        else float(balanced_accuracy_score(y, pred))
    )
    result: dict[str, Any] = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": balanced,
        "log_loss": None,
    }
    if proba is not None:
        try:
            result["log_loss"] = float(log_loss(y, proba, labels=np.unique(y)))
        except ValueError:
            pass
    return result


def _regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    if len(y) == 0:
        return {"n": 0, "mae": None, "rmse": None, "correlation": None}
    errors = pred - y
    corr = float(np.corrcoef(y, pred)[0, 1]) if len(y) > 1 and np.std(y) and np.std(pred) else None
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "correlation": corr,
    }


def _aggregate_metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    numeric = [key for key in rows[0] if key != "n" and isinstance(rows[0][key], (float, int))]
    total = sum(int(row.get("n", 0)) for row in rows)
    result: dict[str, Any] = {"n": total}
    for key in numeric:
        values = [(float(row[key]), int(row["n"])) for row in rows if row.get(key) is not None]
        result[key] = (
            sum(value * count for value, count in values) / sum(count for _, count in values)
            if values
            else None
        )
    return result


def _metric_for_target(
    work: pd.DataFrame,
    *,
    target: str,
    classification: bool,
    features: list[str],
    train_dates: pd.Index,
    test_dates: pd.Index,
    seed: int,
) -> dict[str, Any]:
    dates = _date_values(work, "date")
    train_mask = dates.isin(train_dates) & work[target].notna()
    test_mask = dates.isin(test_dates) & work[target].notna()
    if not train_mask.any() or not test_mask.any():
        return {"aggregate": {"n": 0}, "per_symbol": {}, "n_train": int(train_mask.sum())}
    train = work.loc[train_mask, features]
    test = work.loc[test_mask, features]
    x_train, x_test = _impute_fit_transform(train, test)
    y_train = work.loc[train_mask, target].to_numpy(dtype=float)
    y_test = work.loc[test_mask, target].to_numpy(dtype=float)
    if classification:
        pred, proba = _classification_predictions(x_train, y_train, x_test, seed)
    else:
        pred = _regression_predictions(x_train, y_train, x_test)
        proba = None
    symbols = (
        work.loc[test_mask, "symbol"].astype(str)
        if "symbol" in work.columns
        else pd.Series("__single_symbol__", index=work.index[test_mask])
    )
    per_symbol: dict[str, Any] = {}
    for symbol in sorted(symbols.unique()):
        mask = symbols.to_numpy() == symbol
        per_symbol[symbol] = (
            _classification_metrics(y_test[mask], pred[mask], proba[mask] if proba is not None else None)
            if classification
            else _regression_metrics(y_test[mask], pred[mask])
        )
    aggregate = (
        _classification_metrics(y_test, pred, proba)
        if classification
        else _regression_metrics(y_test, pred)
    )
    return {
        "aggregate": aggregate,
        "per_symbol": per_symbol,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
    }


def _date_split(
    dates: pd.DatetimeIndex,
    *,
    final_holdout_fraction: float,
    test_size: int | None,
    min_train: int | None,
    embargo: int,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]]:
    unique = pd.DatetimeIndex(sorted(set(dates)))
    if len(unique) < 2:
        return unique, unique[:0], []
    fraction = float(final_holdout_fraction)
    if not 0 < fraction < 1:
        raise ValueError("final_holdout_fraction must be between 0 and 1")
    holdout_n = max(1, int(np.ceil(len(unique) * fraction)))
    research = unique[:-holdout_n]
    holdout = unique[-holdout_n:]
    if len(research) < 2:
        return research, holdout, []
    size = int(test_size or max(1, min(20, len(research) // 5)))
    warmup = int(min_train or max(2, min(20, len(research) - size)))
    folds = purged_walk_forward_splits(
        research, size, step=size, min_train=warmup, embargo=embargo
    )
    return research, holdout, folds


def run_target_research(
    frame: pd.DataFrame,
    *,
    feature_cols: Sequence[str] | None = None,
    horizon: int = DEFAULT_HORIZON,
    volatility_window: int = DEFAULT_VOLATILITY_WINDOW,
    meaningful_threshold: float = MEANINGFUL_RETURN_THRESHOLD,
    volatility_threshold: float = VOL_ADJUSTED_THRESHOLD,
    final_holdout_fraction: float = 0.2,
    test_size: int | None = None,
    min_train: int | None = None,
    seed: int = 0,
    as_of: Any = None,
) -> dict[str, Any]:
    """Compare targets with identical causal folds and a never-trained holdout."""
    work = add_target_candidates(
        frame,
        horizon=horizon,
        volatility_window=volatility_window,
        meaningful_threshold=meaningful_threshold,
        volatility_threshold=volatility_threshold,
        as_of=as_of,
    )
    features = _numeric_features(work, feature_cols)
    if not features:
        raise ValueError("at least one numeric feature is required")
    target_names = {
        "v1_binary": f"target_up_{horizon}d",
        "meaningful_binary": f"target_meaningful_up_{horizon}d",
        "volatility_adjusted_binary": f"target_vol_adj_up_{horizon}d",
        "three_class": f"target_three_class_{horizon}d",
        "regression": f"target_regression_{horizon}d",
    }
    classification = {key: key != "regression" for key in target_names}
    dates = _date_values(work, "date")
    research_dates, holdout_dates, folds = _date_split(
        pd.DatetimeIndex(dates),
        final_holdout_fraction=final_holdout_fraction,
        test_size=test_size,
        min_train=min_train,
        embargo=horizon,
    )
    # The final holdout is evaluation-only. Remove the last horizon dates from
    # its training side so their forward labels cannot overlap the holdout.
    final_train_dates = (
        research_dates[:-horizon] if len(research_dates) > horizon else research_dates[:0]
    )
    all_folds: list[dict[str, Any]] = [
        {
            "fold": int(index),
            "train_start": train.min().date().isoformat(),
            "train_end": train.max().date().isoformat(),
            "test_start": test.min().date().isoformat(),
            "test_end": test.max().date().isoformat(),
            "train_rows": int(dates.isin(train).sum()),
            "test_rows": int(dates.isin(test).sum()),
        }
        for index, (train, test) in enumerate(folds)
    ]
    metrics: dict[str, Any] = {}
    for kind, target in target_names.items():
        fold_results = []
        for index, (train, test) in enumerate(folds):
            result = _metric_for_target(
                work,
                target=target,
                classification=classification[kind],
                features=features,
                train_dates=train,
                test_dates=test,
                seed=seed,
            )
            result["fold"] = index
            fold_results.append(result)
        final = _metric_for_target(
            work,
            target=target,
            classification=classification[kind],
            features=features,
            train_dates=final_train_dates,
            test_dates=holdout_dates,
            seed=seed,
        )
        metrics[kind] = {
            "target": target,
            "model": "logistic" if classification[kind] else "ridge",
            "walk_forward": {
                "folds": fold_results,
                "aggregate": _aggregate_metric([item["aggregate"] for item in fold_results]),
            },
            "final_holdout": final,
        }
    target_cols = list(target_names.values())
    forward_col = f"target_ret_{horizon}d"
    diagnostics = agreement_correlation_diagnostics(work, target_cols)
    diagnostics["near_zero_buckets"] = near_zero_buckets(work[forward_col])
    config = {
        "horizon": int(horizon),
        "volatility_window": int(volatility_window),
        "meaningful_threshold": float(meaningful_threshold),
        "volatility_threshold": float(volatility_threshold),
        "final_holdout_fraction": float(final_holdout_fraction),
        "test_size": int(test_size) if test_size is not None else None,
        "min_train": int(min_train) if min_train is not None else None,
        "seed": int(seed),
        "features": features,
        "as_of": str(pd.Timestamp(as_of).normalize().date()) if as_of is not None else None,
    }
    fingerprint = fp.compose_fingerprint(
        {
            "data": fp.fingerprint_dataframe("target-panel", work),
            "config": fp.fingerprint_object("target-config", config),
        },
        name="v2-target-research",
    )
    return {
        "metadata": {
            "target_version": TARGET_VERSION,
            "schema_version": SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "fingerprint_algorithm": "sha256",
            "no_production_integration": True,
            "research_data_only": True,
        },
        "config": config,
        "target_manifest": target_manifest(
            horizon=horizon,
            meaningful_threshold=meaningful_threshold,
            volatility_window=volatility_window,
            volatility_threshold=volatility_threshold,
        ),
        "label_stats": label_stats(work, target_cols),
        "diagnostics": diagnostics,
        "folds": all_folds,
        "final_holdout": {
            "isolated": True,
            "train_end": final_train_dates.max().date().isoformat() if len(final_train_dates) else None,
            "holdout_start": holdout_dates.min().date().isoformat() if len(holdout_dates) else None,
            "holdout_end": holdout_dates.max().date().isoformat() if len(holdout_dates) else None,
            "rows": int(dates.isin(holdout_dates).sum()),
            "embargo": int(horizon),
        },
        "metrics": metrics,
        "dataset": work,
    }


def compare_targets(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility alias for :func:`run_target_research`."""
    return run_target_research(*args, **kwargs)
