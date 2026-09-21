"""Task 18 selective forecasting research.

This module is deliberately disconnected from the production forecast path.  It
evaluates fixed, uncertainty-aware abstention rules on the frozen Task 17
snapshot.  The rules are declared in this file; they are not selected on the
final holdout.  In particular, this module must never be imported by the V1
pipeline, signal, calibration, threshold, API, or frontend code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from app.ml import v2_v1_fair_compare as fair
from app.ml import versions as ver

logger = logging.getLogger("equitylens.selective_forecast")

MODULE_VERSION = "task18-selective-v1"
SCHEMA_VERSION = "task18-selective-results-1"
DEFAULT_HORIZON = 20
DEFAULT_SEEDS = (17, 31, 53)
DEFAULT_FINAL_HOLDOUT_FRACTION = 0.20
# Task 18 is deliberately a decision-layer experiment over frozen V1
# predictions.  It never fits a pooled candidate model.
V1_COMPONENT_MODELS = ("voting", "logistic", "ridge", "rf", "xgboost")
DEFAULT_MODEL_NAMES = V1_COMPONENT_MODELS
MIN_COVERAGE = 0.30
MIN_ACCURACY_UPLIFT = 0.02
MIN_RISK_REDUCTION = 0.02
MIN_SEED_COVERAGE = 0.20

# These values are frozen before looking at the final holdout.  A margin is in
# probability units; disagreement is the standard deviation across models.
POLICY_RULES: dict[str, dict[str, Any]] = {
    "always_accept": {"kind": "always", "threshold": None},
    "margin_005": {"kind": "margin", "threshold": 0.05},
    "margin_010": {"kind": "margin", "threshold": 0.10},
    "margin_015": {"kind": "margin", "threshold": 0.15},
    "disagreement_005": {"kind": "disagreement", "threshold": 0.05},
    "disagreement_010": {"kind": "disagreement", "threshold": 0.10},
    # Candidate A/B are predeclared and evaluated without holdout selection.
    "candidate_a_entropy": {"kind": "entropy", "threshold": 0.65},
    "candidate_b_disagreement": {"kind": "disagreement", "threshold": 0.05},
    # Stable short names retained for notebook/report compatibility.
    "entropy_065": {"kind": "entropy", "threshold": 0.65},
    "conformal_90": {"kind": "conformal", "alpha": 0.10},
}


def _finite(value: Any) -> Any:
    """Convert numpy scalars and non-finite values to JSON-safe values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [_finite(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def _json_dump(payload: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_finite(payload), handle, indent=2, sort_keys=True)


def load_shared_snapshot(path: str | Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the persisted Task 17 panel without making a network request."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "ml_v2_fair_compare" / "task17_shared_snapshot.csv"
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Task 17 snapshot not found at {csv_path}; run the Task 17 research "
            "experiment first or pass --snapshot."
        )
    panel = pd.read_csv(csv_path, low_memory=False)
    if "date" not in panel or "symbol" not in panel:
        raise ValueError("Task 17 snapshot must contain symbol and date columns")
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    if panel["date"].isna().any():
        raise ValueError("Task 17 snapshot contains invalid dates")
    panel = panel.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    meta = {
        "path": str(csv_path),
        "fingerprint": digest,
        "n_rows": int(len(panel)),
        "n_symbols": int(panel["symbol"].nunique()),
        "symbols": sorted(panel["symbol"].astype(str).unique().tolist()),
        "date_range": {"start": str(panel["date"].min().date()), "end": str(panel["date"].max().date())},
    }
    return panel, meta


def _as_array(values: Sequence[Any]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def compute_uncertainty(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
) -> dict[str, np.ndarray]:
    """Return model-agnostic uncertainty diagnostics for each row.

    ``probabilities`` is shaped ``(n_models, n_rows)`` or ``(n_rows,)``.
    ``disagreement`` is model standard deviation, ``margin`` is distance from
    0.5 of the mean probability, and ``entropy`` is binary predictive entropy.
    """
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError("probabilities must be a one- or two-dimensional array")
    if matrix.shape[0] == 0:
        return {key: np.array([], dtype=float) for key in ("probability", "disagreement", "margin", "entropy")}
    with np.errstate(invalid="ignore"):
        probability = np.nanmean(matrix, axis=0)
        disagreement = np.nanstd(matrix, axis=0)
    probability = np.clip(probability, 0.0, 1.0)
    entropy = -(probability * np.log(np.clip(probability, 1e-12, 1.0))
                + (1.0 - probability) * np.log(np.clip(1.0 - probability, 1e-12, 1.0)))
    return {
        "probability": probability,
        "disagreement": np.nan_to_num(disagreement, nan=np.inf),
        "margin": np.abs(probability - 0.5),
        "entropy": entropy,
    }


def conformal_quantiles(
    calibration_probability: Sequence[float],
    calibration_y: Sequence[float],
    alpha: float = 0.10,
) -> dict[str, Any]:
    """Compute class-conditional split-conformal nonconformity quantiles."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    p = _as_array(calibration_probability)
    y = _as_array(calibration_y)
    mask = np.isfinite(p) & np.isfinite(y)
    p, y = np.clip(p[mask], 0.0, 1.0), y[mask].astype(int)
    out: dict[str, Any] = {"alpha": float(alpha), "n": int(len(y)), "q_by_class": {}}
    for cls in (0, 1):
        scores = (1.0 - p[y == 1]) if cls == 1 else p[y == 0]
        if len(scores) == 0:
            out["q_by_class"][str(cls)] = None
            continue
        # Higher-quantile interpolation is conservative for finite samples.
        q = float(np.quantile(scores, min(1.0, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores)),
                              method="higher"))
        out["q_by_class"][str(cls)] = q
    return out


def apply_conformal(
    probability: Sequence[float],
    quantiles: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return conformal prediction-set size and an acceptance mask.

    A row is accepted only when exactly one class is in the prediction set.
    The returned decision is 1 for class one, 0 for class zero, and -1 for an
    abstention.
    """
    p = np.clip(_as_array(probability), 0.0, 1.0)
    q = quantiles.get("q_by_class", {})
    q0, q1 = q.get("0"), q.get("1")
    if q0 is None or q1 is None:
        return (
            np.full(len(p), 2, dtype=int),
            np.full(len(p), False, dtype=bool),
            np.full(len(p), -1, dtype=int),
        )
    in0 = p <= float(q0)
    in1 = (1.0 - p) <= float(q1)
    size = in0.astype(int) + in1.astype(int)
    accepted = size == 1
    decision = np.full(len(p), -1, dtype=int)
    decision[accepted & in1] = 1
    decision[accepted & in0 & ~in1] = 0
    return size, accepted, decision


def select_predictions(
    probability: Sequence[float],
    *,
    uncertainty: Mapping[str, Sequence[float]] | None = None,
    policy: str = "always_accept",
    conformal: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Apply one predeclared policy and return accepted mask and decisions."""
    if policy not in POLICY_RULES:
        raise ValueError(f"unknown policy {policy!r}; choose from {sorted(POLICY_RULES)}")
    p = np.clip(_as_array(probability), 0.0, 1.0)
    u = compute_uncertainty(p) if uncertainty is None else {
        key: _as_array(value) for key, value in uncertainty.items()
    }
    rule = POLICY_RULES[policy]
    if rule["kind"] == "always":
        accepted = np.isfinite(p)
        decision = (p >= 0.5).astype(int)
    elif rule["kind"] == "margin":
        accepted = np.isfinite(p) & (u["margin"] >= float(rule["threshold"]))
        decision = (p >= 0.5).astype(int)
    elif rule["kind"] == "disagreement":
        accepted = np.isfinite(p) & (u["disagreement"] <= float(rule["threshold"]))
        decision = (p >= 0.5).astype(int)
    elif rule["kind"] == "entropy":
        accepted = np.isfinite(p) & (u["entropy"] <= float(rule["threshold"]))
        decision = (p >= 0.5).astype(int)
    elif rule["kind"] == "conformal":
        if conformal is None:
            raise ValueError("conformal quantiles are required for conformal_90")
        _, accepted, decision = apply_conformal(p, conformal)
    else:  # pragma: no cover - POLICY_RULES is module-owned
        raise ValueError(f"unsupported policy kind {rule['kind']!r}")
    decision = np.where(accepted, decision, -1).astype(int)
    return {"accepted": accepted.astype(bool), "decision": decision}


def _ece(probability: np.ndarray, y: np.ndarray, bins: int = 10) -> float | None:
    if len(probability) == 0:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probability >= lo) & ((probability <= hi) if hi == 1 else (probability < hi))
        if mask.any():
            total += float(mask.mean()) * abs(float(probability[mask].mean()) - float(y[mask].mean()))
    return total


def selective_metrics(
    y_true: Sequence[float],
    probability: Sequence[float],
    accepted: Sequence[bool],
    *,
    target_return: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Measure coverage, selective risk, calibration, and optional return."""
    y = _as_array(y_true)
    p = np.clip(_as_array(probability), 0.0, 1.0)
    a = np.asarray(accepted, dtype=bool)
    valid = np.isfinite(y) & np.isfinite(p)
    n = int(valid.sum())
    a &= valid
    n_accepted = int(a.sum())
    out: dict[str, Any] = {
        "n": n, "n_accepted": n_accepted,
        "n_abstained": int(max(0, n - n_accepted)),
        "coverage": float(n_accepted / n) if n else None,
        "abstention_rate": float(1 - n_accepted / n) if n else None,
        "accuracy": None, "selective_risk": None, "roc_auc": None,
        "brier": None, "ece": None, "positive_rate": None,
    }
    if n_accepted == 0:
        return out
    ya, pa = y[a].astype(int), p[a]
    out["positive_rate"] = float(ya.mean())
    out["accuracy"] = float(np.mean((pa >= 0.5).astype(int) == ya))
    out["selective_risk"] = float(1.0 - out["accuracy"])
    out["brier"] = float(np.mean((pa - ya) ** 2))
    out["ece"] = _ece(pa, ya)
    if len(np.unique(ya)) > 1:
        from sklearn.metrics import roc_auc_score
        out["roc_auc"] = float(roc_auc_score(ya, pa))
    if target_return is not None:
        ret = _as_array(target_return)
        if len(ret) == len(p):
            signed = np.where(pa >= 0.5, 1.0, -1.0) * ret[a]
            signed = signed[np.isfinite(signed)]
            out["mean_signed_return"] = float(signed.mean()) if len(signed) else None
            out["win_rate_return"] = float(np.mean(signed > 0)) if len(signed) else None
        else:
            out["mean_signed_return"] = None
            out["win_rate_return"] = None
    return out


def null_baselines(
    y_true: Sequence[float],
    probability: Sequence[float],
    coverage: float,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Return fixed null controls at the candidate's observed coverage."""
    y, p = _as_array(y_true), np.clip(_as_array(probability), 0.0, 1.0)
    n = min(len(y), len(p))
    y, p = y[:n], p[:n]
    k = int(round(max(0.0, min(1.0, coverage)) * n))
    rng = np.random.default_rng(seed)
    random_accept = np.zeros(n, dtype=bool)
    if k:
        random_accept[rng.choice(n, size=min(k, n), replace=False)] = True
    shuffled = y.copy()
    rng.shuffle(shuffled)
    return {
        "always_accept": selective_metrics(y, p, np.ones(n, dtype=bool)),
        "always_abstain": selective_metrics(y, p, np.zeros(n, dtype=bool)),
        "random_accept": selective_metrics(y, p, random_accept),
        "shuffled_target": selective_metrics(shuffled, p, np.ones(n, dtype=bool)),
    }


def _build_selective_splits(
    panel: pd.DataFrame,
    *,
    horizon: int,
    n_folds: int = 4,
    min_train_dates: int = 260,
    final_holdout_fraction: float = DEFAULT_FINAL_HOLDOUT_FRACTION,
) -> dict[str, Any]:
    """Build global-date expanding folds and an untouched newest holdout.

    This is split infrastructure only.  Model fitting is delegated to the
    frozen per-symbol V1 implementation in :mod:`v2_v1_fair_compare`.
    """
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(panel["date"]).dt.normalize().unique()))
    holdout_n = max(1, int(np.ceil(len(dates) * final_holdout_fraction)))
    if len(dates) <= holdout_n + min_train_dates:
        raise ValueError("panel does not contain enough dates for a final holdout")
    holdout_dates = dates[-holdout_n:]
    pre = dates[:-holdout_n]
    gap = int(horizon)
    if len(pre) <= min_train_dates + gap:
        raise ValueError("panel does not contain enough pre-holdout dates")
    test_size = max(1, (len(pre) - min_train_dates) // n_folds)
    folds: list[dict[str, Any]] = []
    for fold in range(n_folds):
        start = min_train_dates + fold * test_size
        end = min(start + test_size, len(pre))
        if start >= len(pre) or end <= start:
            break
        candidate_train = pre[:start]
        train_dates = candidate_train[:-gap]
        purge_dates = candidate_train[-gap:]
        test_dates = pre[start:end]
        folds.append({
            "fold": fold,
            "train_dates": [d.date().isoformat() for d in train_dates],
            "purge_dates": [d.date().isoformat() for d in purge_dates],
            "embargo_dates": [d.date().isoformat() for d in purge_dates],
            "test_dates": [d.date().isoformat() for d in test_dates],
            "train_end": train_dates[-1].date().isoformat(),
            "test_start": test_dates[0].date().isoformat(),
            "test_end": test_dates[-1].date().isoformat(),
        })
    final_train = pre[:-gap]
    final_gap = pre[-gap:]
    return {
        "horizon": int(horizon),
        "embargo": gap,
        "final_holdout_fraction": float(final_holdout_fraction),
        "folds": folds,
        "final_holdout": {
            "isolated": True,
            "train_dates": [d.date().isoformat() for d in final_train],
            "purge_dates": [d.date().isoformat() for d in final_gap],
            "embargo_dates": [d.date().isoformat() for d in final_gap],
            "holdout_dates": [d.date().isoformat() for d in holdout_dates],
            "holdout_start": holdout_dates[0].date().isoformat(),
            "holdout_end": holdout_dates[-1].date().isoformat(),
        },
    }


def _v1_date_folds(split_records: Sequence[Mapping[str, Any]]) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    folds = []
    for item in split_records:
        test_values = item["test_dates"] if "test_dates" in item else item["holdout_dates"]
        folds.append((
            pd.DatetimeIndex(pd.to_datetime(item["train_dates"])),
            pd.DatetimeIndex(pd.to_datetime(test_values)),
        ))
    return folds


def _run_v1_predictions(
    panel: pd.DataFrame,
    splits: Mapping[str, Any],
    *,
    seeds: tuple[int, ...],
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Collect strictly OOS V1 component and ensemble probabilities."""
    folds = _v1_date_folds(splits["folds"])
    oof_result = fair.run_v1_reference(panel, folds, seeds=seeds, horizon=horizon)
    final_result = fair.run_v1_reference(
        panel,
        _v1_date_folds([splits["final_holdout"]]),
        seeds=seeds,
        horizon=horizon,
    )
    oof = oof_result["oof_panel"].copy()
    holdout = final_result["oof_panel"].copy()
    holdout["fold"] = -1
    components = [
        name for name in oof_result.get("usable_models", [])
        if f"prob_{name}" in oof.columns and f"prob_{name}" in holdout.columns
    ]
    if not components:
        raise ValueError("V1 reference returned no component probabilities")
    return oof, holdout, components


def _row_records(
    predictions: pd.DataFrame,
    components: Sequence[str],
    conformal: Mapping[str, Any],
) -> pd.DataFrame:
    rows = predictions.reset_index(drop=True)
    model_probabilities = rows[[f"prob_{name}" for name in components]].to_numpy(dtype=float).T
    u = compute_uncertainty(model_probabilities)
    base_probability = pd.to_numeric(rows["ensemble"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(rows["y"], errors="coerce").to_numpy(dtype=float)
    target_return = pd.to_numeric(
        rows.get("target_ret", pd.Series(np.nan, index=rows.index)), errors="coerce"
    ).to_numpy(dtype=float)
    records: list[pd.DataFrame] = []
    for policy in POLICY_RULES:
        selected = select_predictions(base_probability, uncertainty=u, policy=policy, conformal=conformal)
        frame = rows[["symbol", "date", "fold", "seed"]].copy()
        frame["policy"] = policy
        frame["probability"] = base_probability
        frame["disagreement"] = u["disagreement"]
        frame["entropy"] = u["entropy"]
        frame["margin"] = u["margin"]
        frame["y"] = y
        frame["target_return"] = target_return
        frame["accepted"] = selected["accepted"]
        frame["decision"] = selected["decision"]
        for name in components:
            frame[f"prob_{name}"] = rows[f"prob_{name}"].to_numpy(dtype=float)
        records.append(frame)
    return pd.concat(records, ignore_index=True)


def run_selective_experiment(
    panel: pd.DataFrame,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    horizon: int = DEFAULT_HORIZON,
    model_names: tuple[str, ...] = DEFAULT_MODEL_NAMES,
    final_holdout_fraction: float = DEFAULT_FINAL_HOLDOUT_FRACTION,
) -> dict[str, Any]:
    """Run fixed rules over strictly out-of-sample frozen-V1 probabilities."""
    requested_models = tuple(model_names or DEFAULT_MODEL_NAMES)
    if any(name not in V1_COMPONENT_MODELS for name in requested_models):
        raise ValueError(
            "Task 18 accepts only frozen V1 components; pooled candidates are not "
            f"valid inputs (allowed={V1_COMPONENT_MODELS})"
        )
    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work = work.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)
    if "target_up_20d" not in work:
        raise ValueError("panel missing target_up_20d")
    splits = _build_selective_splits(
        work, horizon=horizon, n_folds=4, min_train_dates=260,
        final_holdout_fraction=final_holdout_fraction,
    )
    oof_v1, holdout_v1, components = _run_v1_predictions(
        work, splits, seeds=seeds, horizon=horizon,
    )
    oof_frames: list[pd.DataFrame] = []
    holdout_frames: list[pd.DataFrame] = []
    # Calibrate conformal sets only from earlier OOS V1 folds.  The first
    # fold has no prior calibration sample and therefore abstains conformally.
    for fold in sorted(oof_v1["fold"].unique()):
        fold_rows = oof_v1[oof_v1["fold"] == fold].copy()
        calibration_rows = oof_v1[oof_v1["fold"] < fold]
        quantiles = conformal_quantiles(
            calibration_rows["ensemble"].to_numpy(dtype=float),
            calibration_rows["y"].to_numpy(dtype=float),
            alpha=0.10,
        )
        if len(fold_rows):
            oof_frames.append(_row_records(fold_rows, components, quantiles))
    final_quantiles = conformal_quantiles(
        oof_v1["ensemble"].to_numpy(dtype=float),
        oof_v1["y"].to_numpy(dtype=float),
        alpha=0.10,
    )
    if len(holdout_v1):
        holdout_frames.append(_row_records(holdout_v1, components, final_quantiles))

    oof = pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame()
    holdout = pd.concat(holdout_frames, ignore_index=True) if holdout_frames else pd.DataFrame()
    if oof.empty or holdout.empty:
        raise ValueError("selective experiment produced no valid OOF or holdout rows")

    def metrics_by(frame: pd.DataFrame, group_cols: Sequence[str]) -> dict[str, Any]:
        """Return ``group -> policy -> metrics`` for readable diagnostics."""
        out: dict[str, Any] = {}
        for keys, group in frame.groupby([*group_cols, "policy"], dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            parent = "|".join(str(v) for v in keys[:-1])
            out.setdefault(parent, {})[str(keys[-1])] = selective_metrics(
                group["y"].to_numpy(), group["probability"].to_numpy(),
                group["accepted"].to_numpy(), target_return=group["target_return"].to_numpy(),
            )
        return out

    def policy_metrics(frame: pd.DataFrame) -> dict[str, Any]:
        return {
            policy: selective_metrics(
                group["y"].to_numpy(), group["probability"].to_numpy(),
                group["accepted"].to_numpy(), target_return=group["target_return"].to_numpy(),
            )
            for policy, group in frame.groupby("policy", sort=True)
        }

    oof_policy = policy_metrics(oof)
    holdout_policy = policy_metrics(holdout)
    per_symbol = metrics_by(holdout, ["symbol"])
    per_fold = metrics_by(oof, ["fold"])
    regimes: dict[str, Any] = {}
    for column in ("regime_trend", "regime_vol", "regime_risk"):
        if column in work:
            regime_values = work[["symbol", "date", column]].copy()
            regime_values["date"] = pd.to_datetime(regime_values["date"]).dt.date.astype(str)
            joined = holdout.merge(regime_values, on=["symbol", "date"], how="left")
            regimes[column] = metrics_by(joined.dropna(subset=[column]), [column])

    seed_diag = {}
    for seed, group in oof.groupby("seed", sort=True):
        seed_diag[str(int(seed))] = policy_metrics(group)

    nulls = {}
    for policy, group in holdout.groupby("policy", sort=True):
        coverage = float(group["accepted"].mean()) if len(group) else 0.0
        nulls[policy] = null_baselines(group["y"], group["probability"], coverage, seed=17)

    candidates: dict[str, Any] = {}
    baseline = holdout_policy.get("always_accept", {})
    for policy, metrics in holdout_policy.items():
        if policy == "always_accept":
            continue
        coverage = metrics.get("coverage") or 0.0
        risk = metrics.get("selective_risk")
        base_risk = baseline.get("selective_risk")
        uplift = (baseline.get("accuracy") is not None and metrics.get("accuracy") is not None
                  and metrics["accuracy"] - baseline["accuracy"] or 0.0)
        risk_reduction = (base_risk is not None and risk is not None and base_risk - risk or 0.0)
        robust = all(
            (entry.get("coverage") or 0) >= MIN_SEED_COVERAGE and
            (entry.get("selective_risk") is not None) and
            (baseline.get("selective_risk") is None or entry["selective_risk"] <= baseline["selective_risk"])
            for entry in (seed_diag.get(str(seed), {}).get(policy, {}) for seed in seeds)
        )
        candidates[policy] = {
            "classification": "PROMISING_RESEARCH_ONLY" if coverage >= MIN_COVERAGE and uplift >= MIN_ACCURACY_UPLIFT and risk_reduction >= MIN_RISK_REDUCTION and robust else "NOT_SUPPORTED",
            "coverage": coverage, "accuracy_uplift": uplift,
            "risk_reduction": risk_reduction, "seed_robust": robust,
        }
    supported = [name for name, item in candidates.items() if item["classification"] == "PROMISING_RESEARCH_ONLY"]
    decision = "PROMISING_RESEARCH_ONLY" if supported else "KEEP FROZEN V1"
    return _finite({
        "schema_version": SCHEMA_VERSION,
        "module_version": MODULE_VERSION,
        "research_only": True,
        "production_v1_unchanged": True,
        "protocol": {
            "horizon": horizon, "seeds": list(seeds),
            "models": list(components),
            "model_source": "frozen V1 out-of-sample component probabilities",
            "ensemble_source": "fair.run_v1_reference",
            "final_holdout_fraction": final_holdout_fraction,
            "calibration_source": "earlier out-of-sample V1 folds only",
            "policies": POLICY_RULES,
            "no_final_holdout_tuning": True,
            "decision_gate": {
                "min_coverage": MIN_COVERAGE,
                "min_accuracy_uplift": MIN_ACCURACY_UPLIFT,
                "min_risk_reduction": MIN_RISK_REDUCTION,
                "min_seed_coverage": MIN_SEED_COVERAGE,
            },
            "folds": [{"fold": f["fold"], "train_end": f["train_end"],
                       "test_start": f["test_start"], "test_end": f["test_end"]}
                      for f in splits["folds"]],
            "final_holdout": splits["final_holdout"],
        },
        "oof": {
            "n_rows": int(len(oof)), "metrics": oof_policy,
            "per_fold": per_fold, "seed_diagnostics": seed_diag,
            "null_baselines": {policy: null_baselines(group["y"], group["probability"],
                                                       float(group["accepted"].mean()), seed=17)
                               for policy, group in oof.groupby("policy", sort=True)},
        },
        "final_holdout": {"n_rows": int(len(holdout)), "metrics": holdout_policy,
                          "null_baselines": nulls},
        "per_symbol": per_symbol,
        "regime_diagnostics": regimes,
        "candidates": candidates,
        "decision": decision,
        "generated_at": ver.utc_now_iso(),
    })


def run_from_snapshot(
    snapshot: str | Path | None = None,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    panel, metadata = load_shared_snapshot(snapshot)
    result = run_selective_experiment(panel, seeds=seeds)
    result["snapshot"] = metadata
    return result


def save_result(result: Mapping[str, Any], output_dir: str | Path = "ml_v2_selective_forecast") -> Path:
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parents[2] / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "task18_selective_forecast_result.json"
    _json_dump(result, path)
    # Keep the short Task-N naming convention used by the earlier research
    # artifacts as a compatibility alias.
    _json_dump(result, out_dir / "task18_result.json")
    return path


# Descriptive aliases make the small research API convenient to use from
# notebooks without creating a second implementation.
SELECTIVE_POLICIES = POLICY_RULES
compute_selective_metrics = selective_metrics
apply_abstention_rule = select_predictions
run_research = run_selective_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 18 selective forecasting research")
    parser.add_argument("--snapshot", default=None, help="Path to the Task 17 CSV snapshot")
    parser.add_argument("--output-dir", default="ml_v2_selective_forecast")
    parser.add_argument("--seeds", default="17,31,53")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    started = time.time()
    result = run_from_snapshot(args.snapshot, seeds=tuple(int(item) for item in args.seeds.split(",") if item))
    result["runtime_seconds"] = round(time.time() - started, 1)
    path = save_result(result, args.output_dir)
    logger.info("Task 18 result saved to %s; decision=%s", path, result["decision"])


if __name__ == "__main__":
    main()
