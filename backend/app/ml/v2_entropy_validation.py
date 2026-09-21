"""Task 19 validation of the frozen Task 18 entropy candidate.

This is a measurement-only layer.  It intentionally delegates candidate
construction to :mod:`v2_selective_forecast` and never changes its threshold,
the V1 models, targets, or production forecast path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from app.ml import v2_selective_forecast as sf

MODULE_VERSION = "task19-entropy-validation-v1"
SCHEMA_VERSION = "task19-entropy-validation-1"
POLICY = "candidate_a_entropy"
CLASSIFICATIONS = (
    "PROMISING FOR PRODUCTION INTEGRATION",
    "REJECTED",
    "EVIDENCE STILL INSUFFICIENT",
)


def _load(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Task 18 result must be a JSON object")
    return value


def _finite(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    return value


def paired_symbol_bootstrap(
    per_symbol: Mapping[str, Any],
    *,
    policy: str = POLICY,
    metric: str = "accuracy",
    n_bootstrap: int = 2000,
    seed: int = 17,
) -> dict[str, Any]:
    """Bootstrap selective-minus-all deltas by symbol (not by dependent rows)."""
    deltas = []
    for item in per_symbol.values():
        if not isinstance(item, Mapping):
            continue
        selective = item.get(policy, {})
        baseline = item.get("always_accept", {})
        a, b = selective.get(metric), baseline.get(metric)
        if a is not None and b is not None and np.isfinite([a, b]).all():
            deltas.append(float(a) - float(b))
    if not deltas:
        return {"n_symbols": 0, "metric": metric, "delta": None, "ci95": None}
    values = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(max(1, n_bootstrap), len(values)), replace=True).mean(axis=1)
    return {
        "n_symbols": int(len(values)),
        "metric": metric,
        "delta": float(values.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "seed": int(seed),
        "n_bootstrap": int(max(1, n_bootstrap)),
    }


def validate_task18_result(
    result: Mapping[str, Any],
    *,
    policy: str = POLICY,
    n_bootstrap: int = 2000,
    seed: int = 17,
) -> dict[str, Any]:
    """Validate the persisted Task 18 result without refitting or retuning."""
    protocol = result.get("protocol", {})
    declared = protocol.get("policies", {}).get(policy, {})
    if declared.get("kind") != "entropy" or declared.get("threshold") != 0.65:
        raise ValueError("Task 19 requires the exact Task 18 entropy threshold 0.65")
    holdout = result.get("final_holdout", {})
    metrics = holdout.get("metrics", {})
    baseline = metrics.get("always_accept", {})
    candidate = metrics.get(policy, {})
    coverage = candidate.get("coverage")
    uplift = (candidate.get("accuracy") - baseline.get("accuracy")
              if candidate.get("accuracy") is not None and baseline.get("accuracy") is not None else None)
    risk_reduction = (baseline.get("selective_risk") - candidate.get("selective_risk")
                      if baseline.get("selective_risk") is not None and candidate.get("selective_risk") is not None else None)
    gate = protocol.get("decision_gate", {})
    enough = coverage is not None and coverage >= gate.get("min_coverage", sf.MIN_COVERAGE)
    passes = enough and (uplift or 0) >= gate.get("min_accuracy_uplift", sf.MIN_ACCURACY_UPLIFT) and (
        risk_reduction or 0) >= gate.get("min_risk_reduction", sf.MIN_RISK_REDUCTION)
    symbol_ci = paired_symbol_bootstrap(result.get("per_symbol", {}), policy=policy,
                                         n_bootstrap=n_bootstrap, seed=seed)
    if not metrics or coverage is None:
        classification = "EVIDENCE STILL INSUFFICIENT"
    elif not passes:
        classification = "REJECTED"
    else:
        # Passing the Task 18 gate is not enough for production integration.
        # This validation must also establish robust cross-symbol and
        # chronological-fold evidence before using the promotion label.
        classification = "EVIDENCE STILL INSUFFICIENT"
    return _finite({
        "schema_version": SCHEMA_VERSION,
        "module_version": MODULE_VERSION,
        "research_only": True,
        "production_v1_unchanged": True,
        "policy": policy,
        "policy_definition": {"kind": "entropy", "threshold": 0.65, "source": "Task 18"},
        "all_vs_selective": {
            "all": baseline,
            "selective": candidate,
            "coverage": coverage,
            "n_retained": candidate.get("n_accepted"),
            "n_abstained": candidate.get("n_abstained"),
            "accuracy_uplift": uplift,
            "risk_reduction": risk_reduction,
        },
        "predictive": candidate,
        "calibration": {key: candidate.get(key) for key in ("brier", "ece")},
        "fold": result.get("oof", {}).get("per_fold", {}),
        "cross_symbol": result.get("per_symbol", {}),
        "seed": result.get("oof", {}).get("seed_diagnostics", {}),
        "economic": {key: candidate.get(key) for key in ("mean_signed_return", "win_rate_return")},
        "null_shuffled": result.get("final_holdout", {}).get("null_baselines", {}).get(policy, {}),
        "paired_symbol_bootstrap": symbol_ci,
        "classification": classification,
        "classification_options": list(CLASSIFICATIONS),
        "classification_basis": {
            "coverage_gate": enough,
            "accuracy_gate": (uplift or 0) >= gate.get("min_accuracy_uplift", sf.MIN_ACCURACY_UPLIFT),
            "risk_gate": (risk_reduction or 0) >= gate.get("min_risk_reduction", sf.MIN_RISK_REDUCTION),
            "no_holdout_tuning": protocol.get("no_final_holdout_tuning") is True,
        },
        "source_task18": {
            "module_version": result.get("module_version"),
            "schema_version": result.get("schema_version"),
            "snapshot": result.get("snapshot"),
        },
    })


def run_entropy_validation(
    result: Mapping[str, Any] | None = None,
    *,
    result_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate an existing Task 18 artifact; no network or model fitting."""
    if result is None:
        if result_path is None:
            result_path = Path(__file__).resolve().parents[2] / "ml_v2_selective_forecast" / "task18_selective_forecast_result.json"
        result = _load(result_path)
    return validate_task18_result(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", default=None, help="Task 18 result JSON")
    parser.add_argument("--output-dir", default="ml_v2_selective_forecast")
    args = parser.parse_args()
    result = run_entropy_validation(result_path=args.result)
    output = Path(args.output_dir) / "task19_entropy_validation_result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
