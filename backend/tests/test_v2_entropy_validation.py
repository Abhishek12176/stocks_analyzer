import json

import pytest

from app.ml import v2_entropy_validation as ev


def _result(accuracy=0.55):
    return {
        "module_version": "task18-selective-v1",
        "schema_version": "task18-selective-results-1",
        "protocol": {
            "policies": {"candidate_a_entropy": {"kind": "entropy", "threshold": 0.65}},
            "decision_gate": {"min_coverage": 0.30, "min_accuracy_uplift": 0.02,
                              "min_risk_reduction": 0.02},
            "no_final_holdout_tuning": True,
        },
        "final_holdout": {
            "metrics": {
                "always_accept": {"accuracy": 0.50, "selective_risk": 0.50},
                "candidate_a_entropy": {"accuracy": accuracy, "selective_risk": 1 - accuracy,
                                        "coverage": 0.5, "n_accepted": 5, "n_abstained": 5,
                                        "brier": 0.2, "ece": 0.1},
            },
            "null_baselines": {"candidate_a_entropy": {"shuffled_target": {"accuracy": 0.5}}},
        },
        "per_symbol": {
            "A": {"always_accept": {"accuracy": 0.5},
                  "candidate_a_entropy": {"accuracy": accuracy}},
            "B": {"always_accept": {"accuracy": 0.5},
                  "candidate_a_entropy": {"accuracy": accuracy}},
        },
        "oof": {"policy_metrics": {}, "seed_diagnostics": {}},
        "snapshot": {"fingerprint": "abc"},
    }


def test_exact_entropy_definition_and_all_vs_selective():
    out = ev.validate_task18_result(_result())
    assert out["policy_definition"] == {"kind": "entropy", "threshold": 0.65, "source": "Task 18"}
    assert out["all_vs_selective"]["n_retained"] == 5
    assert out["all_vs_selective"]["n_abstained"] == 5
    assert out["classification"] == "EVIDENCE STILL INSUFFICIENT"
    assert set(out["classification_options"]) == set(ev.CLASSIFICATIONS)


def test_failed_gate_keeps_v1_and_empty_is_insufficient():
    assert ev.validate_task18_result(_result(0.505))["classification"] == "REJECTED"
    result = _result()
    result["final_holdout"]["metrics"] = {}
    assert ev.validate_task18_result(result)["classification"] == "EVIDENCE STILL INSUFFICIENT"


def test_bootstrap_is_paired_by_symbol_and_deterministic():
    symbols = _result()["per_symbol"]
    first = ev.paired_symbol_bootstrap(symbols, n_bootstrap=100, seed=4)
    assert first == ev.paired_symbol_bootstrap(symbols, n_bootstrap=100, seed=4)
    assert first["n_symbols"] == 2


def test_threshold_cannot_drift():
    result = _result()
    result["protocol"]["policies"]["candidate_a_entropy"]["threshold"] = 0.7
    with pytest.raises(ValueError, match="exact Task 18"):
        ev.validate_task18_result(result)


def test_json_serializable():
    json.dumps(ev.validate_task18_result(_result()))
