"""Offline contract tests for Task 18 selective research."""

import json

import numpy as np
import pandas as pd
import pytest

from app.ml import v2_selective_forecast as sf


def test_policy_rules_are_fixed_and_research_only():
    assert sf.MODULE_VERSION.startswith("task18-")
    assert sf.POLICY_RULES["margin_010"]["threshold"] == pytest.approx(0.10)
    assert sf.POLICY_RULES["conformal_90"]["alpha"] == pytest.approx(0.10)
    assert sf.POLICY_RULES["candidate_a_entropy"]["kind"] == "entropy"
    assert sf.POLICY_RULES["candidate_b_disagreement"]["kind"] == "disagreement"
    assert set(sf.V1_COMPONENT_MODELS) == {"voting", "logistic", "ridge", "rf", "xgboost"}


def test_compute_uncertainty_shapes_and_bounds():
    values = sf.compute_uncertainty([[0.4, 0.8], [0.6, 0.7]])
    assert values["probability"].tolist() == pytest.approx([0.5, 0.75])
    assert values["margin"].tolist() == pytest.approx([0.0, 0.25])
    assert values["disagreement"][0] == pytest.approx(0.1)
    assert np.all(values["entropy"] >= 0)


def test_select_predictions_margin_and_always():
    selected = sf.select_predictions([0.49, 0.51, 0.8], policy="margin_010")
    assert selected["accepted"].tolist() == [False, False, True]
    assert selected["decision"].tolist() == [-1, -1, 1]
    assert sf.select_predictions([0.49], policy="always_accept")["decision"].tolist() == [0]


def test_entropy_and_component_disagreement_are_fixed_policies():
    uncertainty = sf.compute_uncertainty(
        np.array([[0.5, 0.70], [0.5, 0.90], [0.5, 0.99]])
    )
    entropy = sf.select_predictions(
        uncertainty["probability"], uncertainty=uncertainty, policy="candidate_a_entropy"
    )
    disagreement = sf.select_predictions(
        uncertainty["probability"], uncertainty=uncertainty, policy="candidate_b_disagreement"
    )
    assert entropy["accepted"].tolist() == [False, True]
    assert disagreement["accepted"].tolist() == [True, False]


def test_conformal_is_class_conditional_and_abstains_when_set_is_ambiguous():
    q = sf.conformal_quantiles([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], alpha=0.1)
    size, accepted, decision = sf.apply_conformal([0.5, 0.99, 0.01], q)
    assert len(size) == len(accepted) == len(decision) == 3
    assert decision[0] == -1 or not accepted[0]
    assert np.all(np.isin(decision, [-1, 0, 1]))


def test_selective_metrics_reports_coverage_and_risk():
    metrics = sf.selective_metrics([0, 1, 1, 0], [0.1, 0.9, 0.6, 0.4],
                                   [True, True, False, False])
    assert metrics["n"] == 4
    assert metrics["n_accepted"] == 2
    assert metrics["coverage"] == pytest.approx(0.5)
    assert metrics["selective_risk"] == pytest.approx(0.0)


def test_selective_metrics_empty_acceptance_is_explicit():
    metrics = sf.selective_metrics([0, 1], [0.1, 0.9], [False, False])
    assert metrics["coverage"] == 0
    assert metrics["selective_risk"] is None
    assert metrics["n_abstained"] == 2


def test_null_baselines_are_deterministic():
    args = ([0, 1, 0, 1], [0.2, 0.8, 0.4, 0.6], 0.5)
    first = sf.null_baselines(*args, seed=17)
    second = sf.null_baselines(*args, seed=17)
    assert first == second
    assert set(first) == {"always_accept", "always_abstain", "random_accept", "shuffled_target"}


def test_load_snapshot_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        sf.load_shared_snapshot(tmp_path / "missing.csv")


def test_pooled_model_names_are_rejected():
    panel = pd.DataFrame({
        "symbol": ["A", "A"], "date": pd.date_range("2020-01-01", periods=2),
        "target_up_20d": [0, 1],
    })
    with pytest.raises(ValueError, match="frozen V1 components"):
        sf.run_selective_experiment(panel, model_names=("random_forest",))


def test_experiment_has_holdout_isolation_and_diagnostics(monkeypatch):
    rng = np.random.default_rng(17)
    dates = pd.date_range("2018-01-01", periods=720, freq="D")
    rows = []
    for symbol in ("A", "B"):
        x = rng.normal(size=len(dates))
        y = (x + rng.normal(scale=0.5, size=len(dates)) > 0).astype(float)
        for date, feature, target in zip(dates, x, y):
            rows.append({
                "symbol": symbol, "date": date, "feature": feature,
                "target_up_20d": target, "target_ret_20d": (target * 2 - 1) * 0.01,
                "regime_trend": "up" if feature > 0 else "down",
                "regime_vol": "high" if abs(feature) > 1 else "low",
            })
    panel = pd.DataFrame(rows)

    def fake_v1(panel, splits, *, seeds, horizon):
        components = list(sf.V1_COMPONENT_MODELS)

        def records(date_lists, fold):
            records = []
            dates = {pd.Timestamp(value).date().isoformat() for value in date_lists}
            for seed in seeds:
                for _, row in panel.iterrows():
                    date = pd.Timestamp(row["date"]).date().isoformat()
                    if date not in dates:
                        continue
                    base = float(0.35 + 0.30 * row["feature"])
                    record = {
                        "symbol": row["symbol"], "date": date, "fold": fold,
                        "seed": seed, "ensemble": np.clip(base, 0.01, 0.99),
                        "y": int(row["target_up_20d"]),
                        "target_ret": float(row["target_ret_20d"]),
                    }
                    for index, name in enumerate(components):
                        record[f"prob_{name}"] = np.clip(
                            base + (index - 2) * 0.01, 0.01, 0.99
                        )
                    records.append(record)
            return pd.DataFrame(records)

        oof = pd.concat(
            [records(fold["test_dates"], fold["fold"]) for fold in splits["folds"]],
            ignore_index=True,
        )
        holdout = records(splits["final_holdout"]["holdout_dates"], -1)
        return oof, holdout, components

    monkeypatch.setattr(sf, "_run_v1_predictions", fake_v1)
    result = sf.run_selective_experiment(panel, seeds=(17,), model_names=("logistic",))
    assert result["research_only"] is True
    assert result["production_v1_unchanged"] is True
    assert result["protocol"]["no_final_holdout_tuning"] is True
    assert result["final_holdout"]["n_rows"] > 0
    assert result["oof"]["n_rows"] > 0
    assert "per_symbol" in result and "regime_diagnostics" in result
    assert set(result["candidates"]) == set(sf.POLICY_RULES) - {"always_accept"}
    # Ensure the result is machine-serializable without numpy extensions.
    json.dumps(result)
