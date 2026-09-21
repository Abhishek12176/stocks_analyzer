"""Network-free contract tests for Task 16 pooled model research."""

import numpy as np
import pandas as pd

from app.ml.v2_pooled_models import (
    binary_metrics,
    build_feature_matrices,
    build_global_panel_splits,
    evaluate_pooled_models,
)


def _panel(n=100):
    dates = pd.bdate_range("2024-01-01", periods=n)
    rows = []
    for symbol, offset in (("AAA", 0.0), ("BBB", 2.0), ("CCC", -1.0)):
        for i, date in enumerate(dates):
            rows.append({
                "symbol": symbol, "date": date, "Close": 100 + offset + i,
                "Volume": 1000 + i, "relative_rank": (i % 5) / 5,
                "target_up_20d": float((i + int(offset)) % 2),
            })
    return pd.DataFrame(rows)


def test_matrices_are_aligned_missing_safe_and_identity_is_explicit():
    panel = _panel()
    panel.loc[0, "Volume"] = np.nan
    matrices = build_feature_matrices(
        panel, baseline_features=["Close", "Volume"],
        relative_features=["relative_rank"], identity="one_hot",
    )
    assert list(matrices) == ["baseline", "baseline_plus_relative"]
    assert len(matrices["baseline"]) == len(panel)
    assert "symbol_AAA" in matrices["baseline"].columns
    assert "relative_rank" in matrices["baseline_plus_relative"]
    assert matrices["baseline"].index.equals(pd.RangeIndex(len(panel)))


def test_global_splits_purge_embargo_future_rows_and_holdout():
    panel = _panel()
    splits = build_global_panel_splits(
        panel, horizon=5, n_folds=3, test_size=10,
        min_train_dates=30, final_holdout_fraction=.2,
    )
    for fold in splits["folds"]:
        assert set(fold["train_indices"]).isdisjoint(fold["test_indices"])
        assert len(fold["purge_dates"]) >= 5
        assert max(fold["train_dates"]) < min(fold["test_dates"])
    final = splits["final_holdout"]
    assert final["isolated"] is True
    assert set(final["train_indices"]).isdisjoint(final["test_indices"])
    assert max(final["train_dates"]) < min(final["holdout_dates"])


def test_metrics_cover_probabilities_and_confusion():
    metrics = binary_metrics([0, 1, 1, 0], [0.1, 0.8, 0.6, 0.2])
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
    assert 0 <= metrics["brier"] <= 1
    assert metrics["probability_concentration"]["distinct_levels"] == 4


def test_models_are_deterministic_multi_symbol_and_report_per_symbol_holdout():
    kwargs = dict(
        baseline_features=["Close", "Volume"],
        relative_features=["relative_rank"],
        n_folds=2, test_size=10, min_train_dates=30,
        final_holdout_fraction=.2, models=("logistic", "ridge_probabilistic"),
        seeds=(17, 31),
    )
    first = evaluate_pooled_models(_panel(), **kwargs)
    second = evaluate_pooled_models(_panel(), **kwargs)
    assert first["metadata"]["fingerprint"] == second["metadata"]["fingerprint"]
    assert first["results"] == second["results"]
    result = first["results"]["logistic"]["baseline_plus_relative"]
    assert result["final_holdout"]["isolated"] is True
    assert set(result["final_holdout"]["per_symbol"]) == {"AAA", "BBB", "CCC"}
    assert len(result["seed_sensitivity"]) == 2
    assert "feature_comparison" in first
    assert "shuffled_target_logistic" in first["controls"]["final_holdout"]
