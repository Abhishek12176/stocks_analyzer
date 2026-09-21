"""Network-free contract tests for the isolated V2 target research module."""

import numpy as np
import pandas as pd
import pytest

from app.ml import v2_target_research as targets


def _panel(n: int = 70) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n)
    rows = []
    for symbol, scale in (("AAA", 1.0), ("BBB", 1.2)):
        close = 100.0 + scale * np.arange(n, dtype=float)
        rows.extend(
            {"symbol": symbol, "date": date, "Close": value, "feature": float(i)}
            for i, (date, value) in enumerate(zip(dates, close))
        )
    return pd.DataFrame(rows)


def test_fixed_boundaries_and_frozen_v1():
    forward = pd.Series([-0.0101, -0.01, 0.0, 0.01, 0.0101, np.nan])
    for actual, expected in (
        (targets.v1_binary_label(forward), [0.0, 0.0, 0.0, 1.0, 1.0, np.nan]),
        (targets.meaningful_return_label(forward), [0.0, 0.0, 0.0, 0.0, 1.0, np.nan]),
        (targets.three_class_label(forward), [-1.0, 0.0, 0.0, 0.0, 1.0, np.nan]),
    ):
        assert actual.iloc[:-1].tolist() == expected[:-1]
        assert pd.isna(actual.iloc[-1])


def test_forward_return_is_symbol_local_and_tail_is_nan():
    frame = _panel(8)
    out = targets.add_target_candidates(frame, horizon=2, volatility_window=2)
    for symbol in ("AAA", "BBB"):
        part = out[out.symbol == symbol].reset_index(drop=True)
        assert part["target_ret_2d"].iloc[0] == pytest.approx(part.Close.iloc[2] / part.Close.iloc[0] - 1)
        assert part["target_ret_2d"].iloc[-2:].isna().all()


def test_as_of_truncates_before_target_construction():
    frame = _panel(20)
    cutoff = frame.date.iloc[8]
    out = targets.add_target_candidates(frame, horizon=3, as_of=cutoff, volatility_window=3)
    assert out.date.max() == cutoff
    assert out.loc[out.date == cutoff, "target_ret_3d"].isna().all()


def test_volatility_uses_only_strictly_past_returns():
    close = pd.Series([100.0, 102.0, 101.0, 103.0, 104.0, 105.0, 106.0])
    changed = close.copy()
    changed.iloc[4:] = changed.iloc[4:] * 100.0
    first = targets.causal_past_volatility(close, window=3)
    second = targets.causal_past_volatility(changed, window=3)
    # The first usable estimate is at index 4 and only sees returns through 3.
    assert first.iloc[4] == pytest.approx(second.iloc[4])


def test_nan_and_zero_close_are_not_converted_to_labels():
    close = pd.Series([100.0, 0.0, 101.0, np.nan, 103.0])
    forward = targets.forward_return(close, horizon=1)
    assert forward.iloc[0] == pytest.approx(-1.0)
    assert pd.isna(forward.iloc[1])
    assert pd.isna(targets.v1_binary_label(pd.Series([np.nan])).iloc[0])


def test_class_counts_and_near_zero_buckets_are_explicit():
    frame = _panel(25)
    out = targets.add_target_candidates(frame, horizon=2, volatility_window=3)
    stats = targets.label_stats(out, ["target_up_2d", "target_three_class_2d"])
    assert stats["target_up_2d"]["undefined"] == 4
    assert sum(stats["target_three_class_2d"]["counts"].values()) == stats["target_three_class_2d"]["defined"]
    buckets = targets.near_zero_buckets(out["target_ret_2d"])
    assert sum(buckets.values()) == int(out["target_ret_2d"].notna().sum())


def test_research_comparison_is_reproducible_and_holdout_isolated():
    frame = _panel(80)
    kwargs = dict(
        feature_cols=["feature"],
        horizon=3,
        volatility_window=5,
        test_size=5,
        min_train=20,
        seed=17,
    )
    first = targets.run_target_research(frame, **kwargs)
    second = targets.run_target_research(frame, **kwargs)
    assert first["metadata"]["fingerprint"] == second["metadata"]["fingerprint"]
    assert first["metrics"] == second["metrics"]
    assert first["final_holdout"]["isolated"] is True
    assert all(item["test_end"] < first["final_holdout"]["holdout_start"] for item in first["folds"])
    assert set(first["metrics"]) == {
        "v1_binary",
        "meaningful_binary",
        "volatility_adjusted_binary",
        "three_class",
        "regression",
    }


def test_diagnostics_report_agreement_and_correlation():
    frame = _panel(25)
    out = targets.add_target_candidates(frame, horizon=2, volatility_window=3)
    diagnostics = targets.agreement_correlation_diagnostics(
        out, ["target_up_2d", "target_meaningful_up_2d", "target_regression_2d"]
    )
    assert diagnostics["agreement"]
    assert diagnostics["correlation"]
