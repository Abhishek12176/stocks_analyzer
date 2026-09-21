"""W1 tests — honest measurement of the 20-day forecast (accuracy, baselines,
median conventions, effective sample). All synthetic/offline (no network):
nothing here asserts alpha.

W1 = the old headline ("median full-OOF AUC 0.5703") could be read as an
accuracy claim: AUC is a ranking metric, the aggregate was an upper-median
over just 4 symbols, and the OOF labels overlap. These tests pin the honest
reporting layer added to `forecast_frame`, `benchmark` and `replay_snapshot`.
"""

import numpy as np
import pandas as pd
import pytest

from app.ml import benchmark as bm
from app.ml import pipeline
from app.ml import replay_snapshot as replay


def _ohlcv(n: int = 420, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    rets = rng.normal(0.001, 0.02, n)
    close = pd.Series(100 * np.cumprod(1 + rets), index=idx)
    open_ = close.shift(1).fillna(close)
    spread = close * np.abs(rng.normal(0, 1, n)) * 0.01 + close * 0.005
    df = pd.DataFrame({
        "Open": open_, "High": close + spread.abs(),
        "Low": close - spread.abs(), "Close": close,
    })
    df["Volume"] = rng.integers(100_000, 2_000_000, n).astype(float)
    return df


def _market(n: int = 420, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    close = pd.Series(2000 * np.cumprod(1 + rng.normal(0.0005, 0.008, n)), index=idx)
    return pd.DataFrame({"Close": close})


HORIZON = 5


class TestHonestMetricHelpers:
    def test_classification_accuracy_matches_manual_count(self):
        got = pipeline._classification_accuracy([0.6, 0.4, 0.51], [1, 1, 0])
        # 0.6->1 hit, 0.4->0 but label 1 miss, 0.51->1 but label 0 miss
        assert got == pytest.approx(1 / 3)

    def test_classification_accuracy_drops_nan_rows(self):
        got = pipeline._classification_accuracy([0.6, np.nan], [1, np.nan])
        assert got == pytest.approx(1.0)

    def test_classification_accuracy_empty_is_nan(self):
        assert np.isnan(pipeline._classification_accuracy([], []))
        assert np.isnan(pipeline._classification_accuracy([0.6], [0.6, 0.7]))

    def test_baseline_accuracies_are_base_rates(self):
        base = pipeline._baseline_accuracies([1, 1, 0, 0])
        assert base == {"always_up": 0.5, "always_down": 0.5}
        base = pipeline._baseline_accuracies([1, 1, 1, 0])
        assert base == {"always_up": 0.75, "always_down": 0.25}

    def test_baseline_accuracies_empty_is_nan(self):
        base = pipeline._baseline_accuracies([])
        assert np.isnan(base["always_up"]) and np.isnan(base["always_down"])

    def test_effective_independent_windows_round_up(self):
        assert pipeline._effective_independent_windows(88, 20) == 5
        assert pipeline._effective_independent_windows(440, 20) == 22
        assert pipeline._effective_independent_windows(0, 20) == 0

    def test_true_median_is_not_upper_median(self):
        # The legacy `sorted(x)[n//2]` picks the upper middle for even counts;
        # the honest median averages the two middles.
        vals = [0.4921, 0.5688, 0.5703, 0.6091]
        assert replay._true_median(vals) == pytest.approx((0.5688 + 0.5703) / 2)
        assert replay._upper_median(vals) == pytest.approx(0.5703)
        assert replay._true_median([0.5]) == pytest.approx(0.5)


@pytest.fixture(scope="module")
def forecast_result():
    feats = pipeline.build_feature_frame(_ohlcv(), market={"nifty50": _market()})
    return pipeline.forecast_frame(
        feats, symbol="TEST", horizon=HORIZON, fast=True, seed=3
    )


class TestHonestDiscriminationBlock:
    def test_block_explains_metric(self, forecast_result):
        disc = forecast_result["discrimination"]
        assert "accuracy" in disc["metric_definition"].lower()
        assert disc["full_oof_auc"] is not None
        assert disc["universe_note"] and disc["baseline_rule"]

    def test_accuracy_matches_label_counts(self, forecast_result):
        disc = forecast_result["discrimination"]
        for key in ("full_oof_accuracy_raw", "holdout_accuracy_calibrated",
                    "holdout_accuracy_raw"):
            v = disc[key]
            assert v is None or 0.0 <= v <= 1.0
        base = disc["holdout_baseline_accuracy"]
        assert base["always_up"] is not None
        assert base["always_up"] + base["always_down"] == pytest.approx(1.0)

    def test_edge_is_accuracy_minus_best_baseline(self, forecast_result):
        disc = forecast_result["discrimination"]
        acc = disc["holdout_accuracy_calibrated"]
        best = disc["holdout_best_baseline_accuracy"]
        edge = disc["holdout_edge_vs_best_baseline_accuracy"]
        if acc is not None and best is not None:
            assert edge == pytest.approx(acc - best)

    def test_effective_windows_are_shrunk(self, forecast_result):
        val = forecast_result["validation"]
        disc = forecast_result["discrimination"]
        hold_eff = val["holdout_rows_effective_independent"]
        assert hold_eff is not None and hold_eff < val["holdout_rows"]
        assert disc["effective_independent_windows"]["holdout"] == hold_eff
        assert "near-duplicates" in val["effective_sample_note"]
        assert "TRADE accuracy" in val["accuracy_note"]


class TestBenchmarkHonestAggregate:
    def test_benchmark_aggregate_accuracy_math_on_records(self):
        # NOTE: a full benchmark run is slow; the fast single-symbol path is
        # exercised by test_benchmark.py's own fixture. Here we only pin the
        # aggregate math on synthetic records so this file stays fast.
        records = [
            {
                "ok": True, "symbol": "A",
                "discrimination": {
                    "holdout_accuracy_calibrated": 0.52,
                    "holdout_baseline_accuracy": {"always_up": 0.55},
                    "holdout_edge_vs_best_baseline_accuracy": -0.03,
                },
                "holdout": {"ensemble": {"roc_auc": 0.53, "cum_return": 0.01,
                                         "profit_factor": 1.1, "max_dd": -0.05,
                                         "sharpe": 0.1, "win_rate": 0.5,
                                         "n_trades": 10,
                                         "no_leverage_ok": "yes"}},
            },
            {
                "ok": True, "symbol": "B",
                "discrimination": {
                    "holdout_accuracy_calibrated": 0.61,
                    "holdout_baseline_accuracy": {"always_up": 0.55},
                    "holdout_edge_vs_best_baseline_accuracy": 0.06,
                },
                "holdout": {"ensemble": {"roc_auc": 0.58, "cum_return": 0.05,
                                         "profit_factor": 1.4, "max_dd": -0.02,
                                         "sharpe": 0.4, "win_rate": 0.6,
                                         "n_trades": 12,
                                         "no_leverage_ok": "yes"}},
            },
        ]
        cfg = bm.BenchmarkConfig(
            symbols=["TCS", "RELIANCE"], seed=0, fast=True, period="5y",
            enrich=False, horizon=HORIZON, label="unit",
            run_model_comparison=False, run_ablation=False,
            run_failure_diagnostics=False, output_dir="ml_benchmark",
        )
        agg = bm._aggregate(records, cfg)
        m = agg["metrics"]
        assert m["holdout_classification_accuracy"]["n"] == 2
        assert m["holdout_classification_accuracy"]["median"] == pytest.approx(
            (0.52 + 0.61) / 2
        )
        assert m["holdout_always_up_baseline_accuracy"]["median"] == pytest.approx(0.55)
        assert m["holdout_edge_vs_best_baseline_accuracy"]["median"] is not None
        assert agg["counts"]["beats_best_baseline_accuracy"] == 1
        assert agg["counts"]["worse_than_best_baseline_accuracy"] == 1
        assert agg["counts"]["beats_best_baseline_accuracy"] + agg["counts"][
            "worse_than_best_baseline_accuracy"
        ] == 2
        assert agg["failure"]["counts"]["beats_best_baseline_accuracy"] == 1