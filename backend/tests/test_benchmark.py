"""Task 11 tests — V1 forecast benchmark & failure analysis (measurement layer).

Structural + correctness assertions on synthetic data (no network): schema
stability, graceful failure reporting, aggregate math matching the per-symbol
records, per-bet/equity provenance separation, holdout isolation flags,
ablation not mutating the frame, determinism for a fixed input snapshot,
fingerprint sensitivity to input changes, and JSON round-trip. Nothing here
asserts alpha.
"""

import numpy as np
import pandas as pd
import pytest

from app.ml import benchmark as bm
from app.ml import fingerprint as fp
from app.ml import models as mo
from app.ml import pipeline

HORIZON = 5
N = 420


def _ohlcv(n: int = N, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
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


def _market(n: int = N, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    close = pd.Series(2000 * np.cumprod(1 + rng.normal(0.0005, 0.008, n)), index=idx)
    return pd.DataFrame({"Close": close})


def _symbol_seed(symbol: str) -> int:
    h = 0
    for ch in symbol:
        h = (h * 31 + ord(ch)) & 0x7FFFFFFF
    return h


def _make_features(symbol: str, perturb: int = 0) -> pd.DataFrame:
    seed = _symbol_seed(symbol) + perturb
    return pipeline.build_feature_frame(
        _ohlcv(N, seed), market={"nifty50": _market(N, seed % 97 + 3)}
    )


def make_provider(perturb: int = 0):
    """Deterministic offline input provider (replays across calls)."""
    UNAVAILABLE = {"NODATA": "no data for NODATA (test)", "BAD": "invalid NSE symbol"}

    def provider(symbol: str, period: str = "5y", enrich: bool = False) -> dict:
        clean = symbol.strip().upper()
        if clean in UNAVAILABLE:
            return {"ok": False, "clean": clean, "error": UNAVAILABLE[clean]}
        feats = _make_features(clean, perturb)
        return {
            "ok": True,
            "clean": clean,
            "features": feats,
            "fp_hex": fp.feature_frame_fingerprint(feats),
            "input_meta": {
                "scope": "test_input",
                "as_of": str(feats.index[-1].date()),
                "sources": {"stock": {"rows": int(len(feats))}},
            },
            "as_of": str(feats.index[-1].date()),
        }

    return provider


def _deep_off_cfg(symbols, seed: int = 0) -> bm.BenchmarkConfig:
    return bm.BenchmarkConfig(
        symbols=symbols, seed=seed, fast=True, period="5y", enrich=False,
        horizon=HORIZON, label="unit", run_model_comparison=False,
        run_ablation=False, run_failure_diagnostics=False,
        output_dir="ml_benchmark",
    )


@pytest.fixture(scope="module")
def benchmark_result():
    cfg = bm.BenchmarkConfig(
        symbols=["TCS", "RELIANCE", "NODATA"],
        seed=0, fast=True, period="5y", enrich=False, horizon=HORIZON,
        label="unit", run_model_comparison=True, run_ablation=True,
        run_failure_diagnostics=True, output_dir="ml_benchmark",
    )
    return bm.run_benchmark(cfg, input_provider=make_provider(0))


def _ok_records(result: dict) -> list[dict]:
    return [r for r in result["per_symbol"] if r.get("ok")]


def _ens(r: dict) -> dict:
    return (r.get("holdout") or {}).get("ensemble") or {}


class TestSchemaAndReporting:
    def test_top_level_schema(self, benchmark_result):
        result = benchmark_result
        assert result["benchmark_version"] == bm.BENCHMARK_VERSION
        assert result["schema_version"] == bm.SCHEMA_VERSION
        for key in ("run_metadata", "universe", "aggregate", "per_symbol",
                    "data_quality_issues", "warnings"):
            assert key in result
        assert result["run_metadata"]["horizon"] == HORIZON
        assert result["run_metadata"]["versions"]["benchmark"] == bm.BENCHMARK_VERSION

    def test_unavailable_symbol_reported_not_aborted(self, benchmark_result):
        result = benchmark_result
        assert result["universe"]["n_requested"] == 3
        assert result["universe"]["n_ok"] == 2
        assert result["universe"]["n_skipped"] == 1
        skipped = result["universe"]["skipped"]
        assert skipped and skipped[0]["symbol"] == "NODATA"
        assert "no data" in (skipped[0].get("reason") or "")
        rec = {r["symbol"]: r for r in result["per_symbol"]}["NODATA"]
        assert rec["ok"] is False
        assert any(
            it["symbol"] == "NODATA" and it["issue"] == "unavailable"
            for it in result["data_quality_issues"]
        )
        assert result["aggregate"]["counts"]["symbols_ok"] == 2

    def test_json_round_trip(self, benchmark_result, tmp_path):
        out = tmp_path / "bench"
        (out).mkdir(parents=True, exist_ok=True)
        paths = bm.save_benchmark(benchmark_result, output_dir=str(out))
        reloaded = bm.load_benchmark(paths["json"])
        assert reloaded == benchmark_result
        report = paths["text"].replace(".json", ".txt")
        text = (tmp_path / "bench" / report.rsplit("\\")[-1]).read_text(encoding="utf-8")
        assert "V1 FORECAST BENCHMARK" in text
        assert "AGGREGATE METRICS" in text


class TestAggregateMath:
    def test_aggregate_mean_matches_per_symbol(self, benchmark_result):
        result = benchmark_result
        ok = _ok_records(result)
        assert {r["symbol"] for r in ok} == {"TCS", "RELIANCE"}
        aucs = [_ens(r).get("roc_auc") for r in ok]
        assert all(a is not None for a in aucs)
        agg = result["aggregate"]["metrics"]["holdout_roc_auc"]
        assert agg["n"] == len(ok)
        assert pytest.approx(agg["mean"], abs=0.001) == float(np.mean(aucs))
        assert pytest.approx(agg["median"], abs=0.001) == float(np.median(aucs))
        assert agg["min"] == round(float(min(aucs)), 4)

    def test_count_checks_are_consistent(self, benchmark_result):
        result = benchmark_result
        counts = result["aggregate"]["counts"]
        ok = _ok_records(result)
        aucs = [_ens(r).get("roc_auc") for r in ok]
        assert counts["holdout_auc_gt_0p50"] == sum(1 for a in aucs if a > 0.5)
        assert counts["holdout_auc_gt_0p55"] == sum(1 for a in aucs if a > bm.WEAK_AUC_BOUND)
        assert counts["holdout_auc_le_0p50"] == sum(1 for a in aucs if a <= 0.5)
        assert counts["symbols_ok"] == len(ok)
        assert counts["symbols_failed"] == 1

    def test_per_model_aggregation_populated(self, benchmark_result):
        result = benchmark_result
        by_model = result["aggregate"]["per_model_holdout"]
        expected = set(bm.ENSEMBLE_COMPONENTS)
        assert set(by_model) == expected
        for m in expected:
            assert by_model[m]["roc_auc"]["n"] == 2

    def test_model_comparison_covers_all_installed(self, benchmark_result):
        result = benchmark_result
        mco = result["aggregate"]["model_comparison_oof"]
        assert set(mco) == set(mo.model_names())
        for m in mo.model_names():
            assert mco[m]["roc_auc"]["n"] == 2

    def test_baseline_sections_present(self, benchmark_result):
        result = benchmark_result
        baselines = result["aggregate"]["baselines"]
        for nm in ("always_up", "always_down", "random", "buy_hold", "shuffled_control"):
            assert nm in baselines


class TestPerSymbolRecord:
    def test_metric_groups_provenance_separated(self, benchmark_result):
        rec = _ok_records(benchmark_result)[0]
        ens = _ens(rec)
        assert ens["n_trades"] >= 0
        wr = ens["win_rate"]
        assert wr is None or 0.0 <= wr <= 1.0
        assert ens["profit_factor"] is None or ens["profit_factor"] >= 0.0
        dd = ens["max_dd"]
        assert dd is None or -1.0 <= dd <= 1.0  # a fraction, never a -99%
        assert "metric_groups" not in ens  # stripped from the extracted record
        assert set(rec["holdout"]["per_model"]) == set(bm.ENSEMBLE_COMPONENTS)
        assert {"always_up", "always_down", "random", "buy_hold",
                "shuffled_control"}.issubset(rec["holdout"]["baselines"])

    def test_holdout_isolation_flags(self, benchmark_result):
        rec = _ok_records(benchmark_result)[0]
        val = rec["validation"]
        assert val["final_holdout_never_used_for_fitting"] is True
        assert val["holdout_rows"] > 0
        assert rec["calibration"]["method"] in ("isotonic", "platt", "uncalibrated")
        assert "full OOF" in rec["ablation"]["evaluation_scope"]
        assert "not the final holdout" in rec["model_comparison"]["evaluation_scope"]

    def test_probability_collapse_signature_present(self, benchmark_result):
        rec = _ok_records(benchmark_result)[0]
        ps = rec["probability_support"]
        assert ps["n_distinct_calibrated_levels"] is not None
        assert ps["pct_within_0p05_0p50"] is not None
        assert isinstance(ps["collapse_flag"], bool)

    def test_regime_buckets_schema(self, benchmark_result):
        rec = _ok_records(benchmark_result)[0]
        reg = rec["failure"]["by_regime"]
        assert reg["buckets"]["regime_trend"]["buckets"]
        for label, stats in reg["buckets"]["regime_trend"]["buckets"].items():
            assert "n" in stats
            assert stats["roc_auc"] is None or 0.0 <= stats["roc_auc"] <= 1.0
        assert reg["buckets"]["regime_vol"]["buckets"]

    def test_feature_groups_present_listed(self, benchmark_result):
        from app.ml import ablation as abl
        rec = _ok_records(benchmark_result)[0]
        feats = _make_features(rec["symbol"])
        assert rec["feature_groups_present"] == sorted(
            abl.default_group_columns(list(feats.columns))
        )


class TestReproducibility:
    def test_deterministic_for_same_input(self):
        cfg = _deep_off_cfg(["TCS"])
        r1 = bm.run_symbol_benchmark("TCS", cfg, input_provider=make_provider(0))
        r2 = bm.run_symbol_benchmark("TCS", cfg, input_provider=make_provider(0))
        assert r1["ok"] and r2["ok"]

        def stable(r):
            return {k: v for k, v in r.items() if k not in ("generated_at", "elapsed_s")}

        s1, s2 = stable(r1), stable(r2)
        assert s1["data_fingerprint"] == s2["data_fingerprint"]
        assert s1["holdout"] == s2["holdout"]
        assert s1["calibration"] == s2["calibration"]
        assert s1["latest"] == s2["latest"]
        assert s1["data_fingerprint"] == fp.feature_frame_fingerprint(
            _make_features("TCS", 0)
        )

    def test_changed_input_changes_fingerprint_and_result(self):
        cfg = _deep_off_cfg(["TCS"])
        r0 = bm.run_symbol_benchmark("TCS", cfg, input_provider=make_provider(0))
        r1 = bm.run_symbol_benchmark("TCS", cfg, input_provider=make_provider(1))
        assert r0["data_fingerprint"] != r1["data_fingerprint"]
        assert r1["data_fingerprint"] == fp.feature_frame_fingerprint(
            _make_features("TCS", 1)
        )


class TestSafety:
    def test_ablation_does_not_mutate_frame(self):
        feats = _make_features("TCS", 0)
        cols_before = list(feats.columns)
        cfg = _deep_off_cfg(["TCS"])
        bm._ablation_section(feats, cfg)
        assert list(feats.columns) == cols_before

    def test_shorthand_history_reported_gracefully(self):
        def short_provider(symbol, period="5y", enrich=False):
            clean = symbol.strip().upper()
            from app.ml import dataset as ds
            feats = _make_features(clean, _symbol_seed(clean))
            return {
                "ok": True,
                "clean": clean,
                "features": feats.iloc[:60],
                "fp_hex": fp.feature_frame_fingerprint(feats.iloc[:60]),
                "input_meta": {"scope": "test_input", "as_of": "2022-03-01",
                               "sources": {"stock": {"rows": 60}}},
                "as_of": "2022-03-01",
            }

        cfg = _deep_off_cfg(["SHORT"])
        rec = bm.run_symbol_benchmark("SHORT", cfg, input_provider=short_provider)
        assert rec["ok"] is False
        assert "insufficient history" in (rec.get("error") or "")

    def test_summarize_numeric_only(self):
        assert bm._summarize([1, 2, 3, 4])["mean"] == 2.5
        assert bm._summarize([1, 2, 3, 4])["median"] == 2.5
        assert bm._summarize([])["n"] == 0
        assert bm._summarize([None, float("nan")])["n"] == 0
        assert bm._summarize(["yes", "no"])["n"] == 0