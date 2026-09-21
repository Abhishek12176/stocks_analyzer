"""Reproducibility-layer tests (SHA-256 input-snapshot fingerprint).

Proves, on synthetic data (no network), that:
  - the same input snapshot always yields the same SHA-256 fingerprint;
  - changing one input value changes the fingerprint;
  - the same fingerprint + same seed reproduces identical probabilities,
    signals and backtest numbers through the full pipeline;
  - a different fingerprint is allowed to produce different results;
  - the forecast cache never serves a cached result whose input fingerprint
    differs from the current upstream snapshot.

Nothing here asserts accuracy/alpha — only reproducibility.
"""

import json

import numpy as np
import pandas as pd
import pytest

from app.ml import fingerprint as fp
from app.ml import pipeline


def _ohlcv(n: int = 420, seed: int = 11) -> pd.DataFrame:
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


def _market(n: int = 420, seed: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    close = pd.Series(2000 * np.cumprod(1 + rng.normal(0.0005, 0.008, n)), index=idx)
    return pd.DataFrame({"Close": close})


def _features(n: int = 420, seed: int = 11) -> pd.DataFrame:
    return pipeline.build_feature_frame(
        _ohlcv(n, seed), market={"nifty50": _market(n, 12)}
    )


def _records(n: int = 420, seed: int = 11) -> list[dict]:
    df = _ohlcv(n, seed)
    return [
        {
            "date": idx.isoformat(),
            "open": round(float(r["Open"]), 4),
            "high": round(float(r["High"]), 4),
            "low": round(float(r["Low"]), 4),
            "close": round(float(r["Close"]), 4),
            "volume": int(r["Volume"]),
        }
        for idx, r in df.iterrows()
    ]


HORIZON = 5
NOW = pd.Timestamp("2024-01-15")
SEED = 0


class TestFingerprintHelpers:
    def test_same_data_same_digest(self):
        df1 = _ohlcv()
        df2 = _ohlcv()
        assert fp.fingerprint_object("ohlcv", df1) == fp.fingerprint_object("ohlcv", df2)

    def test_one_value_changed_changes_digest(self):
        df1 = _ohlcv()
        df2 = _ohlcv()
        df2.loc[df2.index[-1], "Close"] = df2.loc[df2.index[-1], "Close"] + 0.01
        assert fp.fingerprint_object("ohlcv", df1) != fp.fingerprint_object("ohlcv", df2)

    def test_dict_key_order_is_irrelevant(self):
        a = fp.compose_fingerprint({"x": "h1", "y": "h2"})
        b = fp.compose_fingerprint({"y": "h2", "x": "h1"})
        assert a == b

    def test_component_name_is_inlined(self):
        assert fp.fingerprint_object("a", {"v": 1}) != fp.fingerprint_object("b", {"v": 1})
        parts_ab = fp.compose_fingerprint({"a": "h1", "b": "h2"})
        parts_aa = fp.compose_fingerprint({"a": "h1", "a": "h2"})
        assert parts_ab != parts_aa

    def test_nan_and_none_are_stable(self):
        s1 = fp.fingerprint_object("frame", pd.DataFrame(
            {"A": [1.0, float("nan")], "B": [None, "x"]}))
        s2 = fp.fingerprint_object("frame", pd.DataFrame(
            {"A": [1.0, float("nan")], "B": [None, "x"]}))
        assert s1 == s2

    def test_hex_length(self):
        d = fp.fingerprint_object("x", [1, 2, 3])
        assert isinstance(d, str) and len(d) == 64


class TestUpstreamFingerprint:
    def test_same_upstream_snapshot_same_digest(self):
        recs = _records()
        f1 = fp.fingerprint_object("ohlcv", recs)
        f2 = fp.fingerprint_object("ohlcv", _records())
        assert f1 == f2

    def test_one_upstream_value_changed_changes_digest(self):
        recs = _records()
        recs[-1]["close"] = recs[-1]["close"] + 0.01
        assert (
            fp.fingerprint_object("ohlcv", recs)
            != fp.fingerprint_object("ohlcv", _records())
        )

    def test_missing_fundamentals_is_distinct_state(self):
        assert (
            fp.fingerprint_object("fundamentals", None)
            != fp.fingerprint_object("fundamentals", {"pe": 21.0})
        )

    def test_market_series_change_changes_digest(self):
        f1 = fp.compose_fingerprint({
            "market:nifty50": fp.fingerprint_object("market:nifty50", _market()),
        })
        m2 = _market()
        m2.loc[m2.index[-1], "Close"] = m2.loc[m2.index[-1], "Close"] * 1.01
        f2 = fp.compose_fingerprint({
            "market:nifty50": fp.fingerprint_object("market:nifty50", m2),
        })
        assert f1 != f2


class TestForecastFrameFingerprint:
    def test_same_input_same_fp_and_identical_outputs(self):
        feats1 = _features()
        feats2 = _features()
        r1 = pipeline.forecast_frame(
            feats1, symbol="TEST", horizon=HORIZON, fast=True, seed=SEED, now=NOW
        )
        r2 = pipeline.forecast_frame(
            feats2, symbol="TEST", horizon=HORIZON, fast=True, seed=SEED, now=NOW
        )
        assert r1["is_available"] is True and r2["is_available"] is True

        # identical inputs -> identical fingerprint
        assert r1["data_fingerprint"] == r2["data_fingerprint"]
        assert r1["data_fingerprint"] == fp.feature_frame_fingerprint(feats1)

        # same fingerprint + same seed -> identical probabilities/signals/backtest
        assert r1["latest"]["probability"] == r2["latest"]["probability"]
        assert r1["latest"]["raw_probability"] == r2["latest"]["raw_probability"]
        assert r1["latest"]["signal"] == r2["latest"]["signal"]
        assert [b["model"] for b in r1["backtest"]] == [b["model"] for b in r2["backtest"]]
        for key in [b["model"] for b in r1["backtest"]]:
            b1 = next(r for r in r1["backtest"] if r["model"] == key)
            b2 = next(r for r in r2["backtest"] if r["model"] == key)
            assert b1["n_trades"] == b2["n_trades"]
            assert b1["cum_return"] == b2["cum_return"]

    def test_one_cell_changed_allowed_different_result(self):
        feats1 = _features()
        feats2 = _features()
        feats2.iloc[-1, feats2.columns.get_loc("Close")] = round(
            float(feats2.iloc[-1]["Close"]) + 0.5, 4
        )
        r1 = pipeline.forecast_frame(
            feats1, symbol="TEST", horizon=HORIZON, fast=True, seed=SEED, now=NOW
        )
        r2 = pipeline.forecast_frame(
            feats2, symbol="TEST", horizon=HORIZON, fast=True, seed=SEED, now=NOW
        )
        assert r1["is_available"] is True and r2["is_available"] is True
        assert r1["data_fingerprint"] != r2["data_fingerprint"]

    def test_meta_persisted_on_result(self):
        r = pipeline.forecast_frame(
            _features(), symbol="TEST", horizon=HORIZON, fast=True, seed=SEED, now=NOW
        )
        info = r["input_fingerprint"]
        assert info["algorithm"] == "sha256"
        assert info["data_fingerprint"] == r["data_fingerprint"]
        assert info["scope"] == "feature_frame"
        assert info["as_of"]
        assert "reconstruction" in info and "not supported" in info["reconstruction"]

    def test_passed_fingerprint_is_respected(self):
        r = pipeline.forecast_frame(
            _features(), symbol="TEST", horizon=HORIZON, fast=True, seed=SEED, now=NOW,
            data_fingerprint="x" * 64, input_meta={"scope": "upstream_input", "as_of": "2024-01-01"},
        )
        assert r["data_fingerprint"] == "x" * 64
        assert r["input_fingerprint"]["scope"] == "upstream_input"
        assert r["input_fingerprint"]["as_of"] == "2024-01-01"


class TestCacheDistinguishesFingerprints:
    def test_cache_requires_matching_fingerprint(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            pipeline, "_forecast_cache_path",
            lambda sym: tmp_path / f"{sym}.json",
        )
        result_a = {"is_available": True, "data_fingerprint": "AAAA", "x": 1}
        result_b = {"is_available": True, "data_fingerprint": "BBBB", "x": 2}

        pipeline._forecast_cache_put("TEST", "2024-01-01", result_a)
        # matching fingerprint -> cached result served
        hit = pipeline._forecast_cache_get("TEST", "2024-01-01", "AAAA")
        assert hit is not None and hit["data_fingerprint"] == "AAAA"
        # different fingerprint (different upstream snapshot) -> miss
        assert pipeline._forecast_cache_get("TEST", "2024-01-01", "BBBB") is None

        pipeline._forecast_cache_put("TEST", "2024-01-01", result_b)
        assert pipeline._forecast_cache_get("TEST", "2024-01-01", "BBBB") is not None
        assert pipeline._forecast_cache_get("TEST", "2024-01-01", "AAAA") is None
        # no-fingerprint lookup keeps legacy behavior
        fetched = pipeline._forecast_cache_get("TEST", "2024-01-01")
        assert fetched is not None and fetched["x"] == 2

    def test_legacy_record_without_fp_is_a_miss(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            pipeline, "_forecast_cache_path",
            lambda sym: tmp_path / f"{sym}.json",
        )
        legacy = {"is_available": True, "model": "old"}  # no data_fingerprint
        path = tmp_path / "TEST.json"
        path.write_text(json.dumps({
            "version": pipeline._FORECAST_CACHE_VERSION,
            "updated": "2024-01-01T00:00:00+00:00",
            "items": {"2024-01-01": legacy},
        }), encoding="utf-8")
        assert pipeline._forecast_cache_get("TEST", "2024-01-01", "Y") is None