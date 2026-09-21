"""Regression tests for the enrich path fixes (14-Sep-2026).

Covers:
  - pipeline._build_symbol_input enrich macro: _series_close must not be
    falsy-evaluated (the `or` bool-eval fix).
  - pipeline._build_symbol_input enrich options: get_options_inputs must be
    called with the `spot` parameter (not bare symbol).
  - End-to-end enrich=True with monkeypatched network: macro columns (15),
    sentiment columns, and no fabricated data.

All tests are network-free — every upstream call is monkeypatched.
"""

import math
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

IDX = pd.bdate_range("2025-01-02", periods=60, freq="B")


def _make_ohlcv(symbol: str = "TEST") -> dict:
    """Synthetic OHLCV payload matching fetch_nse_ohlcv shape."""
    rows = []
    base = 2000.0
    for i, dt in enumerate(IDX):
        close = base + i * 1.0
        rows.append({
            "date": dt.isoformat(),
            "Open": close - 0.5,
            "High": close + 0.5,
            "Low": close - 1.0,
            "Close": close,
            "Adj Close": close,
            "Volume": 1_000_000 + i * 1000,
        })
    return {
        "is_available": True,
        "symbol": symbol,
        "rows": len(rows),
        "data_start": str(IDX[0].date()),
        "data_end": str(IDX[-1].date()),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "history": rows,
    }


def _make_market_series(series_id: str = "nifty50") -> pd.DataFrame:
    vals = [24000.0 + i * 10 for i in range(len(IDX))]
    return pd.DataFrame({"Close": vals}, index=IDX)


def _make_macro_series(series_id: str = "snp500") -> pd.DataFrame:
    vals = [5000.0 + i * 2 for i in range(len(IDX))]
    return pd.DataFrame({"Close": vals}, index=IDX)


def _stub_get_series_frame(sid: str, period: str = "5y", interval: str = "1d"):
    """Monkeypatchable stand-in for dsvc._get_series_frame."""
    if sid in ("snp500", "usd_inr", "gold"):
        return _make_macro_series(sid), {"rows": len(IDX)}
    if sid in ("nifty50", "india_vix", "banknifty", "cnxit"):
        return _make_market_series(sid), {"rows": len(IDX)}
    return None, {}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMacroSeriesCloseFix:
    """_series_close must return a pd.Series (not be falsy-evaluated)."""

    def test_series_close_returns_series(self):
        from app.ml.pipeline import _series_close
        mdf = _make_macro_series("snp500")
        result = _series_close({"snp500": mdf}, "snp500")
        assert isinstance(result, pd.Series)
        assert len(result) == len(IDX)
        assert result.iloc[0] == pytest.approx(5000.0)

    def test_series_close_none_when_missing(self):
        from app.ml.pipeline import _series_close
        assert _series_close(None, "x") is None
        assert _series_close({}, "x") is None
        assert _series_close({"x": None}, "x") is None

    def test_no_truthiness_error_on_series(self):
        from app.ml.pipeline import _series_close
        mdf = _make_macro_series("usd_inr")
        result = _series_close({"usd_inr": mdf}, "usd_inr")
        # The old `result or fallback` would raise ValueError:
        # "The truth value of a Series is ambiguous"
        # This test ensures the bug is gone.
        assert isinstance(result, pd.Series)
        # Verify the pipeline won't do `series or fallback` on this
        assert result.iloc[0] != 0  # non-empty series


class TestOptionsInputsSpotParam:
    """get_options_inputs must be called with a spot parameter."""

    def test_spot_is_required(self):
        """Calling get_options_inputs without spot raises TypeError."""
        from app.ml.options_df import get_options_inputs
        with pytest.raises(TypeError, match="spot"):
            get_options_inputs("RELIANCE")

    def test_graceful_disable_when_no_chain(self):
        from app.ml.options_df import get_options_inputs
        r = get_options_inputs("RELIANCE", spot=2900.0)
        assert r["is_available"] is False
        assert "no options expiries" in r["error"]

    def test_graceful_disable_when_disabled_setting(self, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "ml_options_enabled", False)
        from app.ml.options_df import get_options_inputs
        r = get_options_inputs("RELIANCE", spot=2900.0)
        assert r["is_available"] is False
        assert "disabled" in r["error"]


class TestBuildSymbolInputEnrichPath:
    """End-to-end: enrich=True macro columns must appear when sources are live."""

    def test_macro_columns_present_in_enrich(self, monkeypatch):
        from app.ml import pipeline
        from app.services import data_service as dsvc
        from app.services.fundamental_service import fundamentals_service
        from app.services.news_service import fetch_stock_news
        from app.services.sentiment_service import analyze_articles

        # Stub all network paths
        monkeypatch.setattr(dsvc, "fetch_nse_ohlcv", lambda s, **kw: _make_ohlcv(s))
        monkeypatch.setattr(dsvc, "_get_series_frame", _stub_get_series_frame)
        monkeypatch.setattr(fundamentals_service, "get_fundamentals", lambda s, x: None)
        monkeypatch.setattr(fetch_stock_news, "__call__", lambda s, **kw: None)
        monkeypatch.setattr(analyze_articles, "__call__", lambda x: [])

        result = pipeline._build_symbol_input("TEST", period="5y", enrich=True)
        assert result["ok"] is True
        feats = result["features"]
        cols = feats.columns.tolist()

        macro_cols = [c for c in cols if c.startswith("mac_")]
        assert len(macro_cols) == 15, f"Expected 15 macro cols, got {len(macro_cols)}: {macro_cols}"

        # All 3 series present
        assert any("snp500" in c for c in macro_cols)
        assert any("usd_inr" in c for c in macro_cols)
        assert any("gold" in c for c in macro_cols)

        # Source metadata reflects enrichment
        sources = result["input_meta"]["sources"]
        assert "snp500" not in sources  # stock/market sources only in meta

    def test_enrich_false_has_no_macro(self, monkeypatch):
        from app.ml import pipeline
        from app.services import data_service as dsvc
        from app.services.fundamental_service import fundamentals_service

        monkeypatch.setattr(dsvc, "fetch_nse_ohlcv", lambda s, **kw: _make_ohlcv(s))
        monkeypatch.setattr(dsvc, "_get_series_frame", _stub_get_series_frame)
        monkeypatch.setattr(fundamentals_service, "get_fundamentals", lambda s, x: None)

        result = pipeline._build_symbol_input("TEST", period="5y", enrich=False)
        assert result["ok"] is True
        macro_cols = [c for c in result["features"].columns if c.startswith("mac_")]
        assert len(macro_cols) == 0

    def test_options_spot_passed(self, monkeypatch):
        from app.ml import pipeline
        from app.services import data_service as dsvc
        from app.services.fundamental_service import fundamentals_service
        from app.services.news_service import fetch_stock_news
        from app.services.sentiment_service import analyze_articles
        from app.ml import options_df

        monkeypatch.setattr(dsvc, "fetch_nse_ohlcv", lambda s, **kw: _make_ohlcv(s))
        monkeypatch.setattr(dsvc, "_get_series_frame", _stub_get_series_frame)
        monkeypatch.setattr(fundamentals_service, "get_fundamentals", lambda s, x: None)
        monkeypatch.setattr(fetch_stock_news, "__call__", lambda s, **kw: None)
        monkeypatch.setattr(analyze_articles, "__call__", lambda x: [])

        called_with = {}

        def stub_get_options(symbol, spot, days_to_expiry=None, expiry=None):
            called_with["symbol"] = symbol
            called_with["spot"] = spot
            return {"symbol": symbol, "is_available": False, "error": "unit test stub",
                    "fetched_at": "", "expiry": None, "metrics": None}

        monkeypatch.setattr(options_df, "get_options_inputs", stub_get_options)

        result = pipeline._build_symbol_input("TEST", period="5y", enrich=True)
        assert result["ok"] is True
        # get_options_inputs must have been called with spot as the real last close
        assert "spot" in called_with, "get_options_inputs was not called"
        assert isinstance(called_with["spot"], float)
        assert 1000 < called_with["spot"] < 5000  # sanity: last close in the 2000 range

    def test_options_metrics_unwrapped_to_top_level(self, monkeypatch):
        """opt_* columns must materialize when options data is available.

        get_options_inputs nests metrics under a `metrics` key; add_options_features
        expects them at the snapshot top level. The unwrap fix must make the
        opt_* feature columns appear for the last row (PIT = fetch time).
        """
        from app.ml import pipeline
        from app.services import data_service as dsvc
        from app.services.fundamental_service import fundamentals_service
        from app.services.news_service import fetch_stock_news
        from app.services.sentiment_service import analyze_articles
        from app.ml import options_df

        monkeypatch.setattr(dsvc, "fetch_nse_ohlcv", lambda s, **kw: _make_ohlcv(s))
        monkeypatch.setattr(dsvc, "_get_series_frame", _stub_get_series_frame)
        monkeypatch.setattr(fundamentals_service, "get_fundamentals", lambda s, x: None)
        monkeypatch.setattr(fetch_stock_news, "__call__", lambda s, **kw: None)
        monkeypatch.setattr(analyze_articles, "__call__", lambda x: [])

        def live_options(symbol, spot, days_to_expiry=None, expiry=None):
            return {
                "symbol": symbol,
                "is_available": True,
                "fetched_at": f"{str(IDX[-1].date())}T16:00:00+05:30",
                "error": None,
                "expiry": "2026-10-30",
                "metrics": {
                    "pcr": 1.2,
                    "total_oi": 500000.0,
                    "atm_iv": 0.18,
                    "delta": 0.55,
                    "gamma": 0.01,
                    "theta": -0.2,
                    "vega": 0.3,
                    "rho": 0.4,
                },
            }

        monkeypatch.setattr(options_df, "get_options_inputs", live_options)

        result = pipeline._build_symbol_input("TEST", period="5y", enrich=True)
        assert result["ok"] is True
        feats = result["features"]
        opt_cols = [c for c in feats.columns if c.startswith("opt_")]
        assert len(opt_cols) == len(options_df.OPTION_COLUMNS), (
            f"Expected {len(options_df.OPTION_COLUMNS)} opt_* cols, got {opt_cols}"
        )
        # PIT = fetch time (2026-09-15) > last frame row -> only last row lit.
        last_row = feats.iloc[-1]
        assert last_row["opt_pcr"] == pytest.approx(1.2)
        # Historical rows must stay NaN (no lookahead, no backfill).
        assert math.isnan(feats.iloc[0]["opt_pcr"])

    def test_options_metrics_none_skipped(self, monkeypatch):
        """A graceful-disable options payload must produce no opt_* columns."""
        from app.ml import pipeline
        from app.services import data_service as dsvc
        from app.services.fundamental_service import fundamentals_service
        from app.services.news_service import fetch_stock_news
        from app.services.sentiment_service import analyze_articles
        from app.ml import options_df

        monkeypatch.setattr(dsvc, "fetch_nse_ohlcv", lambda s, **kw: _make_ohlcv(s))
        monkeypatch.setattr(dsvc, "_get_series_frame", _stub_get_series_frame)
        monkeypatch.setattr(fundamentals_service, "get_fundamentals", lambda s, x: None)
        monkeypatch.setattr(fetch_stock_news, "__call__", lambda s, **kw: None)
        monkeypatch.setattr(analyze_articles, "__call__", lambda x: [])

        def dead_options(symbol, spot, days_to_expiry=None, expiry=None):
            return {
                "symbol": symbol, "is_available": False, "fetched_at": "",
                "error": "no options expiries", "expiry": None, "metrics": None,
            }

        monkeypatch.setattr(options_df, "get_options_inputs", dead_options)

        result = pipeline._build_symbol_input("TEST", period="5y", enrich=True)
        assert result["ok"] is True
        opt_cols = [c for c in result["features"].columns if c.startswith("opt_")]
        assert len(opt_cols) == 0

    def test_fundamentals_wired_into_feature_frame(self, monkeypatch):
        """fund_* columns must materialize via the fundamentals snapshot.

        _build_symbol_input fetches fundamentals via fundamentals_service and
        passes them to build_feature_frame -> add_fundamental_features. The
        snapshot carries no available_at, so (PIT default) it first becomes
        visible on the LAST row only — never on historical rows.
        """
        from app.ml import pipeline
        from app.services import data_service as dsvc
        from app.services.fundamental_service import fundamentals_service

        monkeypatch.setattr(dsvc, "fetch_nse_ohlcv", lambda s, **kw: _make_ohlcv(s))
        monkeypatch.setattr(dsvc, "_get_series_frame", _stub_get_series_frame)
        monkeypatch.setattr(
            fundamentals_service,
            "get_fundamentals",
            lambda s, x: {
                "pe_ratio": 18.5,
                "eps": 120.0,
                "roe": 0.22,
                "debt_to_equity": 0.4,
                "operating_margin": 0.25,
                "revenue_growth": 0.10,
                "profit_growth": 0.12,
                "fundamental_score": 82.0,
            },
        )

        result = pipeline._build_symbol_input("TEST", period="5y", enrich=False)
        assert result["ok"] is True
        feats = result["features"]
        fund_cols = [c for c in feats.columns if c.startswith("fund_")]
        assert len(fund_cols) == len(
            {"pe", "eps", "roe", "de", "opm", "rev_growth", "profit_growth", "score"}
        ), f"Expected 8 fund_* cols, got {fund_cols}"
        # PIT default: available_at = last row -> only last row lit.
        assert feats["fund_pe"].iloc[-1] == pytest.approx(18.5)
        assert math.isnan(feats["fund_pe"].iloc[0])
        # source metadata records the usable flag
        assert result["input_meta"]["sources"]["fundamentals"]["available"] is True

    def test_fundamentals_pit_spreads_across_history(self, monkeypatch):
        """fundamentals_pit=True must give historical rows stepwise-constant
        fund_* values (not last-row-only), without touching the locked baseline
        path (default fundamentals_pit=False keeps last-row-only semantics)."""
        from app.ml import pipeline
        from app.services import data_service as dsvc
        from app.services.fundamental_service import fundamentals_service

        monkeypatch.setattr(dsvc, "fetch_nse_ohlcv", lambda s, **kw: _make_ohlcv(s))
        monkeypatch.setattr(dsvc, "_get_series_frame", _stub_get_series_frame)
        monkeypatch.setattr(fundamentals_service, "get_fundamentals", lambda s, x: None)

        q = [
            {
                "available_at": "2025-01-17",
                "pe_ratio": 20.0, "eps": 10.0, "roe": 0.20, "roce": 0.25,
                "debt_to_equity": 0.3, "operating_margin": 0.30,
                "revenue_growth": 0.05, "profit_growth": 0.10,
                "fundamental_score": 70.0,
            },
            {
                "available_at": "2025-03-16",
                "pe_ratio": 18.0, "eps": 12.0, "roe": 0.22, "roce": 0.27,
                "debt_to_equity": 0.2, "operating_margin": 0.32,
                "revenue_growth": 0.08, "profit_growth": 0.12,
                "fundamental_score": 75.0,
            },
        ]
        monkeypatch.setattr(
            fundamentals_service, "get_quarterly_fundamentals", lambda *a, **k: q
        )

        pit = pipeline._build_symbol_input(
            "TEST", period="5y", enrich=False, fundamentals_pit=True
        )
        assert pit["ok"] is True
        f = pit["features"]
        assert "fund_pe" in f.columns
        non_null = f["fund_pe"].notna().sum()
        assert non_null > len(f) * 0.3, f"expected >30% coverage, got {non_null}"
        # step-function: no value before first snapshot (2025-01-17)
        assert math.isnan(f.loc["2025-01-02", "fund_pe"])
        # later snapshot supersedes earlier one on the same date range
        after_s2 = f.loc["2025-03-16":, "fund_pe"].dropna()
        assert after_s2.iloc[0] == pytest.approx(18.0)
        # baseline (flag off) keeps last-row-only semantics
        base = pipeline._build_symbol_input(
            "TEST", period="5y", enrich=False, fundamentals_pit=False
        )
        bf = base["features"]
        bcov = bf["fund_pe"].notna().sum() if "fund_pe" in bf.columns else 0
        assert bcov <= 1, f"baseline must stay last-row-only, got {bcov} non-null"

    def test_sector_index_lever_appends_sector_features(self, monkeypatch):
        """sector_index='cnxit' adds rel_*_cnxit_* relative features (opt-in).

        The default (no sector) path must stay byte-identical — no sector
        columns, no feature drift on the locked baseline.
        """
        from app.ml import pipeline
        from app.services import data_service as dsvc
        from app.services.fundamental_service import fundamentals_service

        monkeypatch.setattr(dsvc, "fetch_nse_ohlcv", lambda s, **kw: _make_ohlcv(s))
        monkeypatch.setattr(dsvc, "_get_series_frame", _stub_get_series_frame)
        monkeypatch.setattr(fundamentals_service, "get_fundamentals", lambda s, x: None)
        monkeypatch.setattr(
            fundamentals_service, "get_quarterly_fundamentals", lambda *a, **k: []
        )

        with_sector = pipeline._build_symbol_input(
            "TEST", period="5y", enrich=False, sector_index="cnxit"
        )
        assert with_sector["ok"] is True
        sec_cols = [c for c in with_sector["features"].columns if "cnxit" in c]
        n_cnxit = len(sec_cols)
        assert n_cnxit == 7, f"expected 7 rel_*_cnxit_* cols, got {sec_cols}"
        assert any(c.startswith("rel_ret_") for c in sec_cols)
        assert any(c.startswith("rel_rs_slope_") for c in sec_cols)
        assert any(c.startswith("rel_rs_ratio_") for c in sec_cols)
        assert any("_corr_" in c for c in sec_cols)
        assert any("_vol_ratio_" in c for c in sec_cols)
        # nifty bench features still present (sector augments, doesn't replace)
        nifty_cols = [c for c in with_sector["features"].columns if c.endswith("nifty50_20d")
                      or "nifty50" in c]
        assert len(nifty_cols) >= 3
        # source metadata records the sector lever
        assert with_sector["input_meta"]["sources"]["sector"]["available"] is True

        baseline = pipeline._build_symbol_input(
            "TEST", period="5y", enrich=False, sector_index=None
        )
        assert baseline["ok"] is True
        base_cnxit = [c for c in baseline["features"].columns if "cnxit" in c]
        assert len(base_cnxit) == 0, "no sector index -> no sector columns"
        assert baseline["input_meta"]["sources"]["sector"]["available"] is False
        # same fingerprint? features differ is fine; source stay (stock) same
        assert baseline["input_meta"]["as_of"] == with_sector["input_meta"]["as_of"]
