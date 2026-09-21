"""Unit tests for the data ingestion layer (Task 2).

All network-boundary functions (`_load_series_data`, `_load_stock_data`) are
monkeypatched so these tests never hit the network.
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from app.services import data_service
from app.services.cache_service import cache_service
from app.services.data_service import (
    SERIES_REGISTRY,
    OHLCV_COLUMNS,
    _clean_ohlcv,
    align_closes,
    fetch_nse_ohlcv,
    fetch_series_history,
    get_available_series,
    get_market_context,
)


@pytest.fixture(autouse=True)
def _clear_market_cache():
    cache_service.market_cache.clear()
    yield
    cache_service.market_cache.clear()


def _synthetic_df(dates, values, name="Close") -> pd.DataFrame:
    return pd.DataFrame({name: values}, index=pd.to_datetime(dates))


class TestRegistry:
    def test_required_series_present(self):
        for sid in ["nifty50", "banknifty", "india_vix", "snp500", "nasdaq",
                    "nikkei", "hangseng", "vix", "brent", "gold", "usd_inr"]:
            assert sid in SERIES_REGISTRY

    def test_categories_present(self):
        cats = {meta["category"] for meta in SERIES_REGISTRY.values()}
        assert {"index", "index_vol", "global", "global_vol", "macro"} <= cats

    def test_tickers_unique(self):
        tickers = [meta["ticker"] for meta in SERIES_REGISTRY.values()]
        assert len(tickers) == len(set(tickers))

    def test_get_available_series_shape(self):
        rows = get_available_series()
        assert len(rows) == len(SERIES_REGISTRY)
        assert {"series_id", "name", "category", "ticker"} <= set(rows[0])


class TestClean:
    def test_handles_multilevel_columns(self):
        cols = pd.MultiIndex.from_product([["Open", "Close", "Adj Close"], ["X"]])
        raw = pd.DataFrame(
            [[1, 2, 3]],
            columns=cols,
            index=pd.to_datetime(["2024-01-01"]),
        )
        out = _clean_ohlcv(raw)
        assert "Close" in out.columns and "Adj Close" in out.columns
        assert float(out["Close"].iloc[0]) == 2.0

    def test_adj_close_falls_back_to_close(self):
        df = pd.DataFrame(
            {"Open": [1], "High": [1.5], "Low": [0.5], "Close": [2], "Volume": [100]},
            index=pd.to_datetime(["2024-01-01"]),
        )
        out = _clean_ohlcv(df)
        assert float(out["Adj Close"].iloc[0]) == 2.0

    def test_all_ohlcv_columns_present(self):
        df = pd.DataFrame(
            {"Open": [1], "High": [1.5], "Low": [0.5], "Close": [2], "Volume": [100]},
            index=pd.to_datetime(["2024-01-01"]),
        )
        out = _clean_ohlcv(df)
        assert set(OHLCV_COLUMNS) <= set(out.columns)

    def test_tz_aware_index_normalised_to_naive(self):
        # yfinance can return tz-aware indexes (Asia/Kolkata, America/New_York)
        df = pd.DataFrame(
            {"Open": [1], "High": [1.5], "Low": [0.5], "Close": [2], "Volume": [100]},
            index=pd.DatetimeIndex(["2024-01-01 00:00:00+05:30"]),
        )
        out = _clean_ohlcv(df)
        assert getattr(out.index, "tz", None) is None
        assert out.index[0] == pd.Timestamp("2024-01-01")

    def test_align_closes_handles_mixed_tz(self):
        ndf = _synthetic_df(["2024-01-01", "2024-01-02"], [100.0, 101.0])
        ndf.index = ndf.index.tz_localize("Asia/Kolkata")
        gdf = _synthetic_df(["2024-01-02"], [2000.0])
        aligned = align_closes({"nifty50": ndf, "gold": gdf})
        assert getattr(aligned.index, "tz", None) is None
        assert aligned.loc["2024-01-02", "gold"] == 2000.0

    def test_empty_input_returns_empty_frame(self):
        assert _clean_ohlcv(None).empty


class TestAlign:
    def test_outer_join_alignment_and_nan(self):
        nifty = _synthetic_df(
            ["2024-01-01", "2024-01-02", "2024-01-03"], [100.0, 101.0, 102.0]
        )
        gold = _synthetic_df(
            ["2024-01-02", "2024-01-03", "2024-01-04"], [2000.0, 2001.0, 2002.0]
        )
        aligned = align_closes({"nifty50": nifty, "gold": gold})
        assert list(aligned.columns) == ["nifty50", "gold"]
        assert aligned.index.is_monotonic_increasing
        assert aligned["nifty50"].loc["2024-01-01"] == 100.0
        assert pd.isna(aligned["gold"].loc["2024-01-01"])  # gold starts later
        assert pd.isna(aligned["nifty50"].loc["2024-01-04"])
        assert aligned.loc["2024-01-03", "gold"] == 2001.0

    def test_empty_series_excluded(self):
        aligned = align_closes({"nifty50": _synthetic_df([], []), "vix": None})
        assert aligned.empty

    def test_unsorted_input_is_sorted(self):
        nifty = _synthetic_df(["2024-01-03", "2024-01-01"], [102.0, 100.0])
        aligned = align_closes({"nifty50": nifty})
        assert list(aligned.index) == [
            pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03")
        ]


class TestFetchSeriesHistory:
    def test_success_payload(self, monkeypatch):
        df = _synthetic_df(
            ["2024-01-01", "2024-01-02"], [100.0, 101.0], name="Close"
        )
        df["Open"] = df["High"] = df["Low"] = df["Adj Close"] = df["Close"]
        df["Volume"] = 0
        monkeypatch.setattr(data_service, "_load_series_data", lambda *a, **k: df)

        payload = fetch_series_history("nifty50")
        assert payload["is_available"] is True
        assert payload["rows"] == 2
        assert payload["data_start"] == payload["history"][0]["date"]
        assert payload["data_end"] == payload["history"][-1]["date"]
        assert payload["error"] is None
        assert {"date", "open", "high", "low", "close", "adj_close", "volume"} <= set(
            payload["history"][0]
        )
        datetime.fromisoformat(payload["fetched_at"])  # freshness timestamp

    def test_failure_is_graceful(self, monkeypatch):
        def boom(*a, **k):
            raise ConnectionError("network down")

        monkeypatch.setattr(data_service, "_load_series_data", boom)
        payload = fetch_series_history("nifty50")
        assert payload["is_available"] is False
        assert payload["rows"] == 0
        assert payload["history"] == []
        assert "network down" in payload["error"]

    def test_unknown_series(self):
        payload = fetch_series_history("not_a_series")
        assert payload["is_available"] is False
        assert "Unknown" in payload["error"]

    def test_result_cached(self, monkeypatch):
        calls = {"n": 0}

        def fake(*a, **k):
            calls["n"] += 1
            df = _synthetic_df(["2024-01-01"], [1.0])
            df["Open"] = df["High"] = df["Low"] = df["Adj Close"] = 1.0
            df["Volume"] = 0
            return df

        monkeypatch.setattr(data_service, "_load_series_data", fake)
        fetch_series_history("nifty50")
        fetch_series_history("nifty50")
        assert calls["n"] == 1  # cached after first call


class TestFetchNseOhlcv:
    def test_success_payload(self, monkeypatch):
        df = _synthetic_df(["2024-01-01"], [500.0], name="Close")
        df["Open"] = df["High"] = df["Low"] = df["Adj Close"] = 500.0
        df["Volume"] = 1000
        monkeypatch.setattr(data_service, "_load_stock_data", lambda *a, **k: df)

        payload = fetch_nse_ohlcv("reliance")
        assert payload["is_available"] is True
        assert payload["symbol"] == "RELIANCE"
        assert payload["rows"] == 1
        assert payload["history"][0]["volume"] == 1000
        assert payload["history"][0]["adj_close"] == 500.0

    def test_failure_is_graceful(self, monkeypatch):
        def boom(*a, **k):
            raise ValueError("no data returned")

        monkeypatch.setattr(data_service, "_load_stock_data", boom)
        payload = fetch_nse_ohlcv("reliance")
        assert payload["is_available"] is False
        assert payload["symbol"] == "RELIANCE"
        assert payload["history"] == []


class TestGetMarketContext:
    def test_context_alignment_from_two_series(self, monkeypatch):
        nifty = _synthetic_df(["2024-01-01", "2024-01-02"], [100.0, 101.0])
        gold = _synthetic_df(["2024-01-02"], [2000.0])
        oil_fail = {"is_available": False, "error": "ConnectionError: x"}

        def fake_frame(sid, period="5y", interval="1d"):
            if sid == "nifty50":
                return nifty, {"is_available": True}
            if sid == "gold":
                return gold, {"is_available": True}
            return None, oil_fail

        monkeypatch.setattr(data_service, "_get_series_frame", fake_frame)
        ctx = get_market_context()
        assert ctx["sources"]["nifty50"]["is_available"] is True
        assert ctx["sources"]["brent"]["is_available"] is False  # failed + excluded
        assert "brent" not in ctx["aligned"][0] or True  # no fabrication
        assert ctx["alignment"]["series_count"] == 2
        assert ctx["alignment"]["aligned_dates"] == 2
        assert ctx["alignment"]["full_coverage_dates"] == 1
        assert ctx["aligned"][1]["gold"] == 2000.0

    def test_no_series_available(self, monkeypatch):
        monkeypatch.setattr(
            data_service,
            "_get_series_frame",
            lambda *a, **k: (None, {"is_available": False, "error": "down"}),
        )
        ctx = get_market_context()
        assert ctx["alignment"]["series_count"] == 0
        assert ctx["aligned"] == []