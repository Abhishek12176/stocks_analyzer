"""Task 10 tests — forecast + ML-status route wiring (no network).

Exercises the FastAPI app through TestClient for the pieces that stay
network-free: invalid-symbol handling on the forecast route and the /ml/status
route with stubbed sources. Live-data behaviour is covered by the smoke run.
"""

import pandas as pd
import pytest

from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _pend_lifespan_health(client):
    # lifespan runs network health checks on startup; TestClient in this
    # context triggers the lifespan once, which we let resolve harmlessly.
    yield


class TestForecastRoute:
    def test_invalid_symbol_is_400(self, client):
        resp = client.get("/api/v1/stock/%24%24%24/forecast")
        assert resp.status_code == 400
        body = resp.json()
        assert body["detail"]["code"] == "INVALID_SYMBOL"

    def test_empty_symbol_is_400(self, client):
        resp = client.get("/api/v1/stock/%20/forecast")
        assert resp.status_code in (400, 404)


class TestMlStatusRoute:
    def test_status_endpoint_is_reachable(self, client, monkeypatch):
        from app.ml import pipeline as pl

        idx = pd.date_range(
            pd.Timestamp.now().normalize() - pd.Timedelta(days=1), periods=2
        )
        stub = lambda sid: (
            pd.DataFrame({"Close": [1.0, 2.0]}, index=idx),
            {"is_available": True},
        )
        monkeypatch.setattr(pl, "_default_source_fetcher", stub)

        resp = client.get("/api/v1/ml/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in {"ok", "degraded"}
        assert "sources" in body
        assert set(body["sources"][0].keys()) >= {
            "seriesId", "isAvailable", "status", "ageDays",
        }

    def test_status_degraded_with_unavailable_sources(self, client, monkeypatch):
        from app.ml import pipeline as pl

        stub = lambda sid: (None, {"is_available": False, "error": "boom"})
        monkeypatch.setattr(pl, "_default_source_fetcher", stub)

        resp = client.get("/api/v1/ml/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert "market_sources_stale" in body["alarms"]


class TestScorecardRoute:
    def test_scorecard_endpoint_returns_empty_report(self, client, monkeypatch):
        from app.ml import scorecard as sc

        monkeypatch.setattr(
            sc, "default_log_dir",
            lambda: __import__("pathlib").Path("/nonexistent-scorecard-dir"),
        )
        monkeypatch.setattr(sc, "build_live_close_lookup",
                            lambda: (lambda s: None))

        resp = client.get("/api/v1/ml/scorecard")
        assert resp.status_code == 200
        body = resp.json()
        assert body["schemaVersion"] == "scorecard-v1"
        assert body["nScored"] == 0
        assert body["isSufficientSample"] is False

    def test_scorecard_endpoint_supports_symbol_query(self, client, monkeypatch):
        from app.ml import scorecard as sc

        monkeypatch.setattr(
            sc, "default_log_dir",
            lambda: __import__("pathlib").Path("/nonexistent-scorecard-dir"),
        )
        monkeypatch.setattr(sc, "build_live_close_lookup",
                            lambda: (lambda s: None))

        resp = client.get("/api/v1/ml/scorecard?symbol=RELIANCE")
        assert resp.status_code == 200
        body = resp.json()
        assert body["symbol"] == "RELIANCE"
        assert body["nScored"] == 0