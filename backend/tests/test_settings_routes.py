"""Settings route tests — API key status + update.

Covers:
- GET /settings/api-keys returns status booleans + masked values, never raw secrets.
- POST persists new keys to a temp .env and applies them to runtime settings.
- Validation: empty key -> 400, nothing-to-update -> 400.
"""

import pytest

from app.routes import settings as settings_route
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Point persistence at a temp .env and reset key attrs per test."""
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(settings_route, "_resolve_env_path", lambda: env_path)
    saved = (
        settings_route.settings.newsdata_api_key,
        settings_route.settings.openai_api_key,
    )
    settings_route.settings.newsdata_api_key = ""
    settings_route.settings.openai_api_key = ""
    yield env_path
    settings_route.settings.newsdata_api_key, settings_route.settings.openai_api_key = saved


def _get_keys():
    return client.get("/api/v1/settings/api-keys").json()["keys"]


def test_get_api_keys_reports_status_and_redacts():
    resp = client.get("/api/v1/settings/api-keys")
    assert resp.status_code == 200
    keys = resp.json()["keys"]
    fields = {k["field"] for k in keys}
    assert "newsdata_api_key" in fields
    assert "openai_api_key" in fields
    for k in keys:
        assert isinstance(k["configured"], bool)
        if k["configured"]:
            assert k["masked"] is not None
            assert len(k["masked"]) > 4
        else:
            assert k["masked"] is None


def test_get_never_leaks_secret():
    settings_route.settings.openai_api_key = "gsk-sk_test_secret_value_1234"
    resp = client.get("/api/v1/settings/api-keys")
    body = resp.text
    assert "sk_test_secret_value_1234" not in body
    entry = next(k for k in resp.json()["keys"] if k["field"] == "openai_api_key")
    assert entry["configured"] is True
    assert entry["masked"] is not None
    assert "1234" in entry["masked"]


def test_post_persists_key_to_env_and_runtime(isolated_env):
    resp = client.post(
        "/api/v1/settings/api-keys",
        json={"newsdata_api_key": "nd_test_key_abc"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["saved"] == ["NEWSDATA_API_KEY"]
    entry = next(k for k in body["keys"] if k["field"] == "newsdata_api_key")
    assert entry["configured"] is True

    content = isolated_env.read_text(encoding="utf-8")
    assert "NEWSDATA_API_KEY=nd_test_key_abc" in content
    assert settings_route.settings.newsdata_api_key == "nd_test_key_abc"


def test_post_replaces_existing_key_value(isolated_env):
    client.post("/api/v1/settings/api-keys", json={"newsdata_api_key": "old_value"})
    resp = client.post("/api/v1/settings/api-keys", json={"newsdata_api_key": "new_value"})
    assert resp.status_code == 200
    content = isolated_env.read_text(encoding="utf-8")
    assert "NEWSDATA_API_KEY=new_value" in content
    assert "NEWSDATA_API_KEY=old_value" not in content


def test_post_empty_key_rejected():
    resp = client.post("/api/v1/settings/api-keys", json={"newsdata_api_key": "   "})
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "INVALID_INPUT"


def test_post_nothing_to_update_rejected():
    resp = client.post("/api/v1/settings/api-keys", json={})
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "INVALID_INPUT"


def test_post_both_keys(isolated_env):
    resp = client.post(
        "/api/v1/settings/api-keys",
        json={"newsdata_api_key": "nd_abc", "openai_api_key": "gsk_xyz"},
    )
    assert resp.status_code == 200
    saved = sorted(resp.json()["saved"])
    assert saved == ["NEWSDATA_API_KEY", "OPENAI_API_KEY"]
    content = isolated_env.read_text(encoding="utf-8")
    assert "nd_abc" in content
    assert "gsk_xyz" in content