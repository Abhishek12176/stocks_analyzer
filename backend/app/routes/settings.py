"""Settings routes — API key status + update (local .env persistence).

Honest-by-construction:
- GET never returns secret values, only a "configured" boolean + masked suffix.
- POST persists new keys to backend/.env and applies them to the runtime
  settings immediately (no server restart required).
- No fabricated status: a key is "configured" only when a non-empty value exists.
"""

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import settings

router = APIRouter(prefix="/settings", tags=["settings"])

# Keys the user can configure from the UI -> env var name -> settings attr.
CONFIGURABLE_KEYS: list[dict] = [
    {
        "field": "newsdata_api_key",
        "label": "NEWSDATA_API_KEY",
        "env": "NEWSDATA_API_KEY",
        "display": "NewsData.io",
        "help": "Used for stock news headlines and sentiment.",
        "redacted_suffix": 4,
    },
    {
        "field": "openai_api_key",
        "label": "OPENAI_API_KEY",
        "env": "OPENAI_API_KEY",
        "display": "OpenAI / Groq",
        "help": "Used for the AI Chat Assistant (OpenAI-compatible endpoint).",
        "redacted_suffix": 4,
    },
]


def _resolve_env_path() -> Path:
    # backend/.env — same file pydantic-settings loads (Settings.env_file=".env").
    for base in (Path(__file__).resolve().parents[2], Path.cwd()):
        cand = base / ".env"
        if cand.is_file():
            return cand
    return Path(__file__).resolve().parents[2] / ".env"


def _redact(value: str) -> str | None:
    if not value:
        return None
    value = value.strip()
    if len(value) <= 8:
        return "*" * len(value)
    return value[:2] + "*" * (len(value) - 4 - 2) + value[-4:]


def _masked_status() -> list[dict]:
    out = []
    for meta in CONFIGURABLE_KEYS:
        raw = getattr(settings, meta["field"], "")
        out.append(
            {
                "field": meta["field"],
                "label": meta["label"],
                "display": meta["display"],
                "help": meta["help"],
                "configured": bool(raw),
                "masked": _redact(raw),
            }
        )
    return out


@router.get("/api-keys")
async def get_api_keys():
    """Report API-key status. Never returns secret values."""
    return {"keys": _masked_status()}


class UpdateApiKeysRequest(BaseModel):
    newsdata_api_key: str | None = Field(
        default=None, max_length=500, description="NewsData.io API key. Leave empty to keep current."
    )
    openai_api_key: str | None = Field(
        default=None, max_length=500, description="OpenAI-compatible (e.g. Groq) API key. Leave empty to keep current."
    )


def _upsert_env(env_path: Path, updates: dict[str, str]) -> None:
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8", errors="replace").splitlines()

    # Keep marker comments, drop any pre-existing bare assignments for these vars.
    kept = [ln for ln in lines if not any(ln.strip().startswith(k + "=") for k in updates)]

    for key, value in updates.items():
        kept.append(f"{key}={value}")

    env_path.write_text("\n".join(kept) + "\n", encoding="utf-8")


@router.post("/api-keys")
async def update_api_keys(req: UpdateApiKeysRequest):
    """Persist API keys to backend/.env and apply them to runtime settings."""
    env_path = _resolve_env_path()
    updates: dict[str, str] = {}

    if req.newsdata_api_key is not None:
        key = req.newsdata_api_key.strip()
        if not key:
            raise HTTPException(400, {"error": "API key cannot be empty", "code": "INVALID_INPUT"})
        updates["NEWSDATA_API_KEY"] = key
        os.environ["NEWSDATA_API_KEY"] = key
        settings.newsdata_api_key = key

    if req.openai_api_key is not None:
        key = req.openai_api_key.strip()
        if not key:
            raise HTTPException(400, {"error": "API key cannot be empty", "code": "INVALID_INPUT"})
        updates["OPENAI_API_KEY"] = key
        os.environ["OPENAI_API_KEY"] = key
        settings.openai_api_key = key

    if not updates:
        raise HTTPException(400, {"error": "Nothing to update", "code": "INVALID_INPUT"})

    try:
        _upsert_env(env_path, updates)
    except OSError as exc:
        raise HTTPException(500, {"error": f"Could not persist key: {exc}", "code": "WRITE_FAILED"})

    return {"saved": sorted(updates), "keys": _masked_status()}