import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.services.chat_service import process_chat

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[ChatTurn] | None = None


class ChatResponse(BaseModel):
    reply: str
    stocks: list[dict]
    totalFound: int
    intent: dict
    source: str
    generatedAt: str
    forecasts: dict | None = None
    contexts: dict | None = None
    llm_error: str | None = None


@router.get("/models")
async def list_models():
    """List available LLM models from the configured provider."""
    key = (settings.openai_api_key or "").strip()
    is_groq = key.startswith("gsk_") or "groq" in (settings.openai_base_url or "").lower()
    base_url = "https://api.groq.com/openai/v1" if is_groq else settings.openai_base_url.rstrip("/")
    if not key:
        return {"models": [], "error": "No API key configured"}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            resp = await client.get(f"{base_url}/models", headers={"Authorization": f"Bearer {key}"})
            return {"status": resp.status_code, "data": resp.json()}
    except Exception as exc:
        return {"error": str(exc)}


@router.post("", response_model=ChatResponse)
async def chat(body: ChatRequest):
    """AI Chat Assistant — answers stock prediction questions using the existing model.

    - Parses the user message for filters (price range, BUY/SELL/HOLD, top-N).
    - Loads predictions ONLY from the existing technical-indicator model.
    - Uses an LLM to format the answer when OPENAI_API_KEY is set, otherwise a
      deterministic template. Predictions are never fabricated.
    """
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    history = [t.model_dump() for t in body.history] if body.history else None
    return await process_chat(body.message.strip(), history)
