from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

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
