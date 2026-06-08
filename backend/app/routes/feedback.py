import json
import os
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

FEEDBACK_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "feedback.json")


class FeedbackInput(BaseModel):
    name: str = ""
    email: str = ""
    message: str
    rating: int | None = None


@router.post("/feedback")
async def submit_feedback(body: FeedbackInput):
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)

    entry = {
        "id": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f"),
        "name": body.name.strip() or "Anonymous",
        "email": body.email.strip(),
        "message": body.message.strip(),
        "rating": body.rating,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "r") as f:
                data = json.load(f)
        else:
            data = []
    except (json.JSONDecodeError, FileNotFoundError):
        data = []

    data.append(entry)

    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=2)

    return {"status": "ok", "id": entry["id"]}
