from fastapi import APIRouter
from app.services.signals_service import get_all_signals

router = APIRouter(prefix="/signals", tags=["signals"])


@router.get("/all")
async def get_signals():
    return get_all_signals()
