from fastapi import APIRouter
from app.services.health_service import health_checker

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/sources")
async def get_source_health():
    results = await health_checker.check_all()
    return {"sources": [r.dict() for r in results]}
