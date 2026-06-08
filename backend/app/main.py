import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.core.middleware import RateLimitMiddleware
from app.routes import stock, watchlist, compare, market, feedback, health, signals

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("equitylens")


# ---------------------------------------------------------------------------
# Startup / shutdown lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting EquityLens API – version 0.1.0")
    logger.info("Debug mode: %s", settings.debug)
    logger.info("CORS origins: %s", settings.cors_origins)
    logger.info("NewsData API key configured: %s", bool(settings.newsdata_api_key))

    from app.services.health_service import health_checker
    try:
        results = await health_checker.check_all()
        for r in results:
            if r.status == "ok":
                logger.info("Source healthy: %s (%.1fms)", r.name, r.latency_ms)
            else:
                logger.warning("Source unhealthy: %s (%.1fms) %s", r.name, r.latency_ms, r.error or "")
    except Exception as exc:
        logger.warning("Startup health check failed: %s", exc)

    yield

    logger.info("Shutting down EquityLens API")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
app.add_middleware(RateLimitMiddleware)


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "code": "INTERNAL_ERROR",
            "details": str(exc) if settings.debug else None,
        },
    )


# ---------------------------------------------------------------------------
# Health / root endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "status": "Backend Running",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "version": "0.1.0"}


# ---------------------------------------------------------------------------
# Routers (all under /api/v1)
# ---------------------------------------------------------------------------
app.include_router(stock.router, prefix="/api/v1")
app.include_router(watchlist.router, prefix="/api/v1")
app.include_router(compare.router, prefix="/api/v1")
app.include_router(market.router, prefix="/api/v1")
app.include_router(feedback.router, prefix="/api/v1")
app.include_router(health.router, prefix="/api/v1")
app.include_router(signals.router, prefix="/api/v1")


# ---------------------------------------------------------------------------
# Entry-point for direct execution (for local dev without uvicorn CLI)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", str(settings.port)))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=settings.debug,
    )
