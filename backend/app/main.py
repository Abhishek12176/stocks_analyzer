from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config import settings
from app.core.middleware import RateLimitMiddleware
from app.routes import stock, watchlist, compare, market

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
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


# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "code": "INTERNAL_ERROR",
            "details": str(exc) if settings.debug else None,
        },
    )


# Health check
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "version": "0.1.0"}


# Include routers
app.include_router(stock.router, prefix="/api/v1")
app.include_router(watchlist.router, prefix="/api/v1")
app.include_router(compare.router, prefix="/api/v1")
app.include_router(market.router, prefix="/api/v1")
