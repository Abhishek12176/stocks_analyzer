import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.rate_limiter import rate_limiter


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/api/"):
            client_ip = request.client.host if request.client else "unknown"
            rate_limiter.check(client_ip)

        start = time.time()
        response = await call_next(request)
        elapsed = time.time() - start

        response.headers["X-Response-Time"] = f"{elapsed:.3f}s"
        return response
