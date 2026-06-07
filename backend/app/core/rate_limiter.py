import time
from collections import defaultdict
from app.utils.exceptions import RateLimitError


class RateLimiter:
    """Simple in-memory rate limiter using a sliding window."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    def check(self, key: str) -> None:
        now = time.time()
        window_start = now - self.window_seconds

        # Clean old entries
        self.requests[key] = [
            t for t in self.requests[key] if t > window_start
        ]

        if len(self.requests[key]) >= self.max_requests:
            raise RateLimitError()

        self.requests[key].append(now)


rate_limiter = RateLimiter()
