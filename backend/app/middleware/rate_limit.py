from __future__ import annotations
from collections import defaultdict, deque
from threading import Lock
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

class InferenceRateLimitMiddleware(BaseHTTPMiddleware):
    """Small single-process guard for public inference endpoints.

    Production multi-instance deployments should additionally use gateway/provider rate limiting.
    """
    def __init__(self, app, requests_per_minute: int = 30):
        super().__init__(app)
        self.limit = requests_per_minute
        self.window = 60.0
        self.events: dict[str, deque[float]] = defaultdict(deque)
        self.lock = Lock()

    async def dispatch(self, request: Request, call_next):
        watched = request.method == "POST" and request.url.path.startswith("/api/v1/predictions/")
        if watched:
            host = request.client.host if request.client else "unknown"
            now = time.monotonic()
            with self.lock:
                q = self.events[host]
                while q and now - q[0] > self.window:
                    q.popleft()
                if len(q) >= self.limit:
                    return JSONResponse(status_code=429, content={"detail": "Inference rate limit exceeded. Try again shortly."})
                q.append(now)
        return await call_next(request)
