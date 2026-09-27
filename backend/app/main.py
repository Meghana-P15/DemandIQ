from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api.v1.router import api_router
from app.core.config import get_settings
from app.db.base import Base
from app.db.session import engine
from app.services.ml_service import get_ml_service
from app.middleware.rate_limit import InferenceRateLimitMiddleware

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.app_env.lower() == "production":
        weak_secret = settings.jwt_secret.lower()
        if len(settings.jwt_secret) < 32 or any(word in weak_secret for word in ("change", "replace", "example", "default")):
            raise RuntimeError("Production JWT_SECRET must be replaced with a strong secret (32+ characters).")
        if settings.database_url.startswith("sqlite"):
            raise RuntimeError("Production DemandIQ must use PostgreSQL; SQLite is development/test only.")
    Base.metadata.create_all(bind=engine)
    try:
        get_ml_service().load()
    except Exception:
        pass
    yield

app = FastAPI(title=settings.app_name, version="1.0.0", description="DemandIQ retail demand intelligence API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_list, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(InferenceRateLimitMiddleware, requests_per_minute=30)
app.include_router(api_router, prefix=settings.api_v1_prefix)

@app.get("/")
def root():
    return {"name": "DemandIQ API", "docs": "/docs", "health": f"{settings.api_v1_prefix}/health"}

@app.exception_handler(Exception)
async def unhandled_error(_: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": "Internal server error", "type": type(exc).__name__})
