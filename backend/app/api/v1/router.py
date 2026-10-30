from fastapi import APIRouter
from app.api.v1.routes import analytics, auth, health, model, predictions, runs

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(model.router)
api_router.include_router(predictions.router)
api_router.include_router(analytics.router)
api_router.include_router(auth.router)
api_router.include_router(runs.router)
