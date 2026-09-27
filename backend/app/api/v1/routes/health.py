from fastapi import APIRouter
from app.services.ml_service import get_ml_service

router = APIRouter(tags=["health"])

@router.get("/health")
def health():
    service = get_ml_service()
    try:
        info = service.info()
        return {"status": "ok", "model_loaded": True, "compatibility_warnings": info["compatibility_warning_count"]}
    except Exception as exc:
        return {"status": "degraded", "model_loaded": False, "detail": str(exc)}
