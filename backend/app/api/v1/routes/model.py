from fastapi import APIRouter
from app.services.ml_service import get_ml_service
from app.services.data_service import get_data_service

router = APIRouter(prefix="/model", tags=["model"])

@router.get("/info")
def model_info():
    return get_ml_service().info()

@router.get("/performance")
def model_performance():
    return get_data_service().performance()

@router.get("/explainability")
def model_explainability(top_n: int = 15):
    return get_data_service().explainability(top_n=max(1, min(top_n, 50)))
