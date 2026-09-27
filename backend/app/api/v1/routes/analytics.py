from fastapi import APIRouter, HTTPException
from app.services.data_service import get_data_service
from app.services.feature_service import get_feature_service

router = APIRouter(tags=["analytics"])

@router.get("/analytics/summary")
def summary():
    return get_data_service().dashboard()

@router.get("/analytics/trends")
def trends(store_nbr: int | None = None, family: str | None = None, days: int = 30):
    return {"items": get_feature_service().trend(store_nbr, family, max(7, min(days, 90)))}

@router.get("/metadata")
def metadata():
    service = get_feature_service()
    hist = service.history()
    return {
        "stores": sorted(int(x) for x in hist["store_nbr"].unique()),
        "families": sorted(str(x) for x in hist["family"].unique()),
        "min_date": hist["date"].min().strftime("%Y-%m-%d"),
        "max_date": hist["date"].max().strftime("%Y-%m-%d"),
        "presets": service.presets(),
    }

@router.get("/anomalies")
def anomalies(anomaly_type: str | None = None, store_nbr: int | None = None, family: str | None = None, limit: int = 100, offset: int = 0):
    return get_data_service().anomalies(anomaly_type, store_nbr, family, max(1, min(limit, 500)), max(0, offset))

@router.get("/replenishment")
def replenishment(priority: str | None = None, store_nbr: int | None = None, family: str | None = None, limit: int = 100, offset: int = 0):
    return get_data_service().replenishment(priority, store_nbr, family, max(1, min(limit, 500)), max(0, offset))

@router.get("/segments")
def segments(entity: str = "family"):
    try:
        return get_data_service().segments(entity)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
