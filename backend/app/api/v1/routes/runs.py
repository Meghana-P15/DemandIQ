from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.entities import PredictionRun, Prediction

router = APIRouter(prefix="/runs", tags=["runs"])

@router.get("")
def list_runs(limit: int = 25, db: Session = Depends(get_db)):
    limit = max(1, min(limit, 100))
    runs = db.scalars(select(PredictionRun).order_by(PredictionRun.created_at.desc()).limit(limit)).all()
    return {"items": [{"id": r.id, "mode": r.mode, "status": r.status, "row_count": r.row_count, "dataset_id": r.dataset_id, "created_at": r.created_at} for r in runs]}

@router.get("/{run_id}")
def get_run(run_id: int, limit: int = 200, db: Session = Depends(get_db)):
    run = db.get(PredictionRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Prediction run not found.")
    predictions = db.scalars(select(Prediction).where(Prediction.run_id == run_id).order_by(Prediction.id).limit(max(1, min(limit, 1000)))).all()
    return {
        "run": {"id": run.id, "mode": run.mode, "status": run.status, "row_count": run.row_count, "dataset_id": run.dataset_id, "created_at": run.created_at},
        "predictions": [{"id": p.id, "store_nbr": p.store_nbr, "family": p.family, "predicted_demand": p.predicted_demand, "high_demand_probability": p.high_demand_probability, "is_high_demand": p.is_high_demand, "created_at": p.created_at} for p in predictions],
    }
