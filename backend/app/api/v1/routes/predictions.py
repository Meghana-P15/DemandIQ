import io
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session
from app.core.config import get_settings
from app.db.session import get_db
from app.models.entities import Dataset, PredictionRun, Prediction
from app.schemas.prediction import ForecastRequest, DirectPredictionRequest
from app.services.feature_service import get_feature_service
from app.services.ml_service import get_ml_service

router = APIRouter(prefix="/predictions", tags=["predictions"])

def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value

@router.post("/demand")
def forecast(request: ForecastRequest, db: Session = Depends(get_db)):
    try:
        features = get_feature_service().derive(request.store_nbr, request.family, request.date, {"onpromotion": request.onpromotion, "transactions": request.transactions, "dcoilwtico": request.dcoilwtico})
        result = get_ml_service().predict(features)
        result["feature_source"] = "demo_history_reference"
        run = PredictionRun(mode="single", status="completed", row_count=1)
        db.add(run); db.flush()
        db.add(Prediction(run_id=run.id, store_nbr=int(features["store_nbr"]), family=str(features["family"]), predicted_demand=result["predicted_demand"], high_demand_probability=result["high_demand_probability"], is_high_demand=result["is_high_demand"], input_payload=_json_safe(features)))
        db.commit()
        result["run_id"] = run.id
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@router.post("/direct")
def direct_forecast(request: DirectPredictionRequest, db: Session = Depends(get_db)):
    try:
        result = get_ml_service().predict(request.features)
        result["feature_source"] = "direct_23_feature_payload"
        features = result["features"]
        run = PredictionRun(mode="single", status="completed", row_count=1)
        db.add(run); db.flush()
        db.add(Prediction(run_id=run.id, store_nbr=int(features["store_nbr"]), family=str(features["family"]), predicted_demand=result["predicted_demand"], high_demand_probability=result["high_demand_probability"], is_high_demand=result["is_high_demand"], input_payload=_json_safe(features)))
        db.commit()
        result["run_id"] = run.id
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@router.post("/batch")
async def batch_forecast(file: UploadFile = File(...), db: Session = Depends(get_db)):
    settings = get_settings()
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Upload a CSV file.")
    raw = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"CSV exceeds {settings.max_upload_mb} MB upload limit.")
    try:
        frame = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {exc}") from exc
    if len(frame) > settings.max_upload_rows:
        raise HTTPException(status_code=413, detail=f"CSV has {len(frame)} rows; limit is {settings.max_upload_rows}.")
    service = get_ml_service()
    missing = [col for col in service.feature_columns if col not in frame.columns]
    if missing:
        raise HTTPException(status_code=422, detail={"message": "CSV is missing required feature columns", "missing_columns": missing})
    rows = []
    errors = []
    try:
        # Fast path: preprocess and score the entire valid batch once.
        batch_results = service.predict_frame(frame)
        rows = [{"row_number": int(idx) + 2, **result} for idx, result in enumerate(batch_results)]
    except Exception:
        # Diagnostic fallback: isolate malformed rows so valid rows can still be returned.
        for idx, record in frame.iterrows():
            try:
                result = service.predict(record.to_dict())
                result.pop("features", None)
                rows.append({"row_number": int(idx) + 2, **result})
            except Exception as exc:
                errors.append({"row_number": int(idx) + 2, "error": str(exc)})
    dataset = Dataset(name=file.filename, row_count=int(len(frame)), source="csv_upload")
    db.add(dataset); db.flush()
    run = PredictionRun(dataset_id=dataset.id, mode="batch", status="completed_with_errors" if errors else "completed", row_count=len(rows))
    db.add(run); db.flush()
    for item in rows:
        source_index = item["row_number"] - 2
        source = frame.iloc[source_index].to_dict()
        db.add(Prediction(run_id=run.id, store_nbr=int(source["store_nbr"]), family=str(source["family"]), predicted_demand=item["predicted_demand"], high_demand_probability=item["high_demand_probability"], is_high_demand=item["is_high_demand"], input_payload=_json_safe({col: source[col] for col in service.feature_columns})))
    db.commit()
    return {"run_id": run.id, "dataset_id": dataset.id, "filename": file.filename, "input_rows": int(len(frame)), "successful_rows": len(rows), "error_rows": len(errors), "predictions": rows, "errors": errors}
