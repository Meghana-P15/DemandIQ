from datetime import datetime, timezone
from sqlalchemy import String, Integer, Float, DateTime, ForeignKey, Text, JSON, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, default=0)
    source: Mapped[str] = mapped_column(String(80), default="upload")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

class PredictionRun(Base):
    __tablename__ = "prediction_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.id"), nullable=True)
    mode: Mapped[str] = mapped_column(String(40), default="single")
    status: Mapped[str] = mapped_column(String(40), default="completed")
    row_count: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("prediction_runs.id"), nullable=True, index=True)
    store_nbr: Mapped[int] = mapped_column(Integer, index=True)
    family: Mapped[str] = mapped_column(String(120), index=True)
    predicted_demand: Mapped[float] = mapped_column(Float)
    high_demand_probability: Mapped[float] = mapped_column(Float)
    is_high_demand: Mapped[bool] = mapped_column(Boolean)
    input_payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

class AnomalyRecord(Base):
    __tablename__ = "anomalies"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    store_nbr: Mapped[int] = mapped_column(Integer, index=True)
    family: Mapped[str] = mapped_column(String(120), index=True)
    anomaly_score: Mapped[float] = mapped_column(Float)
    anomaly_type: Mapped[str] = mapped_column(String(80))
    observed_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class SegmentResult(Base):
    __tablename__ = "segment_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(40), index=True)
    entity_key: Mapped[str] = mapped_column(String(120), index=True)
    cluster: Mapped[int] = mapped_column(Integer)
    metrics: Mapped[dict] = mapped_column(JSON)

class ReplenishmentRecommendation(Base):
    __tablename__ = "replenishment_recommendations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    store_nbr: Mapped[int] = mapped_column(Integer, index=True)
    family: Mapped[str] = mapped_column(String(120), index=True)
    priority_score: Mapped[float] = mapped_column(Float, index=True)
    priority: Mapped[str] = mapped_column(String(20), index=True)
    action: Mapped[str] = mapped_column(String(120))
    payload: Mapped[dict] = mapped_column(JSON)

class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    version: Mapped[str] = mapped_column(String(80))
    feature_schema: Mapped[dict] = mapped_column(JSON)
    library_versions: Mapped[dict] = mapped_column(JSON)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
