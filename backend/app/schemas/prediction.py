from datetime import date
from typing import Any
from pydantic import BaseModel, Field, field_validator

class ForecastRequest(BaseModel):
    store_nbr: int = Field(ge=1)
    family: str = Field(min_length=1)
    date: date
    onpromotion: float | None = Field(default=None, ge=0)
    transactions: float | None = Field(default=None, ge=0)
    dcoilwtico: float | None = Field(default=None, ge=0)

    @field_validator("family")
    @classmethod
    def normalize_family(cls, value: str) -> str:
        return value.strip().upper()

class DirectPredictionRequest(BaseModel):
    features: dict[str, Any]

class PredictionResponse(BaseModel):
    predicted_demand: float
    high_demand_probability: float
    is_high_demand: bool
    probability_threshold: float
    demand_threshold: float
    feature_source: str
    features: dict[str, Any]

class BatchPredictionRow(PredictionResponse):
    row_number: int
