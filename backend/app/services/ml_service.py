from __future__ import annotations
import json
import platform
import warnings
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from app.core.config import get_settings

class MLService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.bundle = None
        self.load_warnings: list[str] = []

    def load(self) -> None:
        if self.bundle is not None:
            return
        path = Path(self.settings.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model bundle not found: {path}")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.bundle = joblib.load(path)
        self.load_warnings = [str(item.message) for item in caught]

    @property
    def feature_columns(self) -> list[str]:
        self.load()
        return list(self.bundle["feature_columns"])

    def validate_features(self, features: dict) -> dict:
        missing = [col for col in self.feature_columns if col not in features]
        if missing:
            raise ValueError(f"Missing required feature columns: {', '.join(missing)}")
        return {col: features[col] for col in self.feature_columns}

    def predict(self, features: dict) -> dict:
        self.load()
        ordered = self.validate_features(features)
        frame = pd.DataFrame([ordered], columns=self.feature_columns)
        transformed = self.bundle["preprocessor"].transform(frame)
        demand = float(np.maximum(self.bundle["regressor"].predict(transformed)[0], 0))
        probability = float(self.bundle["classifier"].predict_proba(transformed)[0, 1])
        threshold = float(self.bundle["classification_threshold"])
        return {
            "predicted_demand": demand,
            "high_demand_probability": probability,
            "is_high_demand": bool(probability >= threshold),
            "probability_threshold": threshold,
            "demand_threshold": float(self.bundle["high_demand_threshold"]),
            "features": ordered,
        }


    def predict_frame(self, frame: pd.DataFrame) -> list[dict]:
        """Vectorized inference for a fully valid batch."""
        self.load()
        missing = [col for col in self.feature_columns if col not in frame.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {', '.join(missing)}")
        ordered = frame[self.feature_columns].copy()
        transformed = self.bundle["preprocessor"].transform(ordered)
        demand = np.maximum(self.bundle["regressor"].predict(transformed), 0)
        probabilities = self.bundle["classifier"].predict_proba(transformed)[:, 1]
        threshold = float(self.bundle["classification_threshold"])
        return [
            {
                "predicted_demand": float(pred),
                "high_demand_probability": float(prob),
                "is_high_demand": bool(prob >= threshold),
                "probability_threshold": threshold,
                "demand_threshold": float(self.bundle["high_demand_threshold"]),
            }
            for pred, prob in zip(demand, probabilities, strict=True)
        ]

    def info(self) -> dict:
        self.load()
        return {
            "model_bundle": Path(self.settings.model_path).name,
            "feature_count": len(self.feature_columns),
            "feature_columns": self.feature_columns,
            "high_demand_threshold": float(self.bundle["high_demand_threshold"]),
            "classification_threshold": float(self.bundle["classification_threshold"]),
            "regressor": type(self.bundle["regressor"]).__name__,
            "classifier": type(self.bundle["classifier"]).__name__,
            "runtime": {
                "python": platform.python_version(),
                "scikit_learn": sklearn.__version__,
                "xgboost": xgboost.__version__,
                "joblib": joblib.__version__,
            },
            "artifact_training_environment": {
                "python": "3.13.x",
                "scikit_learn": "1.9.0",
                "xgboost": "3.4.1",
                "numpy": "2.5.2",
                "pandas": "3.0.5",
                "joblib": "1.6.0",
            },
            "compatibility_warning_count": len(self.load_warnings),
            "compatibility_ok": (
                sklearn.__version__ == "1.9.0"
                and xgboost.__version__ == "3.4.1"
                and joblib.__version__ == "1.6.0"
            ),
        }

@lru_cache
def get_ml_service() -> MLService:
    return MLService()
