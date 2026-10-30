from functools import lru_cache
from pathlib import Path
import json
import pandas as pd
from app.core.config import get_settings

class DataService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.results = Path(self.settings.reports_results_dir)
        self._cache: dict[str, pd.DataFrame] = {}

    def csv(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            path = self.results / name
            if not path.exists():
                raise FileNotFoundError(name)
            self._cache[name] = pd.read_csv(path)
        return self._cache[name]

    @staticmethod
    def records(frame: pd.DataFrame) -> list[dict]:
        # pandas JSON conversion turns NumPy scalars into standard Python/JSON types.
        return json.loads(frame.to_json(orient="records", date_format="iso"))

    def dashboard(self) -> dict:
        evaluation = self.records(self.csv("week10_final_evaluation.csv"))[0]
        anomaly = self.records(self.csv("week11_anomaly_summary.csv"))[0]
        repl = self.records(self.csv("week12_replenishment_summary.csv"))[0]
        clusters = self.csv("week9_clustering_summary.csv")
        return {
            "evaluation": evaluation,
            "anomalies": anomaly,
            "replenishment": repl,
            "segments": self.records(clusters),
        }

    def performance(self) -> dict:
        return {
            "final_temporal": self.records(self.csv("week10_final_evaluation.csv"))[0],
            "xgboost": self.records(self.csv("week8_xgboost_results.csv")),
            "random_forest": self.records(self.csv("week7_random_forest_results.csv")),
            "decision_tree": self.records(self.csv("week6_decision_tree_results.csv")),
            "baseline": self.records(self.csv("week5_final_baseline.csv")),
        }

    def explainability(self, top_n: int = 15) -> dict:
        reg = self.csv("week8_xgb_regression_feature_importance.csv").sort_values("importance", ascending=False).head(top_n)
        cls = self.csv("week8_xgb_classification_feature_importance.csv").sort_values("importance", ascending=False).head(top_n)
        return {
            "regression_feature_importance": self.records(reg),
            "classification_feature_importance": self.records(cls),
            "shap_artifacts": ["week8_xgb_regression_shap.png", "week8_xgb_classification_shap.png"],
            "note": "SHAP summary images are genuine Week 8 artifacts. The API does not invent per-request SHAP values.",
        }

    def replenishment(self, priority: str | None = None, store_nbr: int | None = None, family: str | None = None, limit: int = 100, offset: int = 0) -> dict:
        frame = self.csv("week12_replenishment_priorities.csv")
        if priority:
            frame = frame[frame["priority"].str.upper() == priority.upper()]
        if store_nbr is not None:
            frame = frame[frame["store_nbr"] == store_nbr]
        if family:
            frame = frame[frame["family"].str.upper() == family.upper()]
        frame = frame.sort_values("replenishment_priority_score", ascending=False)
        return {"total": int(len(frame)), "items": self.records(frame.iloc[offset:offset + limit])}

    def anomalies(self, anomaly_type: str | None = None, store_nbr: int | None = None, family: str | None = None, limit: int = 100, offset: int = 0) -> dict:
        frame = self.csv("week11_detected_anomalies.csv")
        if anomaly_type:
            frame = frame[frame["anomaly_type"].str.lower() == anomaly_type.lower()]
        if store_nbr is not None:
            frame = frame[frame["store_nbr"] == store_nbr]
        if family:
            frame = frame[frame["family"].str.upper() == family.upper()]
        frame = frame.sort_values("date", ascending=False)
        return {"total": int(len(frame)), "items": self.records(frame.iloc[offset:offset + limit])}

    def segments(self, entity: str = "family") -> dict:
        if entity not in {"family", "store"}:
            raise ValueError("entity must be 'family' or 'store'")
        filename = "week9_family_clusters.csv" if entity == "family" else "week9_store_clusters.csv"
        frame = self.csv(filename)
        return {"entity": entity, "items": self.records(frame)}

@lru_cache
def get_data_service() -> DataService:
    return DataService()
