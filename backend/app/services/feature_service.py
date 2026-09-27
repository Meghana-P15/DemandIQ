from functools import lru_cache
import pandas as pd
from app.core.config import get_settings
from app.services.ml_service import get_ml_service

class FeatureService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._history: pd.DataFrame | None = None

    def history(self) -> pd.DataFrame:
        if self._history is None:
            frame = pd.read_csv(self.settings.demo_history_path, parse_dates=["date"])
            self._history = frame.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
        return self._history

    def derive(self, store_nbr: int, family: str, target_date, overrides: dict | None = None) -> dict:
        frame = self.history()
        family = family.upper().strip()
        target = pd.Timestamp(target_date)
        subset = frame[(frame["store_nbr"] == store_nbr) & (frame["family"] == family) & (frame["date"] <= target)]
        if subset.empty:
            raise ValueError("No demo history found for that store/family/date combination. Choose a date in the demo range or submit all 23 features directly.")
        row = subset.iloc[-1].to_dict()
        row["year"] = int(target.year)
        row["month"] = int(target.month)
        row["day"] = int(target.day)
        row["day_of_week"] = int(target.dayofweek)
        row["week_of_year"] = int(target.isocalendar().week)
        row["is_weekend"] = int(target.dayofweek >= 5)
        if overrides:
            for key in ("onpromotion", "transactions", "dcoilwtico"):
                if overrides.get(key) is not None:
                    row[key] = overrides[key]
        columns = get_ml_service().feature_columns
        return {col: row[col] for col in columns}

    def presets(self, limit: int = 12) -> list[dict]:
        frame = self.history()
        latest = frame[frame["date"] == frame["date"].max()]
        keep = ["date", "store_nbr", "family", "onpromotion", "transactions", "dcoilwtico", "city", "state", "type", "cluster"]
        records = latest.sort_values(["store_nbr", "family"])[keep].head(limit).copy()
        records["date"] = records["date"].dt.strftime("%Y-%m-%d")
        return records.to_dict("records")

    def trend(self, store_nbr: int | None = None, family: str | None = None, days: int = 30) -> list[dict]:
        frame = self.history()
        if store_nbr is not None:
            frame = frame[frame["store_nbr"] == store_nbr]
        if family:
            frame = frame[frame["family"] == family.upper().strip()]
        daily = frame.groupby("date", as_index=False)["sales"].sum().sort_values("date").tail(days)
        daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
        return [{"date": r["date"], "sales": round(float(r["sales"]), 3)} for r in daily.to_dict("records")]

@lru_cache
def get_feature_service() -> FeatureService:
    return FeatureService()
