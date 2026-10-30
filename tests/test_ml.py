import pytest
from app.services.ml_service import get_ml_service

FIXTURE = {
    "store_nbr": 1, "family": "AUTOMOTIVE", "city": "Quito", "state": "Pichincha", "type": "D", "cluster": 13,
    "onpromotion": 0, "transactions": 1772.0, "dcoilwtico": 97.62, "year": 2013, "month": 1, "day": 29,
    "day_of_week": 1, "week_of_year": 5, "is_weekend": 0, "is_holiday": 0, "lag_1": 3.0, "lag_7": 1.0,
    "lag_14": 1.0, "lag_28": 0.0, "rolling_mean_7": 2.571428571428572, "rolling_mean_14": 2.142857142857143,
    "rolling_mean_28": 2.142857142857143,
}

def test_model_contract_has_23_features():
    service = get_ml_service()
    assert len(service.feature_columns) == 23
    assert service.feature_columns[0] == "store_nbr"
    assert service.feature_columns[-1] == "rolling_mean_28"

def test_fixed_prediction_fixture_is_stable():
    result = get_ml_service().predict(FIXTURE)
    assert result["predicted_demand"] == pytest.approx(3.6313741207, rel=1e-4, abs=1e-4)
    assert result["high_demand_probability"] == pytest.approx(2.8991084946e-06, rel=1e-3, abs=1e-8)
    assert result["is_high_demand"] is False

def test_missing_feature_is_rejected():
    broken = dict(FIXTURE)
    broken.pop("lag_28")
    with pytest.raises(ValueError, match="lag_28"):
        get_ml_service().predict(broken)
