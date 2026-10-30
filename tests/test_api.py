import io
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app
from app.services.ml_service import get_ml_service
from tests.test_ml import FIXTURE

client = TestClient(app)

def test_health():
    with client:
        response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True

def test_summary_uses_real_artifacts():
    with client:
        response = client.get("/api/v1/analytics/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["evaluation"]["regression_R2"] > 0.9
    assert data["replenishment"]["critical_count"] == 429
    assert data["anomalies"]["anomaly_count"] == 98702

def test_demo_forecast():
    with client:
        response = client.post("/api/v1/predictions/demand", json={"store_nbr":1,"family":"AUTOMOTIVE","date":"2017-08-15"})
    assert response.status_code == 200
    body = response.json()
    assert body["feature_source"] == "demo_history_reference"
    assert body["predicted_demand"] >= 0
    assert body["run_id"] >= 1
    with client:
        stored = client.get(f"/api/v1/runs/{body['run_id']}")
    assert stored.status_code == 200
    assert stored.json()["predictions"][0]["family"] == "AUTOMOTIVE"

def test_direct_prediction():
    with client:
        response = client.post("/api/v1/predictions/direct", json={"features":FIXTURE})
    assert response.status_code == 200
    assert response.json()["predicted_demand"] > 0

def test_batch_rejects_missing_columns():
    raw = b"store_nbr,family\n1,AUTOMOTIVE\n"
    with client:
        response = client.post("/api/v1/predictions/batch", files={"file": ("bad.csv", raw, "text/csv")})
    assert response.status_code == 422
    assert "missing_columns" in response.json()["detail"]

def test_batch_scores_valid_row():
    frame = pd.DataFrame([FIXTURE], columns=get_ml_service().feature_columns)
    raw = frame.to_csv(index=False).encode()
    with client:
        response = client.post("/api/v1/predictions/batch", files={"file": ("ok.csv", raw, "text/csv")})
    assert response.status_code == 200
    body = response.json()
    assert body["successful_rows"] == 1
    assert body["error_rows"] == 0
    assert body["run_id"] >= 1
    assert body["dataset_id"] >= 1


def test_auth_register_login_me():
    import uuid
    email = f"test-{uuid.uuid4().hex[:10]}@example.com"
    with client:
        created = client.post("/api/v1/auth/register", json={"email": email, "password": "correct-horse-battery", "name": "Test User"})
        assert created.status_code == 200
        logged = client.post("/api/v1/auth/login", json={"email": email, "password": "correct-horse-battery"})
        assert logged.status_code == 200
        token = logged.json()["access_token"]
        me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me.status_code == 200
        assert me.json()["email"] == email
