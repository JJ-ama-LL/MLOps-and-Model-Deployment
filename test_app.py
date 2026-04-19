from app import app
from fastapi.testclient import TestClient
import pytest

@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c

VALID_INPUT = {
    "trip_distance": 3.5,
    "fare_amount": 14.5,
    "pickup_hour": 14,
    "passenger_count": 1,
    "trip_duration_minutes": 12.0,
    "total_amount": 18.8
}

def test_predict_valid(client):
    response = client.post("/predict", json=VALID_INPUT)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_tip_amount" in data
    assert "model_version" in data
    assert "prediction_id" in data
    assert isinstance(data["predicted_tip_amount"], float)

def test_predict_batch(client):
    response = client.post("/predict/batch", json={"records": [VALID_INPUT] * 3})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    assert len(data["predictions"]) == 3
    assert "processing_time_ms" in data

def test_predict_missing_field(client):
    response = client.post("/predict", json={"trip_distance": 3.5})
    assert response.status_code == 422

def test_predict_wrong_type(client):
    invalid = VALID_INPUT.copy()
    invalid["pickup_hour"] = "afternoon"
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422

def test_predict_out_of_range(client):
    invalid = VALID_INPUT.copy()
    invalid["pickup_hour"] = 25
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "uptime_seconds" in data

def test_predict_zero_distance(client):
    invalid = VALID_INPUT.copy()
    invalid["trip_distance"] = 0.0
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422

def test_predict_extreme_fare(client):
    invalid = VALID_INPUT.copy()
    invalid["fare_amount"] = 1000.0
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422

def test_batch_exceeds_limit(client):
    response = client.post("/predict/batch", json={"records": [VALID_INPUT] * 101})
    assert response.status_code == 422

def test_model_info(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "features" in data
    assert "metrics" in data