from fastapi.testclient import TestClient

from api.main import app


def test_health_includes_risk_zones() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    data = response.json()
    assert response.status_code == 200
    assert data["model_loaded"] is True
    assert data["risk_zones"] == {"low_below": 0.25, "high_at": data["optimal_threshold"]}


def test_predict_200_and_422_paths() -> None:
    payload = {
        "air_temperature_k": 298.1,
        "process_temperature_k": 308.6,
        "rotational_speed_rpm": 1551,
        "torque_nm": 42.8,
        "tool_wear_min": 0,
    }

    with TestClient(app) as client:
        ok_response = client.post("/predict?shap=true", json=payload)
        invalid_response = client.post(
            "/predict?shap=false",
            json={**payload, "air_temperature_k": 100},
        )

    assert ok_response.status_code == 200
    data = ok_response.json()
    assert 0 <= data["failure_probability"] <= 1
    assert data["prediction"] == int(data["failure_probability"] >= data["threshold_used"])
    assert len(data["top_shap_factors"]) == 5

    assert invalid_response.status_code == 422
