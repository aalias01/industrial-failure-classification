import api.predictor as predictor
from api.schemas import SensorReading


def test_predictor_roundtrip_with_shap() -> None:
    predictor.load_model()
    reading = SensorReading(
        air_temperature_k=298.1,
        process_temperature_k=308.6,
        rotational_speed_rpm=1551,
        torque_nm=42.8,
        tool_wear_min=0,
    )

    response = predictor.predict(reading, include_shap=True)

    assert 0 <= response.failure_probability <= 1
    assert response.prediction == int(response.failure_probability >= response.threshold_used)
    if response.prediction == 1:
        assert response.estimated_cost_if_ignored == predictor._clf.fn_cost
    else:
        assert response.estimated_cost_if_ignored is None
    assert response.top_shap_factors is not None
    assert len(response.top_shap_factors) == 5
