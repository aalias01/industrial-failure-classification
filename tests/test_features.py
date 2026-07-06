import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import api.predictor as predictor
from api.schemas import SensorReading


def test_serving_features_match_model_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    meta = json.loads(Path("models/model_meta.json").read_text())
    monkeypatch.setattr(
        predictor,
        "_clf",
        SimpleNamespace(high_load_cutoff=meta["high_load_cutoff"]),
    )
    reading = SensorReading(
        air_temperature_k=298.1,
        process_temperature_k=308.6,
        rotational_speed_rpm=1551,
        torque_nm=42.8,
        tool_wear_min=100,
    )

    features = predictor._reading_to_features(reading)

    assert list(features) == meta["feature_names"]
    assert features["temp_diff"] == pytest.approx(10.5)
    assert features["power"] == pytest.approx(1551 * 42.8)
    assert features["wear_rate"] == pytest.approx(100 / 1552)
    assert features["torque_per_wear"] == pytest.approx(42.8 / 101)
