from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
from src.model import FailureClassifier
from api.schemas import SensorReading, PredictResponse, SHAPFactor

MODEL_DIR = Path("models")
_clf: Optional[FailureClassifier] = None
_ready = False

RISK_LEVELS = [(0.25, "low"), (0.60, "medium"), (1.01, "high")]


def load_model() -> None:
    global _clf, _ready
    if not (MODEL_DIR / "xgb_classifier.joblib").exists():
        print("[predictor] Model not found — API in degraded mode. Run notebooks first.")
        return
    _clf = FailureClassifier.load(str(MODEL_DIR))
    _ready = True
    print(f"[predictor] Model loaded. Threshold={_clf.optimal_threshold:.2f}")


def is_ready() -> bool:
    return _ready


def predict(reading: SensorReading, include_shap: bool = True) -> PredictResponse:
    if not _ready or _clf is None:
        raise RuntimeError("Model not loaded.")

    features = _reading_to_features(reading)
    row_df = pd.DataFrame([features])
    proba = float(_clf.predict_proba(row_df)[0])
    pred  = int(proba >= _clf.optimal_threshold)

    risk_level = "high"
    for threshold, level in RISK_LEVELS:
        if proba < threshold:
            risk_level = level
            break

    shap_factors = None
    if include_shap:
        try:
            raw = _clf.top_shap_factors(features, top_n=5)
            shap_factors = [SHAPFactor(**f) for f in raw]
        except Exception as e:
            print(f"[predictor] SHAP failed: {e}")

    return PredictResponse(
        failure_probability=round(proba, 4),
        risk_level=risk_level,
        prediction=pred,
        threshold_used=_clf.optimal_threshold,
        estimated_cost_if_ignored=_clf.fn_cost if pred == 1 else None,
        top_shap_factors=shap_factors,
    )


def _reading_to_features(r: SensorReading) -> dict:
    return {
        "Air temperature [K]":       r.air_temperature_k,
        "Process temperature [K]":   r.process_temperature_k,
        "Rotational speed [rpm]":    r.rotational_speed_rpm,
        "Torque [Nm]":               r.torque_nm,
        "Tool wear [min]":           r.tool_wear_min,
        "temp_diff":                 r.process_temperature_k - r.air_temperature_k,
        "power":                     r.rotational_speed_rpm * r.torque_nm,
        "wear_rate":                 r.tool_wear_min / (r.rotational_speed_rpm + 1),
        "torque_per_wear":           r.torque_nm / (r.tool_wear_min + 1),
        "high_load":                 int(r.rotational_speed_rpm * r.torque_nm >= 60000),
    }
