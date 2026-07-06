from api.predictor import risk_level_for_probability


def test_risk_zones_at_cost_threshold() -> None:
    threshold = 0.775

    assert risk_level_for_probability(0.24, threshold) == "low"
    assert risk_level_for_probability(0.25, threshold) == "medium"
    assert risk_level_for_probability(0.50, threshold) == "medium"
    assert risk_level_for_probability(0.774, threshold) == "medium"
    assert risk_level_for_probability(0.775, threshold) == "high"
    assert risk_level_for_probability(0.776, threshold) == "high"


def test_risk_zones_guard_when_threshold_is_below_low_cutoff() -> None:
    threshold = 0.20

    assert risk_level_for_probability(0.19, threshold) == "low"
    assert risk_level_for_probability(0.20, threshold) == "high"
    assert risk_level_for_probability(0.24, threshold) == "high"
