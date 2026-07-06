import json
from pathlib import Path


def test_sweep_self_check_row_matches_published_results() -> None:
    sweep = json.loads(Path("frontend/sweep.json").read_text())
    row = next(item for item in sweep["rows"] if item["t"] == 0.775)

    assert row["fn"] == 6
    assert row["fp"] == 83
    assert sweep["self_check"] == {"t": 0.775, "fn": 6, "fp": 83}
    assert sweep["test_set"]["failures"] == 68
