"""Export threshold sweep counts for the frontend cost fold.

The script rebuilds the same 80/20 stratified split used by
`scripts/train_model.py`, scores the held-out rows once with the committed
model, and writes `frontend/sweep.json`. The required self-check is strict:
at threshold 0.775 the counts must match `figures/model_results.json`
exactly, fn 6 and fp 83.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import build_features, get_X_y, load_raw  # noqa: E402
from src.model import FailureClassifier  # noqa: E402


def threshold_rows(y_true: np.ndarray, y_proba: np.ndarray) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for threshold in np.round(np.arange(0.05, 0.9501, 0.005), 3):
        pred = (y_proba >= threshold).astype(int)
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        rows.append({"t": float(threshold), "fn": fn, "fp": fp})
    return rows


def main() -> None:
    raw = load_raw(PROJECT_ROOT / "data/raw/ai4i2020.csv")
    df = build_features(raw)
    X, y = get_X_y(df)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    clf = FailureClassifier.load(str(PROJECT_ROOT / "models"))
    y_proba = clf.predict_proba(X_test)
    y_true = y_test.to_numpy()
    rows = threshold_rows(y_true, y_proba)

    check = next(row for row in rows if row["t"] == 0.775)
    if check["fn"] != 6 or check["fp"] != 83:
        raise RuntimeError(
            "sweep self-check failed at t=0.775: "
            f"fn {check['fn']}, fp {check['fp']}; expected fn 6, fp 83"
        )

    payload = {
        "source": "scripts/export_sweep.py",
        "test_set": {
            "records": int(len(y_true)),
            "failures": int(y_true.sum()),
            "non_failures": int(len(y_true) - y_true.sum()),
        },
        "self_check": {"t": 0.775, "fn": check["fn"], "fp": check["fp"]},
        "rows": rows,
    }

    out_path = PROJECT_ROOT / "frontend/sweep.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        "wrote frontend/sweep.json "
        f"({len(rows)} rows, self-check fn {check['fn']}, fp {check['fp']})"
    )


if __name__ == "__main__":
    main()
