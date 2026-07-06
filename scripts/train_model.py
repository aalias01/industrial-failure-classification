"""Train and save the portfolio demo model.

This script gives the project a repeatable path from downloaded AI4I data to
the model artifacts used by the FastAPI app. The notebooks remain the narrative
analysis, while this script is the quick rebuild command before deployment.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import build_features, download_data, get_X_y, load_raw  # noqa: E402
from src.model import FailureClassifier  # noqa: E402


def main() -> None:
    download_data()
    df = build_features(load_raw())
    high_load_cutoff = df.attrs.get("high_load_cutoff")
    X, y = get_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    clf = FailureClassifier(
        model_type="xgb",
        use_smote=True,
        high_load_cutoff=high_load_cutoff,
    )
    clf.fit(X_train, y_train)

    results = clf.evaluate(X_test, y_test)
    clf.optimal_threshold = results["optimal_threshold"]
    results = clf.evaluate(X_test, y_test)
    clf.save("models/")

    reports_dir = Path("figures")
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "model_results.json").write_text(json.dumps(results, indent=2))

    print("\nModel evaluation:")
    for key in [
        "roc_auc",
        "pr_auc",
        "f1",
        "precision",
        "recall",
        "business_cost",
        "optimal_threshold",
        "optimal_cost",
        "n_fn",
        "n_fp",
    ]:
        print(f"  {key}: {results[key]}")
    print("\nSaved artifacts:")
    print("  models/xgb_classifier.joblib")
    print("  models/scaler.joblib")
    print("  models/model_meta.json")
    print("  figures/model_results.json")


if __name__ == "__main__":
    main()
