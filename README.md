# Industrial Failure Classification

Binary failure prediction on imbalanced sensor data, tuned for maintenance economics instead of statistical defaults. The cost-optimal prediction threshold is 0.775, not 0.5: at that threshold the model catches 62 of 68 test-set failures (91.2% recall) at an estimated total cost of $466K, versus $50K per missed failure.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![CI](https://github.com/aalias01/industrial-failure-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/aalias01/industrial-failure-classification/actions/workflows/ci.yml)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-teal)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)

**[Live demo](https://machine-failure.alvinalias.com)** | **[API docs](https://industrial-failure-classification.onrender.com/docs)**

The API runs on Render's free tier and spins down when idle; the first request can take ~50 seconds while it wakes.

## Why threshold tuning is the point

Equipment failures in this dataset are rare (3.4% of records) but expensive. A model that predicts "no failure" on every input scores 97% accuracy and is useless. And even a good classifier at the default 0.5 threshold makes the wrong economic trade: it balances false alarms against missed failures as if they cost the same.

They don't. A missed failure (unplanned downtime) is modeled at $50,000; an unnecessary maintenance call at $2,000. Sweeping the threshold and minimizing total cost:

```python
for threshold in np.arange(0.1, 0.9, 0.05):
    preds = (failure_proba >= threshold).astype(int)
    fn_cost = ((preds == 0) & (y_test == 1)).sum() * 50_000
    fp_cost = ((preds == 1) & (y_test == 0)).sum() * 2_000
    total_cost = fn_cost + fp_cost
```

The optimum lands at 0.775. Maintenance managers don't think in F1 scores, they think in downtime dollars; this translates the model into those terms.

## Results

80/20 stratified train/test split, AI4I 2020 dataset.

| Model | PR-AUC | F1 | Recall | Precision | Notes |
|-------|--------|-----|--------|-----------|-------|
| Logistic regression (baseline) | 0.455 | 0.311 | 0.868 | 0.189 | class_weight='balanced' |
| Random Forest | 0.820 | 0.682 | 0.868 | 0.562 | SMOTE |
| XGBoost | 0.841 | 0.601 | 0.897 | 0.452 | SMOTE + scale_pos_weight |
| **XGBoost @ cost-optimal threshold** | 0.841 | 0.582 | 0.912 | 0.428 | threshold=0.775; 6 missed failures, 83 false alarms |

XGBoost ROC-AUC: 0.979. Full metrics in `figures/model_results.json`. Model card: [models/MODEL_CARD.md](models/MODEL_CARD.md).

![Business cost vs threshold](figures/cost_vs_threshold.png)

![Confusion matrices](figures/confusion_matrices.png)

![Precision-recall curve](figures/pr_curve.png)

## Design decisions

**PR-AUC and F1 over accuracy.** Accuracy is meaningless at 3.4% positive rate; the project optimizes the metrics that survive class imbalance.

**SMOTE plus scale_pos_weight.** SMOTE synthesizes minority-class samples in feature space rather than duplicating rows. Combined with XGBoost's `scale_pos_weight`, it gave the best recall without collapsing precision.

**Physics-consistent explanations.** The strongest SHAP contributors are power, rotational speed, tool wear, and torque, which matches mechanical stress and accumulated-wear physics rather than dataset artifacts.

**Classification, not RUL regression.** My [CMAPSS project](https://turbofan.alvinalias.com) predicts how many cycles remain; this one predicts whether a machine fails in the near term. Same domain, different problem shape, different evaluation.

## Dataset

**AI4I 2020 Predictive Maintenance** ([UCI ML Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)): 10,000 records, 3.4% failure rate.

| Feature | Physical meaning |
|---------|-----------------|
| Air temperature [K] | Operating environment |
| Process temperature [K] | Heat buildup, thermal stress |
| Rotational speed [rpm] | Load indicator |
| Torque [Nm] | Mechanical stress |
| Tool wear [min] | Degradation clock |
| Machine failure (target) | Binary, with 5 labeled failure modes (TWF, HDF, PWF, OSF, RNF) |

## How it works

```
AI4I 2020 data (auto-downloaded in notebook 01)
        |
notebooks/01_eda.ipynb          class imbalance, feature distributions
notebooks/02_modeling.ipynb     LR -> RF -> XGBoost, SMOTE, class weights
notebooks/03_evaluation.ipynb   confusion matrix, ROC, PR curve, cost analysis
notebooks/04_shap.ipynb         global + per-prediction SHAP
        |
scripts/train_model.py          repeatable build of deployment artifacts
        |
api/main.py (FastAPI, Render)   POST /predict -> probability + risk tier + top SHAP factors
        |
frontend/ (vanilla JS, Vercel)  go/no-go checkpoint with known records and cost sweep
```

## Tech stack

Python 3.11, XGBoost 2.0, scikit-learn 1.3, imblearn (SMOTE), SHAP in notebooks with XGBoost native `pred_contribs` in the API, FastAPI on Render, vanilla HTML/CSS/JS on Vercel.

## Run it locally

```bash
git clone https://github.com/aalias01/industrial-failure-classification
cd industrial-failure-classification

conda env create -f environment.yml
conda activate industrial-failure
python -m ipykernel install --user --name industrial-failure

# data downloads automatically in notebook 01 via the UCI URL

python scripts/train_model.py   # builds API artifacts
uvicorn api.main:app --reload   # docs at http://localhost:8000/docs
```

Run the notebooks in order (01 through 04). Training writes:

```
models/xgb_classifier.joblib
models/scaler.joblib
models/model_meta.json
figures/model_results.json
```

Run checks:

```bash
ruff check .
pytest -q
```

The calibration fold reads `frontend/sweep.json`. Rebuild it with `python scripts/export_sweep.py`; the script fails unless the 0.775 row reproduces 6 missed failures and 83 false alarms.

## Limitations

- The $50K / $2K cost assumptions are representative, not customer data; the threshold sweep re-runs trivially with real numbers.
- AI4I is a synthetic dataset built to mimic real predictive-maintenance data, so results demonstrate method, not field performance.
- The committed model artifact is a quick build from `scripts/train_model.py`; the notebooks document the full comparison.

## Dataset credit

Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset.* UCI Machine Learning Repository. The dataset keeps its own terms; MIT here covers this repo's code.

Built by [Alvin Alias](https://github.com/aalias01), MS Data Science, University of Washington. 12 years industrial engineering.
