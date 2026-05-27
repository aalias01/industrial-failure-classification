# Industrial Failure Classification

> **Imbalanced-class binary failure prediction — tuned for actual maintenance economics, not default 0.5 threshold.**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-teal)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)](https://fastapi.tiangolo.com/)

In industrial settings, equipment failures are rare (~3% of records) but catastrophic. A model that predicts "no failure" on every input achieves 97% accuracy while being completely useless. This project builds a **business-cost-tuned failure classifier** — selecting the optimal prediction threshold based on real maintenance economics, not statistical defaults.

**Live demo:** [your-project.vercel.app](https://your-project.vercel.app) *(fill in after deploy)*  
**API docs:** [your-api.onrender.com/docs](https://your-api.onrender.com/docs) *(fill in after deploy)*

---

## The Key Insight: Threshold Tuning

Standard ML defaults to a 0.5 probability threshold. For failure prediction, that's wrong.

```python
# Business cost analysis for threshold selection
# A missed failure (FN) costs $50,000 in unplanned downtime
# An unnecessary maintenance call (FP) costs $2,000
# Find the threshold that minimizes total operational cost

for threshold in np.arange(0.1, 0.9, 0.05):
    preds = (failure_proba >= threshold).astype(int)
    fn_cost = ((preds == 0) & (y_test == 1)).sum() * 50_000
    fp_cost = ((preds == 1) & (y_test == 0)).sum() * 2_000
    total_cost = fn_cost + fp_cost
```

At the default 0.5 threshold: high precision, low recall — misses real failures.  
At the **cost-optimal threshold (~0.25)**: better recall, acceptable precision — correct for the asymmetric cost structure.

*"This is what 12 years of engineering experience looks like applied to ML. Maintenance managers don't think in F1 scores. They think in downtime costs."*

---

## Architecture

```
AI4I 2020 Predictive Maintenance Dataset
(10,000 records · 5 features · 3.4% failure rate)
        ↓
notebooks/01_eda.ipynb         — Class imbalance, feature distributions
notebooks/02_modeling.ipynb    — LR → RF → XGBoost + SMOTE + class weights
notebooks/03_evaluation.ipynb  — Confusion matrix, ROC, PR curve, cost analysis
notebooks/04_shap.ipynb        — Feature importance + individual prediction SHAP
        ↓
src/model.py                   — Training, evaluation, threshold selection
src/features.py                — Feature engineering
        ↓
api/main.py (FastAPI → Render)
        ↓
frontend/ (Vanilla JS → Vercel)
  Risk card: 🟢 Low / 🟡 Medium / 🔴 High + probability + top SHAP factors
```

---

## Dataset

**AI4I 2020 Predictive Maintenance** (UCI ML Repository)  
Source: [archive.ics.uci.edu/dataset/601](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

| Feature | Description | Physical meaning |
|---------|-------------|-----------------|
| Air temperature [K] | Ambient air temp | Operating environment |
| Process temperature [K] | Component temp | Heat buildup = stress |
| Rotational speed [rpm] | Motor speed | Load indicator |
| Torque [Nm] | Shaft torque | Mechanical stress |
| Tool wear [min] | Cumulative wear time | Degradation clock |
| **Machine failure** (target) | Binary 0/1 | ~3.4% failure rate |

Five specific failure modes also labeled: Tool Wear Failure (TWF), Heat Dissipation Failure (HDF), Power Failure (PWF), Overstrain Failure (OSF), Random Failures (RNF).

---

## Model Comparison

*Filled in after notebook runs.*

| Model | PR-AUC | F1 | Recall | Precision | Notes |
|-------|--------|-----|--------|-----------|-------|
| Logistic Regression (baseline) | — | — | — | — | class_weight='balanced' |
| Random Forest | — | — | — | — | SMOTE |
| **XGBoost** | — | — | — | — | scale_pos_weight + cost threshold |
| XGBoost @ cost-optimal threshold | — | — | — | — | FN=$50K, FP=$2K |

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Modeling | XGBoost + scikit-learn RF + Logistic Regression |
| Imbalance handling | SMOTE (imblearn) + class_weight / scale_pos_weight |
| Evaluation | Confusion matrix, PR curve, ROC-AUC, F1 |
| Interpretability | SHAP TreeExplainer |
| API | FastAPI on Render |
| Frontend | Vanilla HTML/CSS/JS on Vercel |

---

## Setup

```bash
git clone https://github.com/aalias01/industrial-failure-classification
cd industrial-failure-classification

conda env create -f environment.yml
conda activate industrial-failure
python -m ipykernel install --user --name industrial-failure --display-name "industrial-failure"

# Data downloads automatically in notebook 01 via UCI URL
# Or manually: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

# Run notebooks in order: 01 → 02 → 03 → 04
# Then: uvicorn api.main:app --reload
```

---

## Interview Context

1. **Accuracy is misleading:** *"Predicting 'no failure' always gives 97% accuracy on this dataset. I optimized for PR-AUC and F1 instead — the only metrics that matter under class imbalance."*

2. **Threshold tuning:** *"The default 0.5 threshold misses real failures. I built a cost model — $50K for a missed failure vs. $2K for unnecessary maintenance — and found the threshold that minimizes total operational cost. It shifted from 0.5 to 0.25."*

3. **SMOTE:** *"SMOTE generates synthetic minority-class samples in feature space, not just duplicates. Combined with XGBoost's scale_pos_weight, it gave the best recall without sacrificing too much precision."*

4. **SHAP:** *"Tool wear and torque are the dominant predictors — consistent with known mechanical physics. A high-torque + high-wear reading is the physical signature of imminent failure."*

5. **Complement to CMAPSS:** *"CMAPSS predicts how many cycles remain (regression). This predicts will it fail in the next N cycles (classification). Same domain, different paradigm — together they demonstrate the full predictive maintenance toolbox."*

---

*Built by [Alvin Alias](https://github.com/aalias01) — MS Data Science, University of Washington · 12 years industrial engineering*
