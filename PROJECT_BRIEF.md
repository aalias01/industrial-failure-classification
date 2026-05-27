# Project Brief — Industrial Failure Classification

| Priority Score | Tier | Recommended Ship Slot | Effort |
|----------------|------|----------------------|--------|
| **3.95** | **P2** | **Order #7** *(after CMAPSS · Retail Returns · RAG · HVAC · Supply Chain · Energy Demand)* | 8–10 hrs across 3 sessions |

**Score breakdown** — ED 4 · DIFF 3 · SC 4 · DSS 4 · BV 5 · EE 4
**Lane:** A (Industrial — Manufacturing · Predictive Maintenance — classification angle)
**Target companies:** Siemens, Honeywell, GE Vernova, any manufacturer with production equipment

**Conditions to re-rank:**
- DIFF is the lowest score among industrial briefs (3) — there's thematic overlap with CMAPSS. If CMAPSS already shipped and conveyed the predictive-maintenance story well, this becomes "skill-completion" rather than "headline" — keep at P2.
- If applying to a Siemens / Honeywell / GE Vernova role that explicitly asks for *imbalanced-class classification* or *cost-based threshold tuning*: promote to P1 (the threshold-tuning section is a strong differentiator on its own).
- Highest BV (5) and EE (4) in the portfolio — this is the *cheapest* industrial project to ship if you need a quick win. Use it as a tactical add when calendar is tight.

---

## Problem Statement

In industrial settings, equipment failures are rare but costly. A binary classifier that predicts "will this machine fail in the next N cycles?" must be tuned for the asymmetric cost of errors — missing a real failure (false negative) is far more expensive than a false alarm (false positive). Given sensor readings and process parameters, build a failure classifier that is both accurate and tuned for the operational cost of errors.

---

## Why This Project for Alvin

- **Adds classification to the portfolio:** CMAPSS is regression (how many cycles?). This is binary classification (fail or not?). Same domain, different paradigm.
- **Imbalanced classes:** Failures are rare — typically 1–3% of records. Handling imbalanced data is a core DS skill.
- **Business cost framing:** This is where engineering intuition meets ML — choosing the precision/recall tradeoff based on actual maintenance economics.
- **Clean benchmark dataset:** AI4I 2020 is UCI-hosted, well-documented, and has published baselines to compare against.

---

## Dataset

**Primary:** AI4I 2020 Predictive Maintenance Dataset
- UCI ML Repository: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- 10,000 records, 5 features + 5 failure modes
- Features: air temperature, process temperature, rotational speed, torque, tool wear
- Target: binary (machine failure: yes/no) + specific failure type
- Class imbalance: ~3.4% failure rate

**Stretch:** Add the Microsoft Azure Predictive Maintenance dataset (Kaggle) for a multi-machine comparison.

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Data wrangling | Pandas, NumPy |
| EDA | Matplotlib, Seaborn |
| Baseline model | Logistic Regression |
| Primary model | Random Forest + XGBoost classifier |
| Imbalance handling | SMOTE (oversample minority) + class_weight='balanced' |
| Evaluation | Confusion matrix, precision, recall, F1, ROC-AUC, PR curve |
| Interpretability | SHAP — feature importance + individual prediction explanation |
| Threshold tuning | Precision-recall curve + business cost analysis |
| API | FastAPI: POST /predict → failure probability + SHAP |
| Frontend | Simple form: input sensor readings → failure risk card |
| Environment | conda (environment.yml) |

---

## The Key Concept: Threshold Tuning

Standard ML defaults to 0.5 probability threshold. For failure prediction, that's wrong.

```python
# Business cost analysis for threshold selection
# Assume: false negative (missed failure) costs $50,000 in unplanned downtime
# Assume: false positive (unnecessary maintenance) costs $2,000
# Optimal threshold minimizes: FN_cost * FN_count + FP_cost * FP_count

costs = []
for threshold in np.arange(0.1, 0.9, 0.05):
    preds = (proba >= threshold).astype(int)
    fn = ((preds == 0) & (y_test == 1)).sum()
    fp = ((preds == 1) & (y_test == 0)).sum()
    total_cost = fn * 50000 + fp * 2000
    costs.append({'threshold': threshold, 'cost': total_cost, 'fn': fn, 'fp': fp})
```

This business cost framing is what makes this project different from "I ran XGBoost and got 97% accuracy on imbalanced data."

---

## Deliverables

1. `notebooks/01_eda.ipynb` — class distribution, feature distributions, correlation with failure
2. `notebooks/02_modeling.ipynb` — baseline → RF → XGBoost; imbalance handling; threshold tuning
3. `notebooks/03_evaluation.ipynb` — confusion matrix, ROC, PR curve, business cost analysis
4. `notebooks/04_shap.ipynb` — SHAP feature importance + individual prediction explanations
5. `src/features.py` — feature engineering functions
6. `src/model.py` — training, evaluation, threshold selection
7. `api/main.py` — FastAPI: POST /predict → {failure_probability, risk_level, shap_factors}
8. `frontend/` — sensor input form → failure risk card (red/amber/green)
9. `README.md` — GitHub-ready with results table and live demo

---

## Project Phases

### Phase 1 — EDA + Baseline (2 hrs)
- [ ] Load AI4I dataset, understand schema
- [ ] Class imbalance analysis: plot failure rate by feature range
- [ ] Baseline: logistic regression with class_weight='balanced'
- [ ] Evaluate: precision, recall, F1, AUC — document why accuracy is misleading here

### Phase 2 — Modeling + Imbalance Handling (3 hrs)
- [ ] Random Forest with class_weight='balanced'
- [ ] XGBoost with scale_pos_weight
- [ ] SMOTE: oversample minority class, evaluate impact
- [ ] Cross-validation: stratified k-fold to preserve class ratios
- [ ] Model comparison table: all approaches vs. baseline

### Phase 3 — Threshold Tuning + SHAP (2–3 hrs)
- [ ] Precision-recall curve: visualize tradeoff
- [ ] Business cost analysis: assign FN/FP costs, find optimal threshold
- [ ] SHAP: feature importance globally + explanation for a specific predicted failure
- [ ] Tie SHAP output to physical sensor meaning

### Phase 4 — API + Frontend (2 hrs)
- [ ] FastAPI endpoint: sensor readings in → failure probability + risk level + top SHAP factors
- [ ] Frontend: clean input form → result card (color-coded risk, probability, explanation)
- [ ] Deploy Render + Vercel
- [ ] Update README

---

## Interview Talking Points

1. *"Accuracy is misleading on this dataset — 97% accuracy by predicting 'no failure' every time. I optimized for F1 score and PR-AUC instead."*
2. *"I tuned the classification threshold using a business cost model — a missed failure costs $50K in unplanned downtime vs. $2K for unnecessary maintenance. The optimal threshold shifted from 0.5 to 0.25."*
3. *"SMOTE generates synthetic minority class samples in feature space. Combined with XGBoost's scale_pos_weight, it gave the best recall without sacrificing too much precision."*
4. *"The SHAP analysis showed tool wear and torque are the dominant failure predictors — consistent with known mechanical wear physics."*

---

## Success Criteria

- [ ] Live FastAPI + Vercel demo working
- [ ] Business cost threshold analysis documented and explainable
- [ ] Model outperforms naive baseline on F1 and PR-AUC
- [ ] SHAP analysis completed and tied to physical sensor meaning
- [ ] Resume bullet: *"Built imbalanced-class industrial failure classifier on UCI AI4I dataset; applied SMOTE and business-cost threshold tuning, achieving F1 of X vs. Y baseline; deployed as FastAPI + custom frontend."*

---

*Brief created: April 2026 · Updated April 2026 · May 2026 (slot confirmed #7 — activated; all 6 preceding projects now active) | Priority Score 3.95 · Tier P2 · Ship slot #7 · Cheapest industrial brief to ship when calendar is tight (BV 5, EE 4)*
