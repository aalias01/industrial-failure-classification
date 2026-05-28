# Portfolio Readiness Checklist

This project is close to portfolio-ready, but it is not ready to present as a finished deployed project until the notebooks, final figures, and live URLs are completed.

## Current Status

- Code is present for feature engineering, model training, FastAPI serving, and a vanilla JS frontend.
- Documentation already frames the portfolio story well: imbalanced industrial failures, PR-AUC over accuracy, cost-based threshold tuning, and SHAP explanations.
- Local source files compile when Python is pointed at a writable bytecode cache.
- Model artifacts now exist: `models/xgb_classifier.joblib`, `models/scaler.joblib`, and `models/model_meta.json`.
- Quick-build metrics now exist in `figures/model_results.json`.
- All four notebooks have been executed for the current artifact.
- README live demo/API URLs are still placeholders.

## What I Completed In This Pass

- Added `scripts/train_model.py` so the model can be rebuilt without manually stepping through notebooks.
- Made the frontend API base configurable through `window.INDUSTRIAL_FAILURE_API_BASE` or `localStorage`, with `http://localhost:8000` as the development default.
- Added this portfolio-readiness guide with the remaining steps needed before GitHub/public deployment.
- Created the `industrial-failure` conda environment.
- Trained the first XGBoost artifact and generated `figures/model_results.json`.
- Executed `notebooks/01_eda.ipynb`, `notebooks/02_modeling.ipynb`, `notebooks/03_evaluation.ipynb`, and `notebooks/04_shap.ipynb`.
- Generated `figures/cost_vs_threshold.png`, `figures/confusion_matrices.png`, `figures/pr_curve.png`, and `figures/shap_importance.csv`.
- Smoke-tested the FastAPI app with `TestClient`; `/health` loads the model and `/predict` returns a valid response.
- Verified Python syntax with:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/industrial_failure_pycache python3 -m compileall src api
```

## Remaining Blockers

- Live Render and Vercel deployments are not complete yet.

## Finish Locally

1. Create the environment.

```bash
conda env create -f environment.yml
conda activate industrial-failure
python -m ipykernel install --user --name industrial-failure --display-name "industrial-failure"
```

2. Train or retrain the model and generate first-pass metrics.

```bash
python scripts/train_model.py
```

3. Confirm these files now exist.

```bash
ls -lh models/xgb_classifier.joblib models/scaler.joblib models/model_meta.json figures/model_results.json
```

4. Run notebooks in order for portfolio-quality narrative outputs.

```bash
jupyter notebook
```

Run:

- `notebooks/01_eda.ipynb`
- `notebooks/02_modeling.ipynb`
- `notebooks/03_evaluation.ipynb`
- `notebooks/04_shap.ipynb`

5. Save the key figures into `figures/`.

Expected assets:

- `figures/cost_vs_threshold.png`
- `figures/confusion_matrices.png`
- `figures/pr_curve.png`
- `figures/shap_importance.csv`
- Optional SHAP waterfall screenshot for the README

6. Update `README.md`.

Replace:

- pending Vercel URL
- pending Render URL
- every remaining dash in the model comparison table
- any threshold claim if a retrain changes the cost-selected value materially from `0.775`

Use values from `figures/model_results.json` and notebook output.

7. Test the API locally.

```bash
uvicorn api.main:app --reload
```

Open:

- `http://localhost:8000/health`
- `http://localhost:8000/docs`

Example prediction:

```bash
curl -X POST "http://localhost:8000/predict?shap=true" \
  -H "Content-Type: application/json" \
  -d '{
    "air_temperature_k": 298.1,
    "process_temperature_k": 308.6,
    "rotational_speed_rpm": 1551,
    "torque_nm": 42.8,
    "tool_wear_min": 0
  }'
```

8. Test the frontend locally.

From the project root:

```bash
python -m http.server 8080 --directory frontend
```

Open `http://localhost:8080`. The API status badge should show the model loaded once `uvicorn` is running and model artifacts exist.

## Deploy

1. Push the finished repo to GitHub.

```bash
git status
git add README.md PORTFOLIO_READINESS.md scripts/train_model.py frontend/app.js frontend/index.html models/xgb_classifier.joblib models/scaler.joblib models/model_meta.json figures/
git commit -m "Prepare industrial failure classifier portfolio project"
git push origin main
```

2. Deploy the API on Render.

- Create a new Render web service from the GitHub repo.
- Render should detect `render.yaml`.
- Confirm the build command is `pip install -r requirements.txt`.
- Confirm the start command is `uvicorn api.main:app --host 0.0.0.0 --port $PORT`.
- After deploy, open `https://<your-render-service>.onrender.com/health`.
- Confirm `model_loaded` is `true`.

3. Deploy the frontend on Vercel.

- Use `frontend/` as the project root if Vercel asks.
- Before deployment, set the API URL in `frontend/index.html` by adding this before `app.js`:

```html
<script>
  window.INDUSTRIAL_FAILURE_API_BASE = 'https://<your-render-service>.onrender.com';
</script>
```

4. Update CORS in `api/main.py`.

Add your Vercel domain to `allow_origins`, then redeploy Render.

```python
"https://<your-vercel-project>.vercel.app",
```

5. Update `README.md` with final links.

```markdown
**Live demo:** [industrial-failure-classifier](https://<your-vercel-project>.vercel.app)
**API docs:** [Render FastAPI docs](https://<your-render-service>.onrender.com/docs)
```

## Final Portfolio QA

- README has real metrics, not placeholders.
- README includes at least one cost-threshold plot.
- README clearly explains why accuracy is misleading.
- `/health` returns `model_loaded: true`.
- `/predict?shap=true` returns probability, risk level, threshold, and SHAP factors.
- Frontend shows low, medium, and high examples without layout breakage.
- GitHub repo description is concise: `Business-cost-tuned industrial failure classifier using XGBoost, SMOTE, SHAP, FastAPI, and Vercel.`
- Repository is public only after artifacts and links are working.

## Resume Bullet Template

Use this after final metrics are filled in:

> Built an imbalanced industrial failure classifier on the UCI AI4I dataset using XGBoost, SMOTE, and cost-based threshold tuning; improved recall from `<baseline>` to `<final>` while minimizing a `$50K` false-negative / `$2K` false-positive maintenance cost model, then deployed the model with FastAPI and a custom JS frontend.
