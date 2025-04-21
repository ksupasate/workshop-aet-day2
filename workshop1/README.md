
# 🧪 Hands-on Workshop: Model Deployment & MLOps Foundations

## 🎯 Objective
This workshop guides you through the end-to-end process of deploying a trained ML model using FastAPI and introduces core MLOps practices including model versioning, serialization, experiment tracking, API testing, and basic CI/CD concepts.

---

## 📋 Prerequisites

- Python 3.8+
- `pip install -r requirements.txt`
- Basic understanding of machine learning pipelines
- Familiarity with Git, FastAPI, and Scikit-learn is helpful

---

## 📦 Folder Structure

```
📁 model-deployment-mlops/
├── app.py                  # FastAPI app to serve model predictions
├── train_and_save.py       # Script to train and export the model
├── iris_pipeline.pkl       # Saved trained pipeline
├── requirements.txt        # Required packages
├── test_app.py             # Script to test API
├── dvc.yaml                # (Optional) DVC pipeline config
├── mlruns/                 # (Optional) MLflow tracking directory
└── README.md               # This file
```

---

## ⚙️ Step 1: Train & Save Model

```bash
python train_and_save.py
```

This creates `iris_pipeline.pkl` using Scikit-learn and saves the pipeline to disk.

---

## 🌐 Step 2: Serve with FastAPI

```bash
uvicorn app:app --reload
```

FastAPI server will be available at:  
👉 `http://127.0.0.1:8000`

---

## 🧪 Step 3: Test the API

```bash
python test_app.py
```

Or use curl:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" \
-d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

---

## 📊 Step 4: Track Experiments with MLflow (Optional)

### Start MLflow UI

```bash
mlflow ui
```

Track your model training, metrics, parameters, and artifacts under `http://localhost:5000`.

In your Python script, you can use:

```python
import mlflow
mlflow.log_param("model", "LogisticRegression")
mlflow.log_metric("accuracy", 0.95)
mlflow.sklearn.log_model(pipeline, "model")
```

---

## 🔁 Step 5: Version Data & Pipelines with DVC (Optional)

### Initialize DVC

```bash
dvc init
dvc add iris_pipeline.pkl
git add iris_pipeline.pkl.dvc .gitignore
git commit -m "Track model pipeline with DVC"
```

Use `dvc.yaml` to define automated pipelines (e.g., preprocess → train → save).

---

## 🧪 Step 6: Add Tests and CI/CD (Optional)

- Use `pytest` to test your API routes.
- Add a `.github/workflows/ci.yml` to trigger lint, test, or build jobs.

---

## ✅ Optional Extensions

- Train your own Scikit-learn pipeline on a different dataset
- Replace FastAPI with Flask for comparison
- Deploy your API to Render, Railway, or AWS Lambda
- Connect Prometheus & Grafana for request monitoring

---

## 🧠 Final Tip

> “MLOps makes machine learning a continuous, collaborative, and production-ready discipline — not just a one-time experiment.”

---

Happy deploying and tracking!
