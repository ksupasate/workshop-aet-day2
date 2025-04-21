
# 🧪 Hands-on Workshop: Model Deployment & MLOps Foundations

## 🎯 Objective
This workshop guides you through the end-to-end process of deploying a trained ML model using FastAPI and introduces core MLOps practices including model versioning, serialization, API testing, and basic CI/CD concepts.

---

## 📋 Prerequisites

- Python 3.8+
- `pip install -r requirements.txt`
- Familiarity with Scikit-learn and FastAPI (recommended but not required)

---

## 📦 Folder Structure

```
📁 model-deployment-mlops/
├── app.py                  # FastAPI app to serve model predictions
├── train_and_save.py       # Script to train and export the model
├── iris_pipeline.pkl       # Saved trained pipeline
├── requirements.txt        # Required packages
├── test_app.py             # API test script (optional)
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

This spins up a local web server at `http://127.0.0.1:8000`.

---

## 🧪 Step 3: Test the API

Use `curl` or Postman to send a request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" \
-d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

Expected output:

```json
{"prediction": 0}
```

---

## 🔁 Step 4: MLOps Enhancements

- ✅ **Versioning**: Use `joblib` or `DVC` to version pipelines
- ✅ **Monitoring**: Log requests or use Prometheus for metrics
- ✅ **Testing**: Add `pytest` to test API routes
- ✅ **CI/CD (Optional)**: Add GitHub Actions workflow for automated deployment

---

## ✅ Optional Extensions

- Train your own Scikit-learn pipeline on a different dataset
- Replace FastAPI with Flask and compare
- Deploy to Heroku, Render, or Railway for cloud deployment
- Integrate MLflow for experiment tracking

---

## 🧠 Final Tip

> “MLOps is about turning ML from one-time models into scalable, testable, and maintainable systems.”

---

Happy deploying!
