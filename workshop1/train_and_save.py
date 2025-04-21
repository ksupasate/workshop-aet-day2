import pandas as pd
import joblib
import json
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save model
joblib.dump(pipeline, "iris_pipeline.pkl")

# Save metrics for DVC
metrics = {
    "accuracy": accuracy
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

# Log with MLflow
mlflow.set_experiment("Iris-Pipeline")
with mlflow.start_run():
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(pipeline, "model")