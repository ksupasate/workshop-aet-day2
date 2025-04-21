from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load model
model = joblib.load("iris_pipeline.pkl")

# Define app and request schema
app = FastAPI()

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(features: IrisFeatures):
    X = np.array([[features.sepal_length, features.sepal_width,
                   features.petal_length, features.petal_width]])
    prediction = model.predict(X)[0]
    return {"prediction": int(prediction)}
