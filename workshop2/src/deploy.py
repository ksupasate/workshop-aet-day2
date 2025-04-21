from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

class Item(BaseModel):
    data: list

app = FastAPI()
model = joblib.load('models/model.pkl')

@app.post('/predict')
def predict(item: Item):
    df = pd.DataFrame(item.data)
    preds = model.predict(df)
    return {'predictions': preds.tolist()}
