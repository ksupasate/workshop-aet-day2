import pandas as pd
import os
from src.train import train

def test_train(tmp_path):
    df = pd.DataFrame({
        'feature1': [1,2,3,4],
        'feature2': [4,3,2,1],
        'target': [0,1,0,1]
    })
    processed = tmp_path / "processed"
    models = tmp_path / "models"
    processed.mkdir()
    models.mkdir()
    df.to_csv(processed / "processed_data.csv", index=False)

    train(str(processed), str(models))
    assert os.path.exists(models / "model.pkl")
