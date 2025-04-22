import pandas as pd
import os
from src.train import train

def test_train(tmp_path):
    # Create sample train_processed.csv
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4],
        'PE': [0.1, 0.2, 0.3, 0.4]
    })
    processed = tmp_path / "processed"
    models = tmp_path / "models"
    processed.mkdir()
    models.mkdir()
    df.to_csv(processed / "train_processed.csv", index=False)

    # Run train
    train(str(processed), str(models))
    # Check model file
    assert (models / "model.pkl").exists()
