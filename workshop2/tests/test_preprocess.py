import pandas as pd
import os
from src.preprocess import preprocess

def test_preprocess(tmp_path):
    # Create sample raw data with no missing values
    df = pd.DataFrame({
        'AT': [10, 20, 30, 40],
        'V': [100, 200, 300, 400],
        'AP': [1000, 1001, 1002, 1003],
        'RH': [10, 20, 30, 40],
        'PE': [500, 600, 700, 800]
    })
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    df.to_csv(raw / "raw_data.csv", index=False)

    # Run preprocess
    preprocess(str(raw), str(processed), test_size=0.5, random_state=0)

    # Check files exist
    train_csv = processed / "train_processed.csv"
    test_csv = processed / "test_processed.csv"
    scaler_file = processed / "scaler.pkl"
    assert train_csv.exists() and test_csv.exists() and scaler_file.exists()

    # Check split sizes
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    total_rows = len(train_df) + len(test_df)
    assert total_rows == len(df)
