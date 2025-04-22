import argparse
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

def evaluate(model_path, test_data_dir):
    test_path = os.path.join(test_data_dir, 'test_processed.csv')
    df = pd.read_csv(test_path)
    X = df.drop('PE', axis=1)
    y = df['PE']

    model = joblib.load(model_path)
    preds = model.predict(X)

    mse = mean_squared_error(y, preds)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'R^2: {r2:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help="Path to trained model.pkl")
    parser.add_argument('--test-data', required=True, help="Directory with test_processed.csv")
    args = parser.parse_args()
    evaluate(args.model, args.test_data)
