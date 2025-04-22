import argparse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import mlflow
import joblib
import os

def train(input_dir, model_output):
    train_path = os.path.join(input_dir, 'train_processed.csv')
    df = pd.read_csv(train_path)
    X = df.drop('PE', axis=1)
    y = df['PE']

    mlflow.start_run()
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    mlflow.sklearn.log_model(model, 'model')
    mlflow.log_param('n_estimators', 100)
    mlflow.end_run()

    os.makedirs(model_output, exist_ok=True)
    joblib.dump(model, os.path.join(model_output, 'model.pkl'))
    print(f"Model saved to {os.path.join(model_output, 'model.pkl')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help="Directory with train_processed.csv")
    parser.add_argument('--model-output', required=True, help="Directory to save the trained model")
    args = parser.parse_args()
    train(args.input, args.model_output)
