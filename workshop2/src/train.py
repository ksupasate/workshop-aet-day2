import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import joblib

def train(input_dir, model_output):
    df = pd.read_csv(f'{input_dir}/processed_data.csv')
    X = df.drop('target', axis=1)
    y = df['target']

    mlflow.start_run()
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    mlflow.sklearn.log_model(model, 'model')
    mlflow.log_param('n_estimators', 100)
    mlflow.end_run()

    joblib.dump(model, f'{model_output}/model.pkl')
    print(f'Model saved to {model_output}/model.pkl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--model-output', required=True)
    args = parser.parse_args()
    train(args.input, args.model_output)
