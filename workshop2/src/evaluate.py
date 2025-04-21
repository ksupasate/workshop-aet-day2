import argparse
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

def evaluate(model_path, test_data_dir):
    df = pd.read_csv(f'{test_data_dir}/processed_data.csv')
    X = df.drop('target', axis=1)
    y = df['target']

    model = joblib.load(model_path)
    preds = model.predict(X)

    print('Accuracy:', accuracy_score(y, preds))
    print('Report:\n', classification_report(y, preds))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--test-data', required=True)
    args = parser.parse_args()
    evaluate(args.model, args.test_data)
