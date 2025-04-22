import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess(input_path, output_dir, test_size=0.2, random_state=42):
    # Determine the CSV file path
    if os.path.isdir(input_path):
        csv_files = [f for f in os.listdir(input_path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {input_path}")
        raw_csv = os.path.join(input_path, csv_files[0])
    else:
        raw_csv = input_path

    df = pd.read_csv(raw_csv)
    df = df.dropna()

    # Features and target
    X = df.drop('PE', axis=1)
    y = df['PE']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save processed train and test sets
    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_df['PE'] = y_train.values
    train_df.to_csv(os.path.join(output_dir, 'train_processed.csv'), index=False)

    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df['PE'] = y_test.values
    test_df.to_csv(os.path.join(output_dir, 'test_processed.csv'), index=False)

    # Save scaler
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    print(f"Saved train_processed.csv, test_processed.csv, and scaler.pkl in {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help="Input CSV file or directory")
    parser.add_argument('--output', required=True, help="Output directory for processed data")
    args = parser.parse_args()
    preprocess(args.input, args.output)
