import argparse
import pandas as pd
import os

def ingest(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    df = pd.read_csv(input_path)
    df.to_csv(os.path.join(output_path, 'raw_data.csv'), index=False)
    print(f'Ingested data saved to {os.path.join(output_path, "raw_data.csv")}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help="Input CSV file path")
    parser.add_argument('--output', required=True, help="Output directory for raw data")
    args = parser.parse_args()
    ingest(args.input, args.output)
