import argparse
import pandas as pd

def ingest(input_path, output_path):
    df = pd.read_csv(input_path)
    df.to_csv(f'{output_path}/raw_data.csv', index=False)
    print(f'Ingested data saved to {output_path}/raw_data.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    ingest(args.input, args.output)
