import argparse
import pandas as pd

def preprocess(input_dir, output_dir):
    df = pd.read_csv(f'{input_dir}/raw_data.csv')
    # Placeholder for actual preprocessing steps
    df = df.dropna()
    df.to_csv(f'{output_dir}/processed_data.csv', index=False)
    print(f'Processed data saved to {output_dir}/processed_data.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    preprocess(args.input, args.output)
